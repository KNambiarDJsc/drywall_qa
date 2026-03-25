"""models/sam_dora.py
SAM ViT-H (facebook/sam-vit-huge) + DoRA adapters.

Replaces sam3_dora.py. Key differences from SAM3:
  - No text prompts — uses points or boxes as prompts
  - SamModel + SamProcessor (not Sam3*)
  - Post-processing via processor.image_processor.post_process_masks()
  - forward() accepts point prompts derived from GT mask centroids
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.common import get_logger, get_device, dtype_from_str

logger = get_logger("model")


# ─────────────────────────────────────────────────────────────
# DoRA config
# ─────────────────────────────────────────────────────────────

def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_dora: bool = True,
) -> LoraConfig:
    # SAM ViT-H attention projection layers
    target_modules = [
        "q_proj", "v_proj",
        "query", "value",          # mask decoder attention
    ]
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
        inference_mode=False,
    )


# ─────────────────────────────────────────────────────────────
# Point prompt helpers
# ─────────────────────────────────────────────────────────────

def mask_to_point_prompt(
    mask: torch.Tensor,       # [1, H, W] float {0,1}
    n_points: int = 1,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Derive positive point prompt(s) from GT mask.
    Returns:
        input_points : [1, 1, n_points, 2]  (x, y) in pixel coords
        input_labels : [1, 1, n_points]      all 1 (positive)
    """
    import numpy as np
    m = mask.squeeze().cpu().numpy()
    fg = np.argwhere(m > 0.5)          # (N, 2) → (row, col)

    if len(fg) == 0:
        # Empty mask — use image centre
        h, w = m.shape
        pts = [[w // 2, h // 2]]
    else:
        # Sample n_points from foreground
        idx = np.random.choice(len(fg), size=min(n_points, len(fg)), replace=False)
        pts = [[int(fg[i][1]), int(fg[i][0])] for i in idx]   # col=x, row=y

    pts_tensor    = torch.tensor([pts], dtype=torch.float32)   # [1, n_points, 2]
    pts_tensor    = pts_tensor.unsqueeze(0)                     # [1, 1, n_points, 2]
    labels_tensor = torch.ones(1, 1, len(pts), dtype=torch.long)

    if device:
        pts_tensor    = pts_tensor.to(device)
        labels_tensor = labels_tensor.to(device)

    return pts_tensor, labels_tensor


# ─────────────────────────────────────────────────────────────
# SAM + DoRA wrapper
# ─────────────────────────────────────────────────────────────

class SAMDoRA(nn.Module):
    """
    SAM ViT-H with DoRA adapters.
    Mask decoder is fully unfrozen.
    Image encoder frozen except DoRA deltas.
    """

    def __init__(
        self,
        model_name:   str = "facebook/sam-vit-huge",
        precision:    str = "bfloat16",
        lora_r:       int = 16,
        lora_alpha:   int = 32,
        lora_dropout: float = 0.05,
        use_dora:     bool = True,
        device:       Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype  = dtype_from_str(precision)

        logger.info(f"Loading {model_name} …")
        self.processor = SamProcessor.from_pretrained(model_name)
        base_model     = SamModel.from_pretrained(model_name, torch_dtype=self.dtype)

        # Freeze everything first
        for p in base_model.parameters():
            p.requires_grad_(False)

        # Apply DoRA
        lora_cfg  = build_lora_config(lora_r, lora_alpha, lora_dropout, use_dora)
        self.sam  = get_peft_model(base_model, lora_cfg)

        # Unfreeze mask decoder fully
        if hasattr(self.sam.base_model, "mask_decoder"):
            for p in self.sam.base_model.mask_decoder.parameters():
                p.requires_grad_(True)

        # Always keep DoRA params trainable
        for name, p in self.sam.named_parameters():
            if "lora_" in name or "dora_" in name:
                p.requires_grad_(True)

        self.sam = self.sam.to(self.device)

        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.sam.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    # ─────────────────────────────────────────────────────────
    # Prepare processor inputs
    # ─────────────────────────────────────────────────────────

    def prepare_inputs(
        self,
        images: torch.Tensor,               # [B, 3, H, W] normalised
        input_points: torch.Tensor,          # [B, 1, N, 2]
        input_labels: torch.Tensor,          # [B, 1, N]
    ) -> dict:
        """
        SAM processor expects PIL images or raw arrays for sizing.
        We pass pre-normalised tensors directly — processor handles
        resizing internally to 1024×1024.
        """
        inputs = self.processor(
            images=images,
            input_points=input_points.tolist(),
            input_labels=input_labels.tolist(),
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ─────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        images:        torch.Tensor,   # [B, 3, H, W]
        input_points:  torch.Tensor,   # [B, 1, N, 2]
        input_labels:  torch.Tensor,   # [B, 1, N]
        target_size:   Optional[Tuple[int,int]] = None,
    ) -> torch.Tensor:
        """
        Returns pred_masks [B, 1, H, W] resized to target_size (or 256×256).
        Raw logits — apply sigmoid + threshold downstream.
        """
        inputs  = self.prepare_inputs(images, input_points, input_labels)
        outputs = self.sam(**inputs)

        # outputs.pred_masks: [B, 1, num_masks, 256, 256]
        # Take highest IOU mask (index 0 after sorting by iou_scores)
        pred = outputs.pred_masks.squeeze(1)          # [B, num_masks, 256, 256]

        # Pick best mask per sample using iou_scores
        if outputs.iou_scores is not None:
            best_idx = outputs.iou_scores.argmax(dim=-1)  # [B]
            pred = torch.stack([pred[b, best_idx[b]] for b in range(pred.shape[0])])
        else:
            pred = pred[:, 0]   # fallback: first mask

        pred = pred.unsqueeze(1)   # [B, 1, 256, 256]

        if target_size is not None:
            pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)

        return pred

    def forward_all_masks(
        self,
        images:       torch.Tensor,
        input_points: torch.Tensor,
        input_labels: torch.Tensor,
        target_size:  Optional[Tuple[int,int]] = None,
    ) -> torch.Tensor:
        """Returns all mask proposals [B, num_masks, H, W] for multi-mask supervision."""
        inputs  = self.prepare_inputs(images, input_points, input_labels)
        outputs = self.sam(**inputs)

        pred = outputs.pred_masks.squeeze(1)   # [B, num_masks, 256, 256]
        if target_size is not None:
            B, K, _, _ = pred.shape
            pred = F.interpolate(
                pred.view(B*K, 1, 256, 256),
                size=target_size, mode="bilinear", align_corners=False
            ).view(B, K, *target_size)
        return pred

    # ─────────────────────────────────────────────────────────
    # Save / load
    # ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        self.sam.save_pretrained(str(path))
        self.processor.save_pretrained(str(path))
        logger.info(f"Saved → {path}")

    @classmethod
    def load(
        cls,
        adapter_path: str | Path,
        base_model:   str = "facebook/sam-vit-huge",
        precision:    str = "bfloat16",
        device:       Optional[torch.device] = None,
    ) -> "SAMDoRA":
        from peft import PeftModel
        dev   = device or get_device()
        dtype = dtype_from_str(precision)

        processor  = SamProcessor.from_pretrained(str(adapter_path))
        base       = SamModel.from_pretrained(base_model, torch_dtype=dtype)
        peft_model = PeftModel.from_pretrained(base, str(adapter_path)).to(dev)

        wrapper            = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.device     = dev
        wrapper.dtype      = dtype
        wrapper.processor  = processor
        wrapper.sam        = peft_model
        return wrapper