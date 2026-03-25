"""models/grounded_sam2.py
Grounding DINO (frozen) + SAM 2 (DoRA fine-tuned) pipeline.

Flow:
  images_pil + text prompts  →  DINO  →  bounding boxes
  boxes + image tensors       →  SAM2  →  binary masks

VRAM notes for RTX 4090 (24 GB):
  - DINO is eval-only, no gradients, minimal VRAM footprint
  - SAM2 uses gradient checkpointing (cfg: model.gradient_checkpointing: true)
  - bf16 precision — more stable than fp16 for transformer training on Ada
  - DoRA targets only q_proj / v_proj — peft tracks ~0.65% of params
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Sam2Model, Sam2Processor
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.common import get_logger, get_device, dtype_from_str

logger = get_logger("model")


# ─────────────────────────────────────────────────────────────
# DoRA config for SAM 2
# ─────────────────────────────────────────────────────────────

def build_dora_config(r=16, lora_alpha=32, lora_dropout=0.05, use_dora=True):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
        inference_mode=False,
    )


# ─────────────────────────────────────────────────────────────
# SAM 2 wrapper with DoRA
# ─────────────────────────────────────────────────────────────

class SAM2DoRA(nn.Module):
    def __init__(self, model_name, dtype, device, cfg):
        super().__init__()
        self.device = device
        self.dtype  = dtype

        logger.info(f"Loading SAM 2: {model_name}")
        self.processor = Sam2Processor.from_pretrained(model_name)
        base            = Sam2Model.from_pretrained(model_name, torch_dtype=dtype)

        # Freeze all
        for p in base.parameters():
            p.requires_grad_(False)

        # Apply DoRA
        dora_cfg   = build_dora_config(
            r=cfg["dora"]["r"],
            lora_alpha=cfg["dora"]["lora_alpha"],
            lora_dropout=cfg["dora"]["lora_dropout"],
            use_dora=cfg["dora"]["use_dora"],
        )
        self.sam = get_peft_model(base, dora_cfg)

        # Unfreeze mask decoder
        sam_base = self.sam.base_model
        if hasattr(sam_base, "mask_decoder"):
            for p in sam_base.mask_decoder.parameters():
                p.requires_grad_(True)

        # Always keep DoRA params trainable
        for name, p in self.sam.named_parameters():
            if "lora_" in name or "dora_" in name:
                p.requires_grad_(True)

        # Gradient checkpointing — reduces VRAM significantly in Phase 3
        if cfg["model"].get("gradient_checkpointing", True):
            if hasattr(self.sam, "gradient_checkpointing_enable"):
                self.sam.gradient_checkpointing_enable()
            elif hasattr(sam_base, "gradient_checkpointing_enable"):
                sam_base.gradient_checkpointing_enable()
            logger.info("  Gradient checkpointing: ON")

        self.sam = self.sam.to(device)

        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.sam.parameters())
        logger.info(f"  SAM2 trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    def forward(self, images, boxes, target_size=None):
        """
        images : [B, 3, H, W] normalised tensor
        boxes  : list of [x1, y1, x2, y2] or None per image
        """
        inputs  = self.processor(images=images, input_boxes=boxes, return_tensors="pt")
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.sam(**inputs)

        pred = outputs.pred_masks.squeeze(1)   # [B, num_masks, 256, 256]

        # Pick best mask by iou_scores
        if outputs.iou_scores is not None:
            best = outputs.iou_scores.argmax(dim=-1)
            pred = torch.stack([pred[b, best[b]] for b in range(pred.shape[0])])
        else:
            pred = pred[:, 0]

        pred = pred.unsqueeze(1)   # [B, 1, 256, 256]
        if target_size:
            pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
        return pred

    def save(self, path):
        self.sam.save_pretrained(str(path))
        self.processor.save_pretrained(str(path))

    def load_adapter(self, path):
        from peft import PeftModel
        self.sam = PeftModel.from_pretrained(self.sam.base_model, str(path)).to(self.device)


# ─────────────────────────────────────────────────────────────
# Grounding DINO wrapper (always frozen)
# ─────────────────────────────────────────────────────────────

class GroundingDINO(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.device    = device
        logger.info(f"Loading Grounding DINO: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)

        # Fully frozen — no gradients ever
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        logger.info("  DINO: frozen (eval only)")

    @torch.no_grad()
    def get_boxes(
        self,
        images_pil,              # list of PIL images
        prompts: List[str],      # one prompt per image
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        fallback_full_image: bool = True,
    ) -> List[Optional[List]]:
        """
        Returns list of boxes per image: [[x1,y1,x2,y2], ...] or None.
        If no box found and fallback_full_image=True, returns full-image box.
        """
        boxes_per_image = []

        for img_pil, prompt in zip(images_pil, prompts):
            w, h = img_pil.size
            inputs  = self.processor(images=img_pil, text=prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(h, w)],
            )[0]

            if len(results["boxes"]) > 0:
                # Take highest-confidence box
                best_idx = results["scores"].argmax().item()
                box      = results["boxes"][best_idx].tolist()   # [x1, y1, x2, y2]
                boxes_per_image.append([box])
            elif fallback_full_image:
                boxes_per_image.append([[0, 0, w, h]])
            else:
                boxes_per_image.append(None)

        return boxes_per_image


# ─────────────────────────────────────────────────────────────
# Combined pipeline
# ─────────────────────────────────────────────────────────────

class GroundedSAM2(nn.Module):
    """
    Full pipeline: text → DINO → box → SAM2 → mask.
    Only SAM2 (+ DoRA) is trained. DINO is always frozen.
    """

    def __init__(self, cfg: dict, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or get_device()
        self.dtype  = dtype_from_str(cfg["model"]["precision"])
        self.cfg    = cfg

        dino_name = cfg["model"]["grounding_dino"]
        sam2_name = cfg["model"]["sam2"]

        self.dino = GroundingDINO(dino_name, self.device)
        self.sam2 = SAM2DoRA(sam2_name, self.dtype, self.device, cfg)

        self.box_threshold  = cfg.get("dino", {}).get("box_threshold", 0.25)
        self.text_threshold = cfg.get("dino", {}).get("text_threshold", 0.25)
        self.fallback       = cfg.get("dino", {}).get("fallback_full_image", True)

    def forward(
        self,
        images:      torch.Tensor,    # [B, 3, H, W]
        images_pil:  list,            # list of PIL images
        prompts:     List[str],
        target_size: Optional[Tuple[int,int]] = None,
    ) -> torch.Tensor:
        # DINO always in eval, no grad
        with torch.no_grad():
            boxes = self.dino.get_boxes(
                images_pil, prompts,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                fallback_full_image=self.fallback,
            )

        # SAM 2 forward (with grad for training)
        return self.sam2(images, boxes, target_size=target_size)

    def train(self, mode=True):
        """Ensure DINO always stays in eval mode."""
        super().train(mode)
        self.dino.model.eval()
        return self

    def eval(self):
        super().eval()
        self.dino.model.eval()
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.sam2.save(path)
        logger.info(f"Saved → {path}")

    @classmethod
    def load(cls, checkpoint_path: str | Path, cfg: dict,
             device: Optional[torch.device] = None) -> "GroundedSAM2":
        dev     = device or get_device()
        wrapper = cls(cfg=cfg, device=dev)
        wrapper.sam2.load_adapter(checkpoint_path)
        return wrapper