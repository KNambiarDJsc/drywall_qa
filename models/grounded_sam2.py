"""models/grounded_sam2.py
Grounded SAM 2 pipeline:
  1. Grounding DINO (frozen) — text prompt → bounding box
  2. SAM 2 (DoRA fine-tuned) — bounding box → binary mask

Text prompts come back fully:
  "crack", "wall crack", "drywall seam", "taping area" etc.

Grounding DINO is NEVER trained — it's a frozen feature extractor.
SAM 2 mask decoder + image encoder (last N blocks via DoRA) are trained.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
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
        target_modules=["q_proj", "v_proj", "query", "value"],
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
        inference_mode=False,
    )


# ─────────────────────────────────────────────────────────────
# Grounding DINO wrapper (frozen)
# ─────────────────────────────────────────────────────────────

class GroundingDINO(nn.Module):
    """Frozen text → bbox detector. Never trained."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: torch.device = None):
        super().__init__()
        self.device    = device or get_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        # Fully frozen
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        logger.info(f"Grounding DINO loaded ({model_id}) — frozen")

    @torch.no_grad()
    def get_boxes(
        self,
        images_pil: list,          # list of PIL Images
        text_prompts: List[str],   # one per image e.g. ["crack", "drywall seam"]
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        fallback_full_image: bool = True,
    ) -> List[np.ndarray]:
        """
        Returns list of bounding boxes, one array per image.
        Each array: [N, 4] in xyxy pixel format.
        Falls back to full-image box if no detection.
        """
        # DINO needs text ending with "." 
        texts = [p.rstrip(".") + "." for p in text_prompts]

        inputs = self.processor(
            images=images_pil,
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)

        boxes_per_image = []
        for i, (img, prompt) in enumerate(zip(images_pil, texts)):
            W, H = img.size
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids[i:i+1],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(H, W)],
            )[0]

            boxes = results["boxes"].cpu().numpy()   # [N, 4] xyxy

            if len(boxes) == 0 and fallback_full_image:
                # No detection → use full image as box
                boxes = np.array([[0, 0, W, H]], dtype=np.float32)
            elif len(boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)

            boxes_per_image.append(boxes)

        return boxes_per_image


# ─────────────────────────────────────────────────────────────
# SAM 2 wrapper with DoRA
# ─────────────────────────────────────────────────────────────

class SAM2DoRA(nn.Module):
    """SAM 2 fine-tuned with DoRA. Accepts bounding boxes as prompts."""

    def __init__(
        self,
        model_id:     str   = "facebook/sam2-hiera-large",
        precision:    str   = "fp16",
        lora_r:       int   = 16,
        lora_alpha:   int   = 32,
        lora_dropout: float = 0.05,
        use_dora:     bool  = True,
        device:       Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype  = dtype_from_str(precision)

        logger.info(f"Loading SAM 2 ({model_id}) …")

        # SAM 2 via transformers
        from transformers import Sam2Model, Sam2Processor
        self.processor = Sam2Processor.from_pretrained(model_id)
        base           = Sam2Model.from_pretrained(model_id, torch_dtype=self.dtype)

        # Freeze everything
        for p in base.parameters():
            p.requires_grad_(False)

        # DoRA adapters
        cfg      = build_dora_config(lora_r, lora_alpha, lora_dropout, use_dora)
        self.sam = get_peft_model(base, cfg)

        # Unfreeze mask decoder
        sam_base = self.sam.base_model
        for attr in ["mask_decoder", "sam_mask_decoder"]:
            decoder = getattr(sam_base, attr, None)
            if decoder:
                for p in decoder.parameters():
                    p.requires_grad_(True)
                break

        # Always keep DoRA params trainable
        for name, p in self.sam.named_parameters():
            if "lora_" in name or "dora_" in name:
                p.requires_grad_(True)

        self.sam = self.sam.to(self.device)

        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.sam.parameters())
        logger.info(f"SAM 2 trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def forward(
        self,
        images:      torch.Tensor,      # [B, 3, H, W]
        input_boxes: torch.Tensor,      # [B, 1, 4]  xyxy normalised 0-1
        target_size: Optional[Tuple[int,int]] = None,
    ) -> torch.Tensor:
        """Returns pred_masks [B, 1, H, W] raw logits."""
        inputs  = self.processor(
            images=images,
            input_boxes=input_boxes.tolist(),
            return_tensors="pt",
        )
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.sam(**inputs)

        pred = outputs.pred_masks.squeeze(1)           # [B, num_masks, H, W]
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            best = outputs.iou_scores.argmax(dim=-1)
            pred = torch.stack([pred[b, best[b]] for b in range(pred.shape[0])])
        else:
            pred = pred[:, 0]

        pred = pred.unsqueeze(1)   # [B, 1, H, W]
        if target_size:
            pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
        return pred

    def save(self, path: str | Path) -> None:
        self.sam.save_pretrained(str(path))
        self.processor.save_pretrained(str(path))
        logger.info(f"SAM 2 adapter saved → {path}")

    @classmethod
    def load(cls, adapter_path, base_model="facebook/sam2-hiera-large",
             precision="fp16", device=None):
        from peft import PeftModel
        from transformers import Sam2Model, Sam2Processor
        dev  = device or get_device()
        dtype = dtype_from_str(precision)
        proc = Sam2Processor.from_pretrained(str(adapter_path))
        base = Sam2Model.from_pretrained(base_model, torch_dtype=dtype)
        sam  = PeftModel.from_pretrained(base, str(adapter_path)).to(dev)

        wrapper           = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.device    = dev
        wrapper.dtype     = dtype
        wrapper.processor = proc
        wrapper.sam       = sam
        return wrapper


# ─────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────

class GroundedSAM2(nn.Module):
    """
    Full pipeline:
      text prompt → DINO (frozen) → bbox → SAM 2 (DoRA) → mask
    """

    def __init__(self, cfg: dict, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or get_device()
        self.dtype  = dtype_from_str(cfg["model"]["precision"])
        self.cfg    = cfg

        # Grounding DINO — frozen
        self.dino = GroundingDINO(
            model_id=cfg["model"]["grounding_dino"],
            device=self.device,
        )

        # SAM 2 — DoRA fine-tuned
        self.sam2 = SAM2DoRA(
            model_id=cfg["model"]["sam2"],
            precision=cfg["model"]["precision"],
            lora_r=cfg["dora"]["r"],
            lora_alpha=cfg["dora"]["lora_alpha"],
            lora_dropout=cfg["dora"]["lora_dropout"],
            use_dora=cfg["dora"]["use_dora"],
            device=self.device,
        )

    def encode_boxes(
        self,
        images_pil: list,
        text_prompts: List[str],
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Run DINO on PIL images, return normalised boxes [B, 1, 4].
        Takes the highest-confidence box per image.
        """
        dino_cfg  = self.cfg.get("dino", {})
        boxes_raw = self.dino.get_boxes(
            images_pil, text_prompts,
            box_threshold=dino_cfg.get("box_threshold", 0.25),
            text_threshold=dino_cfg.get("text_threshold", 0.25),
            fallback_full_image=dino_cfg.get("fallback_full_image", True),
        )

        H, W = image_hw
        boxes_norm = []
        for boxes in boxes_raw:
            if len(boxes) == 0:
                b = np.array([[0., 0., 1., 1.]], dtype=np.float32)
            else:
                b = boxes[0:1].copy()              # take highest-conf box
                b[:, [0, 2]] /= W
                b[:, [1, 3]] /= H
            boxes_norm.append(b)

        return torch.tensor(
            np.stack(boxes_norm), dtype=torch.float32
        ).unsqueeze(1).to(self.device)             # [B, 1, 4]

    def forward(
        self,
        images:        torch.Tensor,    # [B, 3, H, W]  normalised tensors
        images_pil:    list,            # PIL images for DINO
        text_prompts:  List[str],       # one per image
        target_size:   Optional[Tuple[int,int]] = None,
    ) -> torch.Tensor:
        H, W = images.shape[-2:]
        boxes = self.encode_boxes(images_pil, text_prompts, (H, W))
        return self.sam2(images, boxes, target_size=target_size or (H, W))

    def save(self, path: str | Path) -> None:
        self.sam2.save(path)

    @classmethod
    def load(cls, adapter_path, cfg: dict, device=None):
        obj       = cls.__new__(cls)
        nn.Module.__init__(obj)
        dev       = device or get_device()
        obj.device = dev
        obj.dtype  = dtype_from_str(cfg["model"]["precision"])
        obj.cfg    = cfg
        obj.dino   = GroundingDINO(cfg["model"]["grounding_dino"], dev)
        obj.sam2   = SAM2DoRA.load(adapter_path, cfg["model"]["sam2"],
                                   cfg["model"]["precision"], dev)
        return obj