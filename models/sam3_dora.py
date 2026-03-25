"""models/sam3_dora.py
Phase 6 — Load facebook/sam3 in bfloat16
Phase 7 — Wrap with DoRA adapters via PEFT
Exposes a forward() that accepts (pixel_values, text_prompts) → pred_masks
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Sam3Processor, Sam3Model
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.common import get_logger, get_device, dtype_from_str

logger = get_logger("model")


# ─────────────────────────────────────────────────────────────
# DoRA config builder
# ─────────────────────────────────────────────────────────────

def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_dora: bool = True,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    if target_modules is None:
        # Covers image encoder attention + mask decoder transformer layers
        target_modules = [
            "q_proj",
            "v_proj",
            # SAM3 mask decoder attention keys
            "transformer.layers.0.self_attn.q_proj",
            "transformer.layers.0.self_attn.v_proj",
            "transformer.layers.1.self_attn.q_proj",
            "transformer.layers.1.self_attn.v_proj",
        ]
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        target_modules=target_modules,
        inference_mode=False,
        # SAM3 is an encoder-decoder; use FEATURE_EXTRACTION task type
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
    )


# ─────────────────────────────────────────────────────────────
# Wrapped SAM3 model
# ─────────────────────────────────────────────────────────────

class SAM3DoRA(nn.Module):
    """
    SAM3 fine-tuned with DoRA adapters.
    Only the adapter parameters + mask decoder are trainable.
    Image encoder is frozen (except DoRA deltas).
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        precision: str = "bfloat16",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_dora: bool = True,
        target_modules: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.dtype = dtype_from_str(precision)

        logger.info(f"Loading {model_name} …")
        self.processor = Sam3Processor.from_pretrained(model_name)
        base_model = Sam3Model.from_pretrained(
            model_name, torch_dtype=self.dtype
        )

        # ── Freeze entire model first ──────────────────────────
        for p in base_model.parameters():
            p.requires_grad_(False)

        # ── Apply DoRA adapters (trainable) ───────────────────
        lora_cfg = build_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            target_modules=target_modules,
        )
        self.sam3 = get_peft_model(base_model, lora_cfg)

        # ── Also unfreeze mask decoder fully ──────────────────
        if hasattr(self.sam3.base_model, "mask_decoder"):
            for p in self.sam3.base_model.mask_decoder.parameters():
                p.requires_grad_(True)

        self.sam3 = self.sam3.to(self.device)

        trainable = sum(p.numel() for p in self.sam3.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam3.parameters())
        logger.info(
            f"Trainable params: {trainable:,} / {total:,}  "
            f"({100 * trainable / total:.2f}%)"
        )

    # ─────────────────────────────────────────────────────────
    # Prepare inputs from raw tensors + text prompts
    # ─────────────────────────────────────────────────────────

    def prepare_inputs(
        self,
        images: torch.Tensor,  # [B, 3, H, W] — already normalised
        prompts: List[str],
    ) -> Dict:
        """
        Re-encode using processor (handles text embedding alignment).
        Returns input dict on self.device.
        """
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ─────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """Returns best mask [B, 1, H, W] (raw logits)."""
        inputs = self.prepare_inputs(images, prompts)
        outputs = self.sam3(**inputs)
        return outputs.pred_masks[:, :1, :, :]

    def forward_all_masks(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Returns ALL mask outputs [B, K, H, W] for multi-mask supervision.
        The trainer picks the best K masks via topk_mask_loss().
        """
        inputs = self.prepare_inputs(images, prompts)
        outputs = self.sam3(**inputs)
        return outputs.pred_masks  # [B, num_queries, H, W]

    # ─────────────────────────────────────────────────────────
    # Save / load
    # ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        self.sam3.save_pretrained(str(path))
        self.processor.save_pretrained(str(path))
        logger.info(f"Saved adapter weights → {path}")

    @classmethod
    def load(
        cls,
        adapter_path: str | Path,
        base_model: str = "facebook/sam3",
        precision: str = "bfloat16",
        device: Optional[torch.device] = None,
    ) -> "SAM3DoRA":
        """Reload from saved adapter checkpoint."""
        from peft import PeftModel

        dev = device or get_device()
        dtype = dtype_from_str(precision)
        processor = Sam3Processor.from_pretrained(str(adapter_path))
        base = Sam3Model.from_pretrained(base_model, torch_dtype=dtype)
        peft_model = PeftModel.from_pretrained(base, str(adapter_path))
        peft_model = peft_model.to(dev)

        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.device = dev
        wrapper.dtype = dtype
        wrapper.processor = processor
        wrapper.sam3 = peft_model
        return wrapper