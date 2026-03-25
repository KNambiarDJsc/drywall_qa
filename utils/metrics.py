"""utils/metrics.py — Dice score + mIoU, computed per-batch and per-epoch."""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
# Core metric functions (work on torch tensors or numpy arrays)
# ─────────────────────────────────────────────────────────────

def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        pred:   [B, 1, H, W] float logits or probabilities
        target: [B, 1, H, W] float binary masks (0 or 1)
    Returns:
        Scalar mean Dice across the batch.
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def miou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Mean IoU (binary segmentation — foreground class only)."""
    pred = (torch.sigmoid(pred) > threshold).float()
    pred_flat = pred.view(pred.size(0), -1)
    tgt_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim=1)
    union = (pred_flat + tgt_flat).clamp(0, 1).sum(dim=1)
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# ─────────────────────────────────────────────────────────────
# Per-class accumulator
# ─────────────────────────────────────────────────────────────

class MetricAccumulator:
    """Accumulates Dice + mIoU per class over an epoch."""

    def __init__(self, classes: List[str] = ("crack", "taping")):
        self.classes = classes
        self._reset()

    def _reset(self):
        self.totals: Dict[str, Dict[str, float]] = {
            c: {"dice": 0.0, "miou": 0.0, "count": 0} for c in self.classes
        }

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        label: str,
        threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """Update accumulator for a single batch. Returns (dice, miou) for this batch."""
        d = dice_score(pred, target, threshold).item()
        m = miou_score(pred, target, threshold).item()
        self.totals[label]["dice"] += d
        self.totals[label]["miou"] += m
        self.totals[label]["count"] += 1
        return d, m

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Return per-class and macro-averaged metrics."""
        results = {}
        all_dice, all_miou = [], []
        for c in self.classes:
            n = self.totals[c]["count"]
            if n == 0:
                results[c] = {"dice": 0.0, "miou": 0.0}
                continue
            d = self.totals[c]["dice"] / n
            m = self.totals[c]["miou"] / n
            results[c] = {"dice": d, "miou": m}
            all_dice.append(d)
            all_miou.append(m)
        results["macro"] = {
            "dice": float(np.mean(all_dice)) if all_dice else 0.0,
            "miou": float(np.mean(all_miou)) if all_miou else 0.0,
        }
        self._reset()
        return results

    def summary_str(self) -> str:
        r = self.compute()
        lines = ["── Metrics ─────────────────────────────"]
        for k, v in r.items():
            lines.append(f"  {k:10s}  Dice={v['dice']:.4f}  mIoU={v['miou']:.4f}")
        return "\n".join(lines)