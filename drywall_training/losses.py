"""training/losses.py
Phase 8 — Full loss suite:
  FocalLoss        — handles class imbalance, alpha tuned for tiny cracks
  TverskyLoss      — replaces Dice; alpha=0.7 penalises FN hard (great for cracks)
  BoundaryLoss     — ON by default; sharpens crack edges
  HybridLoss       — 0.35 Focal + 0.55 Tversky + 0.10 Boundary
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


# ─────────────────────────────────────────────────────────────
# Focal
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.50, gamma: float = 2.0):
        """alpha=0.50 (up from 0.25) — stops model ignoring tiny crack pixels."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.exp(-bce)
        return (self.alpha * (1 - p_t) ** self.gamma * bce).mean()


# ─────────────────────────────────────────────────────────────
# Tversky  (replaces Dice)
# ─────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        """
        alpha > beta  →  penalises false negatives harder than false positives.
        Perfect for thin cracks where the model prefers to predict nothing.
        """
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        flat_p = prob.view(prob.size(0), -1)
        flat_t = target.view(target.size(0), -1)
        tp = (flat_p * flat_t).sum(dim=1)
        fp = (flat_p * (1 - flat_t)).sum(dim=1)
        fn = ((1 - flat_p) * flat_t).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky).mean()


# ─────────────────────────────────────────────────────────────
# Boundary
# ─────────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        dist_maps = self._distance_maps(target)
        return (prob * dist_maps).mean()

    @staticmethod
    def _distance_maps(target: torch.Tensor) -> torch.Tensor:
        cpu = target.detach().cpu().numpy()
        B = cpu.shape[0]
        dist = np.zeros_like(cpu, dtype=np.float32)
        for b in range(B):
            m = cpu[b, 0]
            if m.sum() == 0 or m.sum() == m.size:
                continue
            d_fg = distance_transform_edt(m)
            d_bg = distance_transform_edt(1 - m)
            sd = d_bg - d_fg
            max_val = np.abs(sd).max() + 1e-6
            dist[b, 0] = sd / max_val
        return torch.from_numpy(dist).to(target.device).to(target.dtype)


# ─────────────────────────────────────────────────────────────
# Hybrid  (winning config)
# ─────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Default: 0.35 Focal + 0.55 Tversky + 0.10 Boundary
    Weights must sum to 1.0.
    """

    def __init__(
        self,
        focal_weight:    float = 0.35,
        tversky_weight:  float = 0.55,
        boundary_weight: float = 0.10,
        focal_alpha:     float = 0.50,
        focal_gamma:     float = 2.0,
        tversky_alpha:   float = 0.70,
        tversky_beta:    float = 0.30,
    ):
        super().__init__()
        import warnings
        total = focal_weight + tversky_weight + boundary_weight
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Loss weights must sum to 1.0, got {total:.4f}")
        elif abs(total - 1.0) > 1e-6:
            # renormalise silently for float precision drift
            focal_weight    = focal_weight    / total
            tversky_weight  = tversky_weight  / total
            boundary_weight = boundary_weight / total
        self.fw = focal_weight
        self.tw = tversky_weight
        self.bw = boundary_weight
        self.focal    = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky  = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary = BoundaryLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        fl = self.focal(pred, target)
        tl = self.tversky(pred, target)
        bl = self.boundary(pred, target) if self.bw > 0 else torch.tensor(0.0, device=pred.device)

        loss = self.fw * fl + self.tw * tl + self.bw * bl
        breakdown = {
            "focal": fl.item(), "tversky": tl.item(),
            "boundary": bl.item(), "total": loss.item(),
        }
        return loss, breakdown