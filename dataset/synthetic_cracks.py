from __future__ import annotations
import random
import math
import numpy as np
import cv2
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# Crack generator
# ─────────────────────────────────────────────────────────────

def _jitter_point(pt: Tuple[float, float], sigma: float) -> Tuple[int, int]:
    x = int(pt[0] + random.gauss(0, sigma))
    y = int(pt[1] + random.gauss(0, sigma))
    return x, y


def _draw_crack(
    image: np.ndarray,
    mask: np.ndarray,
    n_segments: int = 8,
    max_len_frac: float = 0.35,
    thickness: int = 1,
    jitter_sigma: float = 4.0,
    branch_prob: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws one crack polyline (possibly branching) on `image` and `mask`.
    image : H×W×3  uint8
    mask  : H×W    uint8  {0, 1}
    """
    h, w = image.shape[:2]

    # Start from a random edge-ish location
    x0 = random.randint(int(w * 0.1), int(w * 0.9))
    y0 = random.randint(int(h * 0.1), int(h * 0.9))

    # Random overall direction
    angle = random.uniform(0, 2 * math.pi)
    seg_len = min(h, w) * max_len_frac / n_segments

    pts = [(x0, y0)]
    x, y = float(x0), float(y0)
    for _ in range(n_segments):
        # Slight angle variation per segment
        angle += random.gauss(0, 0.35)
        dx = seg_len * math.cos(angle) + random.gauss(0, jitter_sigma)
        dy = seg_len * math.sin(angle) + random.gauss(0, jitter_sigma)
        x = np.clip(x + dx, 0, w - 1)
        y = np.clip(y + dy, 0, h - 1)
        pts.append(_jitter_point((x, y), jitter_sigma * 0.5))

    # Draw on mask (white = foreground)
    crack_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, len(pts)):
        cv2.line(crack_mask, pts[i - 1], pts[i], 1, thickness=thickness)

    # Optional branch
    if random.random() < branch_prob and len(pts) > 2:
        branch_start = random.choice(pts[len(pts) // 2 :])
        branch_angle = angle + random.uniform(math.pi / 6, math.pi / 3)
        bx, by = float(branch_start[0]), float(branch_start[1])
        b_pts = [branch_start]
        for _ in range(n_segments // 2):
            branch_angle += random.gauss(0, 0.3)
            bx = np.clip(bx + seg_len * math.cos(branch_angle), 0, w - 1)
            by = np.clip(by + seg_len * math.sin(branch_angle), 0, h - 1)
            b_pts.append(_jitter_point((bx, by), jitter_sigma * 0.5))
        for i in range(1, len(b_pts)):
            cv2.line(crack_mask, b_pts[i - 1], b_pts[i], 1, thickness=max(1, thickness - 1))

    # Blend crack colour: dark + noisy to mimic real cracks
    crack_bool = crack_mask > 0
    noise = np.random.randint(-15, 5, (h, w, 3), dtype=np.int16)
    darken = (image.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
    # Blur crack slightly
    darken_blur = cv2.GaussianBlur(darken, (3, 3), sigmaX=0.8)
    blended = image.copy()
    blended[crack_bool] = (
        blended[crack_bool].astype(np.float32) * 0.4
        + darken_blur[crack_bool].astype(np.float32) * 0.6
    ).clip(0, 255).astype(np.uint8)

    # Merge into existing mask
    updated_mask = np.maximum(mask, crack_mask)
    return blended, updated_mask


# ─────────────────────────────────────────────────────────────
# Albumentations-compatible transform wrapper
# ─────────────────────────────────────────────────────────────

class SyntheticCrackAugmentation:
    """
    Drop this into a custom albumentations pipeline via
    albumentations.Lambda or call directly.

    Usage in dataset:
        if label == "crack" and random.random() < synthetic_prob:
            image, mask = SyntheticCrackAugmentation()(image, mask)
    """

    def __init__(
        self,
        n_cracks: Tuple[int, int] = (1, 3),
        thickness_range: Tuple[int, int] = (1, 2),
    ):
        self.n_cracks = n_cracks
        self.thickness_range = thickness_range

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = random.randint(*self.n_cracks)
        for _ in range(n):
            t = random.randint(*self.thickness_range)
            image, mask = _draw_crack(image, mask, thickness=t)
        return image, mask