"""utils/visualize.py — side-by-side [Image | GT | Pred] grids."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union
import torch


# ─────────────────────────────────────────────────────────────
# Single triplet
# ─────────────────────────────────────────────────────────────

def plot_triplet(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str = "",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """
    image:     H×W×3 uint8
    gt_mask:   H×W uint8 {0,255}
    pred_mask: H×W uint8 {0,255}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="Reds", vmin=0, vmax=255)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Overlay pred on image
    overlay = image.copy()
    pred_bool = pred_mask > 127
    overlay[pred_bool] = (
        overlay[pred_bool] * 0.45 + np.array([255, 80, 80]) * 0.55
    ).clip(0, 255).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Prediction (overlay)")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Grid of N triplets
# ─────────────────────────────────────────────────────────────

def plot_grid(
    samples: List[dict],  # each: {image, gt_mask, pred_mask, label, dice}
    save_path: Union[str, Path],
    cols: int = 4,
) -> None:
    n = len(samples)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 4, rows * 9))
    axes = np.array(axes).reshape(rows * 3, cols)

    for idx, s in enumerate(samples):
        row_base = (idx // cols) * 3
        col = idx % cols
        _show(axes[row_base][col], s["image"], f"{s['label']}")
        _show(axes[row_base + 1][col], s["gt_mask"], "GT", cmap="Reds")
        _show(axes[row_base + 2][col], s["pred_mask"], f"Pred  Dice={s.get('dice',0):.3f}", cmap="Blues")

    # hide unused
    for idx in range(n, rows * cols):
        row_base = (idx // cols) * 3
        col = idx % cols
        for r in range(3):
            axes[row_base + r][col].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _show(ax, img, title="", cmap=None):
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(
    history: dict,  # {"train_loss": [], "val_dice_crack": [], ...}
    save_path: Union[str, Path],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves")

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Dice
    for k in history:
        if "dice" in k:
            axes[1].plot(history[k], label=k)
    axes[1].set_title("Dice")
    axes[1].legend()

    # mIoU
    for k in history:
        if "miou" in k:
            axes[2].plot(history[k], label=k)
    axes[2].set_title("mIoU")
    axes[2].legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)