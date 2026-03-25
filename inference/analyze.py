"""inference/analyze.py
Phase 12 — Visual grid: [Image | GT | Pred]
Phase 13 — Failure analysis: thin cracks, shadows, low contrast
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.common import get_logger, ensure_dir, set_seed
from utils.metrics import dice_score
from utils.visualize import plot_triplet, plot_grid, plot_training_curves
from inference.predict import predict_single

logger = get_logger("analyze")


# ─────────────────────────────────────────────────────────────
# Sample collector
# ─────────────────────────────────────────────────────────────

def collect_samples(
    model,
    metadata_csv: str | Path,
    split: str = "val",
    n_per_class: int = 6,
    image_size: int = 768,
    threshold: float = 0.5,
    cfg: Optional[dict] = None,
) -> list[dict]:
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == split].reset_index(drop=True)

    samples = []
    for label in ["crack", "taping"]:
        sub = df[df["type"] == label].sample(
            min(n_per_class, len(df[df["type"] == label])),
            random_state=42,
        )
        for _, row in sub.iterrows():
            img_bgr = cv2.imread(str(row["image_path"]))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            gt = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue

            pred_mask, _ = predict_single(
                model, row["image_path"], label,
                inference_cfg=cfg or {},
                threshold=threshold,
            )

            # Compute Dice
            gt_t = torch.from_numpy((gt > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            pred_t = torch.from_numpy((pred_mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            if pred_t.shape != gt_t.shape:
                pred_t = nn.functional.interpolate(pred_t.float(), gt_t.shape[-2:])
            logit = (pred_t * 2) - 1
            d = dice_score(logit, gt_t).item()

            # Resize image to match mask for display
            h, w = gt.shape[:2]
            img_display = cv2.resize(img_rgb, (w, h))

            samples.append({
                "image": img_display,
                "gt_mask": gt,
                "pred_mask": pred_mask if pred_mask.shape == gt.shape
                    else cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST),
                "label": label,
                "dice": d,
                "path": str(row["image_path"]),
            })

    return samples


# ─────────────────────────────────────────────────────────────
# Phase 12 — Visual grid
# ─────────────────────────────────────────────────────────────

def generate_visuals(
    model,
    metadata_csv: str | Path,
    viz_dir: str | Path,
    split: str = "val",
    n_per_class: int = 4,
    cfg: Optional[dict] = None,
) -> None:
    viz_dir = ensure_dir(viz_dir)
    image_size = cfg["data"]["image_size"] if cfg else 768
    threshold = cfg["training"]["mask_threshold"] if cfg else 0.5

    logger.info("Collecting samples for visualisation …")
    samples = collect_samples(
        model=model,
        metadata_csv=metadata_csv,
        split=split,
        n_per_class=n_per_class,
        image_size=image_size,
        threshold=threshold,
        cfg=cfg,
    )

    if not samples:
        logger.warning("No samples collected — skipping visualisation")
        return

    # Grid
    plot_grid(samples, save_path=viz_dir / "visual_grid.png", cols=4)
    logger.info(f"Grid → {viz_dir / 'visual_grid.png'}")

    # Individual triplets (best + worst by Dice)
    samples_sorted = sorted(samples, key=lambda s: s["dice"])
    for tag, s in [("worst", samples_sorted[0]), ("best", samples_sorted[-1])]:
        plot_triplet(
            s["image"], s["gt_mask"], s["pred_mask"],
            title=f"{tag.upper()} — {s['label']}  Dice={s['dice']:.3f}",
            save_path=viz_dir / f"triplet_{tag}_{s['label']}.png",
        )


# ─────────────────────────────────────────────────────────────
# Phase 13 — Failure analysis
# ─────────────────────────────────────────────────────────────

def analyze_failures(
    model,
    metadata_csv: str | Path,
    out_path: str | Path = "outputs/failure_analysis.json",
    split: str = "val",
    threshold: float = 0.5,
    dice_fail_threshold: float = 0.40,
    cfg: Optional[dict] = None,
) -> dict:
    """
    Categorises low-Dice samples into failure modes:
    - thin_crack    : GT coverage < 1% of image area
    - low_contrast  : std(image) < 30
    - shadow_noise  : high-frequency noise in error region
    """
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == split].reset_index(drop=True)
    image_size = cfg["data"]["image_size"] if cfg else 768

    failures = {"thin_crack": [], "low_contrast": [], "shadow_noise": [], "other": []}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Failure analysis"):
        label = row["type"]
        pred_mask, _ = predict_single(
            model, row["image_path"], label,
            inference_cfg=cfg or {},
            threshold=threshold,
        )
        gt = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue

        # Compute Dice
        gt_t = torch.from_numpy((gt > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        p = cv2.resize(pred_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_t = torch.from_numpy((p > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        d = dice_score((pred_t * 2) - 1, gt_t).item()

        if d >= dice_fail_threshold:
            continue

        # Categorise failure
        img = cv2.imread(str(row["image_path"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
        gt_coverage = (gt > 127).mean()
        img_std = gray.std() if gray is not None else 999

        info = {"path": str(row["image_path"]), "label": label, "dice": round(d, 3)}

        if label == "crack" and gt_coverage < 0.01:
            failures["thin_crack"].append(info)
        elif img_std < 30:
            failures["low_contrast"].append(info)
        else:
            # Check error region noise
            err_region = np.abs(pred_t.numpy()[0, 0] - gt_t.numpy()[0, 0])
            if err_region.std() > 0.4:
                failures["shadow_noise"].append(info)
            else:
                failures["other"].append(info)

    summary = {k: len(v) for k, v in failures.items()}
    logger.info(f"Failure summary: {summary}")

    result = {"summary": summary, "details": failures}
    ensure_dir(Path(out_path).parent)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Failure analysis → {out_path}")
    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", default="dataset/metadata.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n_per_class", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    from models.sam3_dora import SAM3DoRA
    from utils.common import get_device
    model = SAM3DoRA.load(
        adapter_path=args.checkpoint,
        base_model=cfg["model"]["name"],
        precision=cfg["model"]["precision"],
        device=get_device(),
    )

    generate_visuals(model, args.metadata, cfg["output"]["viz_dir"], args.split, args.n_per_class, cfg)
    analyze_failures(model, args.metadata, "outputs/failure_analysis.json", args.split, cfg=cfg)