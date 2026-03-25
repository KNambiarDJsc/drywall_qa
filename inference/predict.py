"""inference/predict.py — SAM ViT-H inference.
Uses center-point prompt (no text). TTA + multi-scale + morph unchanged.
"""

from __future__ import annotations
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[1]))
from models.sam_dora import SAMDoRA
from utils.common import (get_logger, ensure_dir, set_seed,
                           get_device, to_binary_mask, resize_mask_to_original)
from utils.metrics import MetricAccumulator, dice_score

logger = get_logger("inference")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _preprocess(img_rgb: np.ndarray, size: int) -> torch.Tensor:
    resized = cv2.resize(img_rgb, (size, size))
    norm    = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)


def _center_point_prompt(size: int):
    """Use image centre as the point prompt at inference (no GT available)."""
    cx, cy = size // 2, size // 2
    pts    = torch.tensor([[[[cx, cy]]]], dtype=torch.float32)   # [1, 1, 1, 2]
    labels = torch.tensor([[[1]]], dtype=torch.long)             # [1, 1, 1]
    return pts, labels


def _morph_close(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)


@torch.no_grad()
def _forward(model, img_tensor, pts, labels, size):
    inp    = img_tensor.to(model.device, dtype=model.dtype)
    pts    = pts.to(model.device)
    labels = labels.to(model.device)
    pred   = model(inp, pts, labels, target_size=(size, size))
    return torch.sigmoid(pred).squeeze().cpu().float().numpy()


def predict_single(
    model: SAMDoRA,
    image_path: str | Path,
    label: str,
    inference_cfg: dict = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, float]:
    inference_cfg = inference_cfg or {}
    inf_block     = inference_cfg.get("inference", {})
    scales        = inf_block.get("multiscale", [768])
    tta           = inf_block.get("tta_enabled", True)
    do_close      = inf_block.get("morph_close_crack", True) and label == "crack"
    k_size        = inf_block.get("morph_kernel_size", 3)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    t0        = time.time()
    all_probs = []

    model.eval()
    for size in scales:
        pts, labels_t = _center_point_prompt(size)
        img_tensor    = _preprocess(img_rgb, size)
        prob          = _forward(model, img_tensor, pts, labels_t, size)
        all_probs.append(prob)

        if tta:
            img_flipped = _preprocess(img_rgb[:, ::-1, :].copy(), size)
            prob_flip   = _forward(model, img_flipped, pts, labels_t, size)
            prob_unflip = prob_flip[:, ::-1].copy()
            all_probs[-1] = (prob + prob_unflip) / 2.0

    avg_prob = np.mean(all_probs, axis=0)
    if avg_prob.shape != (scales[0], scales[0]):
        avg_prob = cv2.resize(avg_prob, (scales[0], scales[0]))

    binary = to_binary_mask(avg_prob, threshold)
    if do_close:
        binary = _morph_close(binary, k_size)
    binary = resize_mask_to_original(binary, orig_h, orig_w)
    return binary, time.time() - t0


def run_inference(model, metadata_csv, output_dir, cfg, split="val",
                  compute_metrics=True):
    import pandas as pd
    df        = pd.read_csv(metadata_csv)
    df        = df[df["split"] == split].reset_index(drop=True)
    out_dir   = ensure_dir(output_dir)
    threshold = cfg["training"].get("mask_threshold", 0.5)

    accumulator = MetricAccumulator(["crack", "taping"]) if compute_metrics else None
    times = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inference [{split}]"):
        label    = row["type"]
        img_path = Path(row["image_path"])
        image_id = img_path.stem

        pred_mask, t = predict_single(model, img_path, label, cfg, threshold)
        times.append(t)

        Image.fromarray(pred_mask).save(out_dir / f"{image_id}__segment_{label}.png")

        if accumulator and pd.notna(row.get("mask_path")):
            gt = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt_t   = torch.from_numpy((gt > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                pred_t = torch.from_numpy((pred_mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                if pred_t.shape != gt_t.shape:
                    pred_t = F.interpolate(pred_t.float(), gt_t.shape[-2:])
                accumulator.update((pred_t * 2) - 1, gt_t, label, threshold)

    result = {}
    if accumulator:
        result = accumulator.compute()
        for k, v in result.items():
            logger.info(f"  {k:10s}  Dice={v['dice']:.4f}  mIoU={v['miou']:.4f}")

    avg_t = np.mean(times) if times else 0
    result["runtime"] = {"avg_inference_s": round(avg_t, 4), "total_images": len(df)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata",   default="dataset/metadata.csv")
    parser.add_argument("--split",      default="val")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    model = SAMDoRA.load(
        adapter_path=args.checkpoint,
        base_model=cfg["model"]["name"],
        precision=cfg["model"]["precision"],
        device=get_device(),
    )
    out     = args.output_dir or cfg["output"]["mask_dir"]
    metrics = run_inference(model, args.metadata, out, cfg, args.split)

    import json
    with open(Path(out) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)