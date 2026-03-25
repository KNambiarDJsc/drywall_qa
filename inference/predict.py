"""inference/predict.py
Phase 11 — Production inference:
  • 5-prompt weighted ensemble
  • Multi-scale (768 + 1024) averaging
  • TTA (horizontal flip)
  • Morphological CLOSE on crack masks (connects broken lines)
"""

from __future__ import annotations
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[1]))
from models.sam3_dora import SAM3DoRA
from utils.common import (get_logger, ensure_dir, set_seed,
                           get_device, to_binary_mask, resize_mask_to_original)
from utils.metrics import MetricAccumulator, dice_score

logger = get_logger("inference")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────

def _preprocess(img_rgb: np.ndarray, size: int) -> torch.Tensor:
    resized = cv2.resize(img_rgb, (size, size))
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# Single forward pass (one scale, one prompt)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward(model, img_tensor: torch.Tensor, prompt: str, size: int) -> np.ndarray:
    """Returns H×W float32 probability map at `size` resolution."""
    inp = img_tensor.to(model.device, dtype=model.dtype)
    pred = model(inp, [prompt])
    if pred.shape[-2:] != (size, size):
        pred = F.interpolate(pred, (size, size), mode="bilinear", align_corners=False)
    return torch.sigmoid(pred).squeeze().cpu().float().numpy()


# ─────────────────────────────────────────────────────────────
# Morphological post-processing
# ─────────────────────────────────────────────────────────────

def _morph_close(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Connect broken crack segments via morphological closing."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)


# ─────────────────────────────────────────────────────────────
# Full inference for one image
# ─────────────────────────────────────────────────────────────

def predict_single(
    model: SAM3DoRA,
    image_path: str | Path,
    label: str,
    inference_cfg: dict,           # from config["inference_prompts"] + config["inference"]
    threshold: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Returns:
        pred_mask: H×W uint8 {0,255} at ORIGINAL image size
        inference_time: float seconds
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    prompt_cfg = inference_cfg.get("inference_prompts", {}).get(label, {})
    prompts    = prompt_cfg.get("prompts", [f"segment {label}"])
    weights    = np.array(prompt_cfg.get("weights", [1.0/len(prompts)]*len(prompts)), dtype=np.float32)
    weights   /= weights.sum()

    scales     = inference_cfg.get("inference", {}).get("multiscale", [768])
    tta        = inference_cfg.get("inference", {}).get("tta_enabled", True)
    do_close   = inference_cfg.get("inference", {}).get("morph_close_crack", True) and label == "crack"
    k_size     = inference_cfg.get("inference", {}).get("morph_kernel_size", 3)

    t0 = time.time()
    model.eval()

    all_probs = []
    for size in scales:
        img_tensor       = _preprocess(img_rgb, size)
        img_tensor_flip  = _preprocess(img_rgb[:, ::-1, :].copy(), size) if tta else None

        for prompt, w in zip(prompts, weights):
            prob = _forward(model, img_tensor, prompt, size)
            all_probs.append((prob, w))

            if tta:
                prob_flip = _forward(model, img_tensor_flip, prompt, size)
                prob_unflip = prob_flip[:, ::-1].copy()  # flip back
                # Average original + unflipped
                all_probs[-1] = ((prob + prob_unflip) / 2.0, w)

    # Weighted average across prompts × scales
    total_w = sum(w for _, w in all_probs)
    avg_prob = sum(p * w for p, w in all_probs) / total_w

    # Resize to output size (all scales contributed at their own res — unify at 768)
    if avg_prob.shape != (scales[0], scales[0]):
        avg_prob = cv2.resize(avg_prob, (scales[0], scales[0]))

    binary = to_binary_mask(avg_prob, threshold)

    if do_close:
        binary = _morph_close(binary, k_size)

    binary = resize_mask_to_original(binary, orig_h, orig_w)
    return binary, time.time() - t0


# ─────────────────────────────────────────────────────────────
# Batch inference over metadata split
# ─────────────────────────────────────────────────────────────

def run_inference(
    model: SAM3DoRA,
    metadata_csv: str | Path,
    output_dir: str | Path,
    cfg: dict,
    split: str = "val",
    compute_metrics: bool = True,
) -> dict:
    import pandas as pd
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == split].reset_index(drop=True)
    out_dir = ensure_dir(output_dir)
    threshold = cfg["training"].get("mask_threshold", 0.5)

    accumulator = MetricAccumulator(["crack", "taping"]) if compute_metrics else None
    times = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inference [{split}]"):
        label    = row["type"]
        img_path = Path(row["image_path"])
        image_id = img_path.stem

        pred_mask, t = predict_single(model, img_path, label, cfg, threshold)
        times.append(t)

        out_name = f"{image_id}__segment_{label}.png"
        Image.fromarray(pred_mask).save(out_dir / out_name)

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
        logger.info(f"\n── Inference Metrics [{split}] ──")
        for k, v in result.items():
            logger.info(f"  {k:10s}  Dice={v['dice']:.4f}  mIoU={v['miou']:.4f}")

    avg_t = np.mean(times) if times else 0
    result["runtime"] = {"avg_inference_s": round(avg_t, 4), "total_images": len(df)}
    logger.info(f"  avg {avg_t*1000:.1f} ms/image")
    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

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

    model = SAM3DoRA.load(
        adapter_path=args.checkpoint,
        base_model=cfg["model"]["name"],
        precision=cfg["model"]["precision"],
        device=get_device(),
    )
    out = args.output_dir or cfg["output"]["mask_dir"]
    metrics = run_inference(model, args.metadata, out, cfg, args.split)

    import json
    with open(Path(out) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved → {Path(out) / 'metrics.json'}")