"""preprocessing/coco_to_masks.py

Builds a unified train/val dataset from two Roboflow COCO exports.

Dataset specs:
  Drywall (taping): train=821, val=203  — Roboflow pre-split, use as-is
  Cracks          : train-only 5370     — 80/20 split ourselves (seed=42)

Final counts:
  train : 4296 cracks + 821  taping = ~5117
  val   : 1074 cracks + 203  taping = ~1277

Output layout:
  dataset/
    images/train/    images/val/
    masks/train/     masks/val/
    metadata.csv     (image_path, mask_path, type, split)

Mask filename: {original_stem}__segment_{label}.png
File naming:   original filenames kept as-is, never renamed.
"""

from __future__ import annotations
import json
import shutil
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.common import set_seed, get_logger, ensure_dir

logger = get_logger("preprocess")


# ─────────────────────────────────────────────────────────────
# COCO polygon → binary mask PNG
# ─────────────────────────────────────────────────────────────

def convert_coco_split(
    coco_json: Path,
    images_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path,
    label: str,
) -> list[dict]:
    """
    Convert one COCO annotation file to binary PNG masks.

    Rules:
      - Polygon segmentation → rasterised with pycocotools (no bounding boxes)
      - All annotations for one image merged into a single mask (max)
      - Mask values: {0, 255}, single-channel PNG
      - Mask filename: {stem}__segment_{label}.png
      - Images copied to out_images_dir with ORIGINAL filename (no rename)
      - Images with zero annotations are skipped

    Returns list of dicts: {image_path, mask_path, type}
    """
    ensure_dir(out_images_dir)
    ensure_dir(out_masks_dir)

    if not coco_json.exists():
        logger.warning(f"Annotation file not found: {coco_json}")
        return []

    coco = COCO(str(coco_json))
    img_ids = coco.getImgIds()
    records = []
    skipped_missing = 0
    skipped_no_ann  = 0

    for img_id in tqdm(img_ids, desc=f"  [{label}] {coco_json.parent.name}", leave=False):
        img_info  = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        h, w      = img_info["height"], img_info["width"]

        # Locate source image — Roboflow puts images directly in split dir
        src_img = images_dir / file_name
        if not src_img.exists():
            # Sometimes nested in images/
            src_img = images_dir / "images" / file_name
        if not src_img.exists():
            skipped_missing += 1
            continue

        # Collect annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns    = coco.loadAnns(ann_ids)

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        has_seg = False

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue
            has_seg = True

            if isinstance(seg, list):
                # Polygon list → RLE → decode
                rles = coco_mask_utils.frPyObjects(seg, h, w)
                m    = coco_mask_utils.decode(rles)            # H×W or H×W×N
                if m.ndim == 3:
                    m = m.max(axis=2)
                combined_mask = np.maximum(combined_mask, m.astype(np.uint8))

            elif isinstance(seg, dict):
                # Already RLE
                m = coco_mask_utils.decode(seg)
                combined_mask = np.maximum(combined_mask, m.astype(np.uint8))

        if not has_seg:
            skipped_no_ann += 1
            continue

        # Save mask: {0, 255}
        stem      = Path(file_name).stem
        mask_name = f"{stem}__segment_{label}.png"
        mask_path = out_masks_dir / mask_name
        Image.fromarray(combined_mask * 255, mode="L").save(mask_path)

        # Copy image keeping original filename
        dst_img = out_images_dir / file_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        records.append({
            "image_path": str(dst_img),
            "mask_path":  str(mask_path),
            "type":       label,
        })

    logger.info(
        f"  [{label}] converted={len(records)}  "
        f"skipped_missing={skipped_missing}  skipped_no_ann={skipped_no_ann}"
    )
    return records


def _images_dir(split_dir: Path) -> Path:
    """Roboflow puts images directly in the split folder."""
    return split_dir


# ─────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────

def build_dataset(
    crack_root:  Path,
    taping_root: Path,
    out_root:    Path,
    val_split:   float = 0.20,
    seed:        int   = 42,
) -> None:
    """
    Assembles the unified dataset according to spec.

    Taping  → use Roboflow train/valid split exactly
    Cracks  → convert train/ entirely, then 80/20 random split (seed=42)
    """
    set_seed(seed)

    # ── Intermediate staging dirs ─────────────────────────────
    staging = out_root / "_staging"

    # ═══════════════════════════════════════════════════════════
    # 1. TAPING — respect Roboflow split
    # ═══════════════════════════════════════════════════════════
    logger.info("\n── Taping dataset ──────────────────────────────────────")

    taping_train = convert_coco_split(
        coco_json      = taping_root / "train" / "_annotations.coco.json",
        images_dir     = _images_dir(taping_root / "train"),
        out_images_dir = staging / "taping" / "train" / "images",
        out_masks_dir  = staging / "taping" / "train" / "masks",
        label          = "taping",
    )
    taping_val = convert_coco_split(
        coco_json      = taping_root / "valid" / "_annotations.coco.json",
        images_dir     = _images_dir(taping_root / "valid"),
        out_images_dir = staging / "taping" / "val" / "images",
        out_masks_dir  = staging / "taping" / "val" / "masks",
        label          = "taping",
    )

    if not taping_train:
        raise ValueError(f"No taping train images — check {taping_root}/train/")
    if not taping_val:
        raise ValueError(f"No taping val images — check {taping_root}/valid/")

    logger.info(f"  Taping  train={len(taping_train)}  val={len(taping_val)}")

    # ═══════════════════════════════════════════════════════════
    # 2. CRACKS — convert all, then 80/20 random split
    # ═══════════════════════════════════════════════════════════
    logger.info("\n── Cracks dataset ──────────────────────────────────────")

    crack_all = convert_coco_split(
        coco_json      = crack_root / "train" / "_annotations.coco.json",
        images_dir     = _images_dir(crack_root / "train"),
        out_images_dir = staging / "crack" / "all" / "images",
        out_masks_dir  = staging / "crack" / "all" / "masks",
        label          = "crack",
    )

    if not crack_all:
        raise ValueError(f"No crack images — check {crack_root}/train/")

    # 80/20 random split, seed=42, no stratification
    random.seed(seed)
    indices   = list(range(len(crack_all)))
    random.shuffle(indices)
    n_val     = round(len(crack_all) * val_split)
    val_idx   = set(indices[:n_val])
    crack_train = [crack_all[i] for i in range(len(crack_all)) if i not in val_idx]
    crack_val   = [crack_all[i] for i in val_idx]

    logger.info(f"  Cracks  total={len(crack_all)}  train={len(crack_train)}  val={len(crack_val)}")

    # ═══════════════════════════════════════════════════════════
    # 3. MERGE + copy into final directory layout
    # ═══════════════════════════════════════════════════════════
    logger.info("\n── Building final dataset layout ───────────────────────")

    splits = {
        "train": taping_train + crack_train,
        "val":   taping_val   + crack_val,
    }

    all_rows: list[dict] = []

    for split_tag, records in splits.items():
        img_out  = ensure_dir(out_root / "images" / split_tag)
        mask_out = ensure_dir(out_root / "masks"  / split_tag)

        seen_images = set()
        seen_masks  = set()

        for row in tqdm(records, desc=f"  Copying {split_tag:5s}", leave=False):
            src_img  = Path(row["image_path"])
            src_mask = Path(row["mask_path"])

            dst_img  = img_out  / src_img.name
            dst_mask = mask_out / src_mask.name

            # Guard: no duplicates, no cross-split contamination
            if dst_img.name in seen_images:
                logger.warning(f"  Duplicate image skipped: {dst_img.name}")
                continue
            if dst_mask.name in seen_masks:
                logger.warning(f"  Duplicate mask skipped: {dst_mask.name}")
                continue

            seen_images.add(dst_img.name)
            seen_masks.add(dst_mask.name)

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if not dst_mask.exists():
                shutil.copy2(src_mask, dst_mask)

            # Sanity: mask file must exist after copy
            assert dst_mask.exists(), f"Mask copy failed: {dst_mask}"

            all_rows.append({
                "image_path": str(dst_img),
                "mask_path":  str(dst_mask),
                "type":       row["type"],
                "split":      split_tag,
            })

    # ═══════════════════════════════════════════════════════════
    # 4. Write metadata.csv
    # ═══════════════════════════════════════════════════════════
    meta = pd.DataFrame(all_rows, columns=["image_path", "mask_path", "type", "split"])
    meta.to_csv(out_root / "metadata.csv", index=False)

    # Also write per-split files for convenience
    for split_tag in ("train", "val"):
        meta[meta["split"] == split_tag].to_csv(
            out_root / f"metadata_{split_tag}.csv", index=False
        )

    # ═══════════════════════════════════════════════════════════
    # 5. Validation report
    # ═══════════════════════════════════════════════════════════
    logger.info("\n── Dataset summary ─────────────────────────────────────")
    for split_tag in ("train", "val"):
        sub    = meta[meta["split"] == split_tag]
        counts = sub["type"].value_counts().to_dict()
        logger.info(
            f"  [{split_tag:5s}]  total={len(sub):5d}  "
            f"crack={counts.get('crack', 0):5d}  taping={counts.get('taping', 0):4d}"
        )

    total_train = len(meta[meta["split"] == "train"])
    total_val   = len(meta[meta["split"] == "val"])

    # Alignment check: every image must have a corresponding mask file
    missing = 0
    for _, row in meta.iterrows():
        if not Path(row["mask_path"]).exists():
            logger.error(f"  MISSING mask: {row['mask_path']}")
            missing += 1
    if missing:
        raise RuntimeError(f"{missing} masks are missing — check conversion step")

    # Cross-split check: no filename appears in both train and val
    train_imgs = set(meta[meta["split"]=="train"]["image_path"].apply(lambda p: Path(p).name))
    val_imgs   = set(meta[meta["split"]=="val"  ]["image_path"].apply(lambda p: Path(p).name))
    overlap    = train_imgs & val_imgs
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {len(overlap)} files in both splits!")

    logger.info(f"\n  Expected  train=~5117  val=~1277")
    logger.info(f"  Got       train={total_train}    val={total_val}")
    logger.info(f"  Alignment check: {'✅ PASSED' if missing == 0 else '❌ FAILED'}")
    logger.info(f"  Cross-split check: {'✅ PASSED' if not overlap else '❌ FAILED'}")
    logger.info(f"\n✅ metadata.csv → {out_root / 'metadata.csv'}")

    # Optionally clean up staging
    shutil.rmtree(staging, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO → unified binary-mask dataset")
    parser.add_argument("--crack_root",  required=True, help="Path to cracks Roboflow export")
    parser.add_argument("--taping_root", required=True, help="Path to taping Roboflow export")
    parser.add_argument("--out_root",    default="dataset")
    parser.add_argument("--val_split",   type=float, default=0.20)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    build_dataset(
        crack_root  = Path(args.crack_root),
        taping_root = Path(args.taping_root),
        out_root    = Path(args.out_root),
        val_split   = args.val_split,
        seed        = args.seed,
    )