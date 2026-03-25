"""preprocessing/coco_to_masks.py

Annotation-agnostic COCO → binary mask converter.

Priority order per annotation:
  1. Polygon segmentation  → rasterise with pycocotools
  2. RLE segmentation      → decode with pycocotools
  3. Bounding box fallback → filled rectangle (weak supervision)
  4. Nothing               → skip image

No category filtering. All annotations for an image are merged.

Dataset layout handled:
  Taping : train/_annotations.coco.json + valid/_annotations.coco.json
  Cracks : train/_annotations.coco.json only → 80/20 random split (seed=42)

Output:
  dataset/
    images/train/   images/val/
    masks/train/    masks/val/
    metadata.csv    (image_path, mask_path, type, split)

Mask filename : {original_stem}__segment_{label}.png
Mask values   : {0, 255}, single-channel PNG
Image names   : never renamed
"""

from __future__ import annotations
import os
import json
import random
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save {0,1} mask as {0,255} single-channel PNG."""
    out = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), out)


# ─────────────────────────────────────────────────────────────
# Core converter — one COCO split
# ─────────────────────────────────────────────────────────────

def convert_split(
    ann_file: Path,
    image_dir: Path,
    out_image_dir: Path,
    out_mask_dir: Path,
    label: str,
) -> list[dict]:
    """
    Convert one COCO annotation file → binary masks.
    Returns list of raw records {image_path, mask_path, type}.
    """
    if not ann_file.exists():
        print(f"  [WARN] Missing annotation file: {ann_file}")
        return []

    ensure_dir(out_image_dir)
    ensure_dir(out_mask_dir)

    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()

    records = []
    n_seg = 0          # used polygon/RLE
    n_bbox = 0         # fell back to bbox
    n_missing = 0      # image file not found on disk
    n_no_ann = 0       # image had annotations but none usable

    for img_id in tqdm(img_ids, desc=f"  [{label}] {ann_file.parent.name}", leave=False):
        img_info  = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        h, w      = img_info["height"], img_info["width"]

        # ── Locate source image ────────────────────────────────
        src = image_dir / file_name
        if not src.exists():
            src = image_dir / "images" / file_name   # nested fallback
        if not src.exists():
            n_missing += 1
            continue

        # ── Build combined mask from all annotations ───────────
        combined = np.zeros((h, w), dtype=np.uint8)
        has_ann  = False

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns    = coco.loadAnns(ann_ids)

        for ann in anns:
            seg  = ann.get("segmentation")
            bbox = ann.get("bbox")

            # CASE 1: polygon segmentation
            if seg and len(seg) > 0:
                try:
                    if isinstance(seg, list):
                        rles = coco_mask_utils.frPyObjects(seg, h, w)
                        m    = coco_mask_utils.decode(rles)
                        if m.ndim == 3:
                            m = m.max(axis=2)
                        combined = np.maximum(combined, m.astype(np.uint8))
                        has_ann  = True
                        n_seg   += 1

                    elif isinstance(seg, dict):
                        m       = coco_mask_utils.decode(seg)
                        combined = np.maximum(combined, m.astype(np.uint8))
                        has_ann  = True
                        n_seg   += 1

                except Exception:
                    # malformed segmentation → fall through to bbox
                    pass

            # CASE 2: bbox fallback (cracks dataset)
            if not has_ann and bbox:
                x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                x2 = min(x + bw, w)
                y2 = min(y + bh, h)
                if x2 > x and y2 > y:
                    combined[y:y2, x:x2] = 1
                    has_ann  = True
                    n_bbox  += 1

        if not has_ann:
            n_no_ann += 1
            continue

        # ── Save mask ──────────────────────────────────────────
        stem      = Path(file_name).stem
        mask_name = f"{stem}__segment_{label}.png"
        mask_path = out_mask_dir / mask_name
        save_mask(combined, mask_path)

        # ── Copy image (original filename, never renamed) ──────
        dst_img = out_image_dir / file_name
        if not dst_img.exists():
            shutil.copy2(src, dst_img)

        records.append({
            "image_path": str(dst_img),
            "mask_path":  str(mask_path),
            "type":       label,
        })

    print(f"  [{label}] {ann_file.parent.name}: "
          f"converted={len(records)}  "
          f"seg={n_seg}  bbox_fallback={n_bbox}  "
          f"missing={n_missing}  no_ann={n_no_ann}")

    return records


# ─────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────

def build_dataset(
    crack_root:  Path,
    taping_root: Path,
    out_root:    Path,
    val_split:   float = 0.20,
    seed:        int   = 42,
) -> None:
    set_seed(seed)
    staging = out_root / "_staging"

    # ── 1. Taping — Roboflow pre-split ────────────────────────
    print("\n── Taping dataset ──────────────────────────────────────")
    taping_train = convert_split(
        ann_file      = taping_root / "train" / "_annotations.coco.json",
        image_dir     = taping_root / "train",
        out_image_dir = staging / "taping" / "train" / "images",
        out_mask_dir  = staging / "taping" / "train" / "masks",
        label         = "taping",
    )
    taping_val = convert_split(
        ann_file      = taping_root / "valid" / "_annotations.coco.json",
        image_dir     = taping_root / "valid",
        out_image_dir = staging / "taping" / "val" / "images",
        out_mask_dir  = staging / "taping" / "val" / "masks",
        label         = "taping",
    )

    if not taping_train:
        raise ValueError(f"Zero taping train images. Check {taping_root}/train/")
    if not taping_val:
        raise ValueError(f"Zero taping val images. Check {taping_root}/valid/")

    # ── 2. Cracks — train only, split ourselves ───────────────
    print("\n── Cracks dataset ──────────────────────────────────────")
    crack_all = convert_split(
        ann_file      = crack_root / "train" / "_annotations.coco.json",
        image_dir     = crack_root / "train",
        out_image_dir = staging / "crack" / "all" / "images",
        out_mask_dir  = staging / "crack" / "all" / "masks",
        label         = "crack",
    )

    if not crack_all:
        raise ValueError(f"Zero crack images. Check {crack_root}/train/")

    # 80/20 random split, seed=42, no stratification
    indices = list(range(len(crack_all)))
    random.shuffle(indices)
    n_val       = round(len(crack_all) * val_split)
    val_set     = set(indices[:n_val])
    crack_train = [crack_all[i] for i in range(len(crack_all)) if i not in val_set]
    crack_val   = [crack_all[i] for i in val_set]

    print(f"  [crack] total={len(crack_all)}  "
          f"train={len(crack_train)}  val={len(crack_val)}")

    # ── 3. Merge + copy into final layout ─────────────────────
    print("\n── Building final layout ───────────────────────────────")

    splits = {
        "train": taping_train + crack_train,
        "val":   taping_val   + crack_val,
    }

    all_rows: list[dict] = []

    for split_tag, records in splits.items():
        img_out  = ensure_dir(out_root / "images" / split_tag)
        mask_out = ensure_dir(out_root / "masks"  / split_tag)
        seen     = set()

        for row in tqdm(records, desc=f"  Copying {split_tag:5s}", leave=False):
            src_img  = Path(row["image_path"])
            src_mask = Path(row["mask_path"])
            key      = src_img.name

            if key in seen:
                print(f"  [WARN] Duplicate skipped: {key}")
                continue
            seen.add(key)

            dst_img  = img_out  / src_img.name
            dst_mask = mask_out / src_mask.name

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if not dst_mask.exists():
                shutil.copy2(src_mask, dst_mask)

            all_rows.append({
                "image_path": str(dst_img),
                "mask_path":  str(dst_mask),
                "type":       row["type"],
                "split":      split_tag,
            })

    # ── 4. Write metadata.csv ─────────────────────────────────
    meta = pd.DataFrame(all_rows, columns=["image_path", "mask_path", "type", "split"])
    meta.to_csv(out_root / "metadata.csv", index=False)
    for tag in ("train", "val"):
        meta[meta["split"] == tag].to_csv(out_root / f"metadata_{tag}.csv", index=False)

    # ── 5. Validation report ──────────────────────────────────
    print("\n── Final dataset summary ───────────────────────────────")
    for tag in ("train", "val"):
        sub = meta[meta["split"] == tag]
        c   = sub["type"].value_counts().to_dict()
        print(f"  [{tag:5s}]  total={len(sub):5d}  "
              f"crack={c.get('crack',0):5d}  taping={c.get('taping',0):4d}")

    # Alignment: every mask file must exist
    missing = [r for _, r in meta.iterrows() if not Path(r["mask_path"]).exists()]
    # Cross-split: no filename in both splits
    train_names = set(meta[meta["split"]=="train"]["image_path"].apply(lambda p: Path(p).name))
    val_names   = set(meta[meta["split"]=="val"  ]["image_path"].apply(lambda p: Path(p).name))
    overlap     = train_names & val_names

    print(f"\n  Expected  train=~5117  val=~1277")
    print(f"  Alignment:   {'✅ PASSED' if not missing  else f'❌ {len(missing)} missing masks'}")
    print(f"  No leakage:  {'✅ PASSED' if not overlap  else f'❌ {len(overlap)} files in both splits'}")
    print(f"\n✅ metadata.csv → {out_root / 'metadata.csv'}")

    if missing or overlap:
        raise RuntimeError("Dataset validation failed — see above.")

    # Clean staging
    shutil.rmtree(staging, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO → unified binary-mask dataset")
    parser.add_argument("--crack_root",  required=True)
    parser.add_argument("--taping_root", required=True)
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