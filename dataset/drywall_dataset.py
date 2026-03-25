"""dataset/drywall_dataset.py
Grounded SAM 2 dataset — text prompts + PIL images for DINO.

Returns per sample:
  image         [3, H, W]  float32 normalised tensor
  image_pil     PIL.Image  for Grounding DINO
  mask          [1, H, W]  float32 {0, 1}
  prompt        str        e.g. "crack" or "drywall seam"
  label         str        "crack" | "taping"
  image_id      str
  orig_h/orig_w int
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.append(str(Path(__file__).parents[1]))
from dataset.synthetic_cracks import SyntheticCrackAugmentation
from utils.common import get_logger

logger = get_logger("dataset")


# ─────────────────────────────────────────────────────────────
# Prompt maps
# ─────────────────────────────────────────────────────────────

PROMPT_MAP: Dict[str, List[str]] = {
    "crack":  ["crack", "wall crack", "surface crack", "hairline crack"],
    "taping": ["taping area", "drywall seam", "joint tape", "wall joint"],
}


# ─────────────────────────────────────────────────────────────
# Patch crop
# ─────────────────────────────────────────────────────────────

def patch_crop_around_mask(image, mask, target_size, scale_range=(0.15, 0.40)):
    h, w = image.shape[:2]
    fg   = np.argwhere(mask > 0)
    if len(fg) == 0:
        cy, cx = h//2, w//2
        half   = min(h, w)//2
        y1, y2, x1, x2 = cy-half, cy+half, cx-half, cx+half
    else:
        y_min, x_min = fg.min(axis=0)
        y_max, x_max = fg.max(axis=0)
        scale = random.uniform(*scale_range)
        pad_y = max(int((y_max - y_min) * scale), 10)
        pad_x = max(int((x_max - x_min) * scale), 10)
        y1 = max(0, y_min - pad_y)
        y2 = min(h, y_max + pad_y)
        x1 = max(0, x_min - pad_x)
        x2 = min(w, x_max + pad_x)
    crop_img = image[y1:y2, x1:x2]
    crop_msk = mask[y1:y2, x1:x2]
    if crop_img.size == 0:
        return image, mask
    crop_img = cv2.resize(crop_img, (target_size, target_size))
    crop_msk = cv2.resize(crop_msk, (target_size, target_size),
                          interpolation=cv2.INTER_NEAREST)
    return crop_img, crop_msk


# ─────────────────────────────────────────────────────────────
# Augmentation — class-aware (pixel ops only, no resize in PIL)
# ─────────────────────────────────────────────────────────────

def build_crack_transforms(image_size=768):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.30, p=0.7),
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.4),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
        A.Sharpen(alpha=(0.4, 0.9), lightness=(0.5, 1.2), p=0.80),
        A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.3, 0.7), p=0.50),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def build_taping_transforms(image_size=768):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomResizedCrop(height=image_size, width=image_size,
                            scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.4),
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=(2, 4), tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.6, p=0.8),
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.50),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def build_val_transforms(image_size=768):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class DrywallDataset(Dataset):

    def __init__(
        self,
        metadata_csv,
        split="train",
        image_size=768,
        augment=True,
        prompt_map=None,
        patch_prob_crack=0.25,
        patch_prob_taping=0.70,
        patch_scale_crack=(0.15, 0.40),
        patch_scale_taping=(0.30, 0.60),
        synthetic_crack_prob=0.20,
        active_classes=None,
    ):
        self._src_csv = str(metadata_csv)
        df = pd.read_csv(metadata_csv)
        df = df[df["split"] == split]
        if active_classes:
            df = df[df["type"].isin(active_classes)]
        self.df = df.reset_index(drop=True)

        self.split                = split
        self.image_size           = image_size
        self.augment              = augment
        self.prompt_map           = prompt_map or PROMPT_MAP
        self.patch_prob           = {"crack": patch_prob_crack, "taping": patch_prob_taping}
        self.patch_scale          = {"crack": patch_scale_crack, "taping": patch_scale_taping}
        self.synthetic_crack_prob = synthetic_crack_prob
        self._syn_aug             = SyntheticCrackAugmentation()

        self._transforms = {
            "crack":  build_crack_transforms(image_size) if augment else build_val_transforms(image_size),
            "taping": build_taping_transforms(image_size) if augment else build_val_transforms(image_size),
        }

        logger.info(
            f"[DrywallDataset:{split}] n={len(self.df)}  "
            f"crack={len(self.df[self.df.type=='crack'])}  "
            f"taping={len(self.df[self.df.type=='taping'])}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = row["type"]

        img_bgr = cv2.imread(str(row["image_path"]))
        img     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        mask = (cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        if self.augment:
            if label == "crack" and random.random() < self.synthetic_crack_prob:
                img, mask = self._syn_aug(img, mask)
            if random.random() < self.patch_prob[label]:
                img, mask = patch_crop_around_mask(
                    img, mask, self.image_size, self.patch_scale[label]
                )

        # PIL image BEFORE normalisation (for Grounding DINO)
        img_pil = PILImage.fromarray(img if img.shape[:2] == (self.image_size, self.image_size)
                                     else cv2.resize(img, (self.image_size, self.image_size)))

        # Albumentations (normalise + tensor)
        t            = self._transforms[label](image=img, mask=mask)
        image_tensor = t["image"]
        mask_tensor  = t["mask"].unsqueeze(0).float()

        # Text prompt — random during training, fixed during val
        prompt = (random.choice(self.prompt_map[label]) if self.augment
                  else self.prompt_map[label][0])

        return {
            "image":     image_tensor,
            "image_pil": img_pil,
            "mask":      mask_tensor,
            "prompt":    prompt,
            "label":     label,
            "image_id":  Path(row["image_path"]).stem,
            "orig_h":    orig_h,
            "orig_w":    orig_w,
        }

    def reload_with_classes(self, classes):
        df = pd.read_csv(self._src_csv)
        self.df = df[
            (df["split"] == self.split) & (df["type"].isin(classes))
        ].reset_index(drop=True)
        logger.info(f"[Curriculum] classes={classes}  n={len(self.df)}")

    def set_image_size(self, size):
        self.image_size = size
        self._transforms = {
            "crack":  build_crack_transforms(size) if self.augment else build_val_transforms(size),
            "taping": build_taping_transforms(size) if self.augment else build_val_transforms(size),
        }
        logger.info(f"[Curriculum] image_size → {size}")


# ─────────────────────────────────────────────────────────────
# Guaranteed-minority BatchSampler
# ─────────────────────────────────────────────────────────────

class MinClassBatchSampler(torch.utils.data.Sampler):
    """Guarantees ≥1 taping per batch. Handles single-class phases."""

    def __init__(self, dataset, batch_size, class_weights,
                 minority_class="taping", min_per_batch=1):
        self.batch_size    = batch_size
        self.min_per_batch = min_per_batch

        labels = [dataset.df.iloc[i]["type"] for i in range(len(dataset))]
        self.minority_idx = [i for i, l in enumerate(labels) if l == minority_class]
        self.majority_idx = [i for i, l in enumerate(labels) if l != minority_class]

        if self.majority_idx:
            maj_w = torch.tensor(
                [class_weights.get(labels[i], 1.0) for i in self.majority_idx],
                dtype=torch.float,
            )
            self.maj_probs = (maj_w / maj_w.sum()).numpy()
        else:
            self.maj_probs = None

        self.n_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            if not self.majority_idx:
                # Single-class phase
                batch = np.random.choice(
                    self.minority_idx, size=self.batch_size, replace=True
                ).tolist()
            else:
                minority = np.random.choice(
                    self.minority_idx, size=self.min_per_batch, replace=True
                ).tolist()
                majority = np.random.choice(
                    self.majority_idx,
                    size=self.batch_size - self.min_per_batch,
                    replace=True,
                    p=self.maj_probs,
                ).tolist()
                batch = minority + majority
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ─────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────

def pil_collate_fn(batch):
    """Custom collate that keeps image_pil as a list (not stacked tensor)."""
    images     = torch.stack([b["image"] for b in batch])
    masks      = torch.stack([b["mask"]  for b in batch])
    images_pil = [b["image_pil"] for b in batch]
    prompts    = [b["prompt"]    for b in batch]
    labels     = [b["label"]     for b in batch]
    image_ids  = [b["image_id"]  for b in batch]
    orig_hs    = [b["orig_h"]    for b in batch]
    orig_ws    = [b["orig_w"]    for b in batch]
    return {
        "image":     images,
        "image_pil": images_pil,
        "mask":      masks,
        "prompt":    prompts,
        "label":     labels,
        "image_id":  image_ids,
        "orig_h":    orig_hs,
        "orig_w":    orig_ws,
    }


def build_dataloaders(
    metadata_csv,
    batch_size=4,
    image_size=768,
    num_workers=8,
    class_weights=None,
    synthetic_crack_prob=0.20,
    active_classes=None,
    guarantee_taping=True,
):
    if class_weights is None:
        class_weights = {"crack": 1.0, "taping": 5.4}

    train_ds = DrywallDataset(
        metadata_csv, split="train", image_size=image_size, augment=True,
        synthetic_crack_prob=synthetic_crack_prob,
        active_classes=active_classes,
    )
    val_ds = DrywallDataset(
        metadata_csv, split="val", image_size=image_size, augment=False,
    )

    if guarantee_taping and (active_classes is None or "taping" in (active_classes or [])):
        sampler = MinClassBatchSampler(
            train_ds, batch_size, class_weights,
            minority_class="taping", min_per_batch=1,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            collate_fn=pil_collate_fn,
        )
    else:
        weights = torch.tensor(
            [class_weights.get(train_ds.df.iloc[i]["type"], 1.0)
             for i in range(len(train_ds))], dtype=torch.float,
        )
        sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            drop_last=True, collate_fn=pil_collate_fn,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=pil_collate_fn,
    )
    return train_loader, val_loader, train_ds