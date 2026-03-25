"""dataset/drywall_dataset.py
Phase 3 – PyTorch Dataset
Phase 4 – Class-aware augmentation pipeline (CLAHE, crack-specific sharpen,
           patch-based crop, synthetic cracks)
Phase 5 – WeightedRandomSampler (base) + Hard Example Mining hook

Prompt engine:
  • Dropout  — blank prompt with prob p (forces visual grounding)
  • Noise    — random char-swap typos (robustness)
  • Weighted ensemble available at inference (see predict.py)
"""

from __future__ import annotations
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.append(str(Path(__file__).parents[1]))
from dataset.synthetic_cracks import SyntheticCrackAugmentation
from utils.common import get_logger

logger = get_logger("dataset")

PROMPT_MAP: Dict[str, List[str]] = {
    "crack":  ["segment crack","wall fracture","hairline crack","surface crack","drywall defect"],
    "taping": [
        "segment taping area",
        "drywall seam",
        "joint line",
        "wall joint tape",
        "connection seam",
        "thin seam",               # new — targets faint seams
        "faint wall joint",        # new — low-contrast cases
        "vertical seam line",      # new — signals orientation to SAM3
    ],
}


# ─────────────────────────────────────────────────────────────
# Prompt engine
# ─────────────────────────────────────────────────────────────

def _inject_typo(text: str, n_swaps: int = 1) -> str:
    chars = list(text)
    for _ in range(n_swaps):
        if len(chars) < 4:
            break
        idx = random.randint(1, len(chars) - 2)
        mode = random.choice(["swap", "drop", "replace"])
        if mode == "swap" and idx + 1 < len(chars):
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif mode == "drop":
            chars.pop(idx)
        else:
            chars[idx] = random.choice(string.ascii_lowercase)
    return "".join(chars)


def sample_prompt(label, prompt_map, dropout_prob=0.10, noise_prob=0.15):
    if random.random() < dropout_prob:
        return ""
    prompt = random.choice(prompt_map[label])
    if random.random() < noise_prob:
        prompt = _inject_typo(prompt)
    return prompt


# ─────────────────────────────────────────────────────────────
# Patch crop
# ─────────────────────────────────────────────────────────────

def patch_crop_around_mask(image, mask, target_size, scale_range=(0.15, 0.40)):
    h, w = image.shape[:2]
    fg = np.argwhere(mask > 0)
    if len(fg) == 0:
        cy, cx = h // 2, w // 2
        half = min(h, w) // 2
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
    crop_msk = cv2.resize(crop_msk, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return crop_img, crop_msk


# ─────────────────────────────────────────────────────────────
# Class-aware augmentation pipelines
# ─────────────────────────────────────────────────────────────

def _common_pixel():
    return [
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.30, p=0.7),
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.4),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
    ]


def build_crack_transforms(image_size=768):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.Resize(image_size, image_size),
        *_common_pixel(),
        A.Sharpen(alpha=(0.4, 0.9), lightness=(0.5, 1.2), p=0.80),
        A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.3, 0.7), p=0.50),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def build_taping_transforms(image_size=768):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7,1.0), ratio=(0.9,1.1), p=0.4),
        A.Resize(image_size, image_size),
        # ── Taping-specific: lower clip_limit to preserve seam texture ──
        # clip_limit=(2,4) instead of 4.0 fixed — high clip crushes the
        # faint contrast gradient that distinguishes seam from wall
        A.CLAHE(clip_limit=(2, 4), tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.6, p=0.8),
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.50),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def build_val_transforms(image_size=768):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
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
        prompt_dropout_prob=0.10,
        prompt_noise_prob=0.15,
        patch_prob_crack=0.50,
        patch_prob_taping=0.50,   # was 0.20 — seams are tiny, patch is mandatory
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

        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.prompt_map = prompt_map or PROMPT_MAP
        self.prompt_dropout_prob = prompt_dropout_prob
        self.prompt_noise_prob = prompt_noise_prob
        self.patch_prob = {"crack": patch_prob_crack, "taping": patch_prob_taping}
        self.patch_scale = {"crack": patch_scale_crack, "taping": patch_scale_taping}
        self.synthetic_crack_prob = synthetic_crack_prob
        self._syn_aug = SyntheticCrackAugmentation()

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
        row = self.df.iloc[idx]
        label = row["type"]

        img = cv2.cvtColor(cv2.imread(str(row["image_path"])), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        mask = (cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

        if self.augment:
            if label == "crack" and random.random() < self.synthetic_crack_prob:
                img, mask = self._syn_aug(img, mask)
            if random.random() < self.patch_prob[label]:
                img, mask = patch_crop_around_mask(
                    img, mask, self.image_size, self.patch_scale[label]
                )

        t = self._transforms[label](image=img, mask=mask)
        image_tensor = t["image"]
        mask_tensor  = t["mask"].unsqueeze(0).float()

        prompt = sample_prompt(
            label, self.prompt_map,
            self.prompt_dropout_prob if self.augment else 0.0,
            self.prompt_noise_prob   if self.augment else 0.0,
        )

        return {
            "image":    image_tensor,
            "mask":     mask_tensor,
            "prompt":   prompt,
            "label":    label,
            "image_id": Path(row["image_path"]).stem,
            "orig_h":   orig_h,
            "orig_w":   orig_w,
        }

    def reload_with_classes(self, classes: List[str]) -> None:
        """Curriculum hot-swap — no DataLoader rebuild needed."""
        df = pd.read_csv(self._src_csv)
        self.df = df[
            (df["split"] == self.split) & (df["type"].isin(classes))
        ].reset_index(drop=True)
        logger.info(f"[Curriculum] classes={classes}  n={len(self.df)}")

    def set_image_size(self, size: int) -> None:
        """Resolution curriculum — rebuild transforms in place."""
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
    """
    Guarantees at least `min_per_batch` samples of `minority_class`
    in every batch. The rest are filled with WeightedRandomSampler draws.

    Prevents the edge case where a batch has zero taping samples and the
    mask decoder sees only cracks for an entire update step — which causes
    taping Dice to silently degrade mid-training.
    """

    def __init__(
        self,
        dataset: DrywallDataset,
        batch_size: int,
        class_weights: Dict[str, float],
        minority_class: str = "taping",
        min_per_batch: int = 1,
    ):
        self.batch_size     = batch_size
        self.min_per_batch  = min_per_batch
        self.minority_class = minority_class

        labels = [dataset.df.iloc[i]["type"] for i in range(len(dataset))]
        self.minority_idx = [i for i, l in enumerate(labels) if l == minority_class]
        self.majority_idx = [i for i, l in enumerate(labels) if l != minority_class]

        # Weighted probabilities for majority draws
        maj_weights = torch.tensor(
            [class_weights.get(labels[i], 1.0) for i in self.majority_idx],
            dtype=torch.float,
        )
        self.maj_probs = (maj_weights / maj_weights.sum()).numpy()
        self.n_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            # Guaranteed minority slots
            minority_sample = np.random.choice(
                self.minority_idx,
                size=self.min_per_batch,
                replace=True,
            ).tolist()
            # Fill remainder from majority (weighted)
            n_maj = self.batch_size - self.min_per_batch
            majority_sample = np.random.choice(
                self.majority_idx,
                size=n_maj,
                replace=True,
                p=self.maj_probs,
            ).tolist()
            batch = minority_sample + majority_sample
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ─────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    metadata_csv,
    batch_size=4,
    image_size=768,
    num_workers=4,
    class_weights=None,
    prompt_dropout_prob=0.10,
    prompt_noise_prob=0.15,
    synthetic_crack_prob=0.20,
    active_classes=None,
    guarantee_taping: bool = True,
):
    """
    Returns (train_loader, val_loader, train_dataset).

    guarantee_taping=True uses MinClassBatchSampler so every batch
    contains at least 1 taping sample — prevents silent taping regression.
    """
    if class_weights is None:
        class_weights = {"crack": 1.0, "taping": 5.4}

    train_ds = DrywallDataset(
        metadata_csv, split="train", image_size=image_size, augment=True,
        prompt_dropout_prob=prompt_dropout_prob,
        prompt_noise_prob=prompt_noise_prob,
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
        )
    else:
        weights = torch.tensor(
            [class_weights.get(train_ds.df.iloc[i]["type"], 1.0) for i in range(len(train_ds))],
            dtype=torch.float,
        )
        sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds