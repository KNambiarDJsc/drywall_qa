"""training/trainer.py — Curriculum trainer for SAM ViT-H + DoRA.
Uses point prompts (not text). All other logic unchanged.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[1]))
from training.losses import HybridLoss
from dataset.hard_example_mining import HardExampleMiner
from utils.metrics import MetricAccumulator, dice_score
from utils.common import get_logger, ensure_dir

logger = get_logger("trainer")


# ─────────────────────────────────────────────────────────────
# Multi-mask supervision
# ─────────────────────────────────────────────────────────────

def topk_mask_loss(pred_all, target, criterion, k=3):
    """Min loss across K mask proposals."""
    B, K_actual, H, W = pred_all.shape
    k = min(k, K_actual)
    losses = []
    for i in range(k):
        l, _ = criterion(pred_all[:, i:i+1], target)
        losses.append(l.unsqueeze(0))
    stacked  = torch.cat(losses)
    loss     = stacked.min()
    best_idx = stacked.argmin().item()
    best_pred = pred_all[:, best_idx:best_idx+1]
    _, breakdown = criterion(best_pred, target)
    return loss, breakdown, best_pred


# ─────────────────────────────────────────────────────────────
# Encoder freeze helpers
# ─────────────────────────────────────────────────────────────

def _set_encoder_frozen(model, frozen):
    base = model.sam.base_model if hasattr(model.sam, "base_model") else model.sam
    encoder = getattr(base, "vision_encoder", None) or getattr(base, "image_encoder", None)
    if encoder:
        for p in encoder.parameters():
            p.requires_grad_(not frozen)
    for name, p in model.sam.named_parameters():
        if "lora_" in name or "dora_" in name:
            p.requires_grad_(True)
    logger.info(f"  Encoder: {'FROZEN' if frozen else 'UNFROZEN'}")


def _unfreeze_last_encoder_layers(model, n_layers=4):
    base = model.sam.base_model if hasattr(model.sam, "base_model") else model.sam
    encoder = getattr(base, "vision_encoder", None) or getattr(base, "image_encoder", None)
    if not encoder:
        return
    blocks = getattr(encoder, "layers", None) or getattr(encoder, "blocks", [])
    for block in list(blocks)[-n_layers:]:
        for p in block.parameters():
            p.requires_grad_(True)
    logger.info(f"  Unfroze last {n_layers} encoder blocks")


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class DrywallTrainer:
    def __init__(self, model, train_loader, val_loader, train_dataset, cfg,
                 checkpoint_dir="checkpoints"):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.train_ds     = train_dataset
        self.cfg          = cfg
        self.ckpt_dir     = ensure_dir(checkpoint_dir)

        loss_cfg = cfg.get("loss", {})
        self.criterion = HybridLoss(
            focal_weight=loss_cfg.get("focal_weight", 0.40),
            tversky_weight=loss_cfg.get("tversky_weight", 0.50),
            boundary_weight=loss_cfg.get("boundary_weight", 0.10),
            focal_alpha=loss_cfg.get("focal_alpha", 0.50),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.70),
            tversky_beta=loss_cfg.get("tversky_beta", 0.30),
        )

        self.top_k  = cfg["training"].get("top_k_masks", 3)
        self.scaler = GradScaler(enabled=(model.dtype == torch.float16))

        hem_cfg = cfg.get("hem", {})
        self.hem_enabled = hem_cfg.get("enabled", True)
        self.hem = HardExampleMiner(
            train_dataset,
            class_weights=cfg.get("class_weights", {"crack": 1.0, "taping": 5.4}),
            dice_threshold=hem_cfg.get("dice_threshold", 0.45),
            boost_factor=hem_cfg.get("boost_factor", 3.0),
        ) if self.hem_enabled else None

        if cfg["model"].get("torch_compile", False):
            self.model.sam = torch.compile(self.model.sam)
            logger.info("torch.compile() enabled")

        self.history = {
            "train_loss": [],
            "val_dice_crack": [], "val_dice_taping": [], "val_dice_macro": [],
            "val_miou_crack": [], "val_miou_taping": [], "val_miou_macro": [],
            "lr": [],
        }
        self.best_score = 0.0

    # ── Phase 0 ───────────────────────────────────────────────

    def run_baseline(self):
        logger.info("─" * 55)
        logger.info("Phase 0 — Zero-shot baseline")
        metrics = self._eval_epoch()
        logger.info(
            f"Baseline  Dice crack={metrics['crack']['dice']:.4f}  "
            f"taping={metrics['taping']['dice']:.4f}"
        )
        return metrics

    # ── Main training loop ────────────────────────────────────

    def train(self):
        for phase_cfg in self.cfg["training"]["curriculum"]:
            name       = phase_cfg["name"]
            epochs     = phase_cfg["epochs"]
            lr         = phase_cfg["lr"]
            classes    = phase_cfg.get("classes", ["crack", "taping"])
            img_size   = phase_cfg.get("image_size", self.cfg["data"]["image_size"])
            freeze_enc = phase_cfg.get("freeze_encoder", True)
            hem_start  = phase_cfg.get("hem_start_epoch", 999)

            logger.info(f"\n{'═'*55}\n{name}")
            self.train_ds.reload_with_classes(classes)
            self.train_ds.set_image_size(img_size)
            _set_encoder_frozen(self.model, freeze_enc)
            if not freeze_enc:
                _unfreeze_last_encoder_layers(self.model, n_layers=4)

            cw = self.cfg.get("class_weights", {"crack": 1.0, "taping": 5.4})
            self._rebuild_loader(classes, cw)

            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr, weight_decay=self.cfg["training"].get("weight_decay", 1e-4),
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

            for epoch in range(1, epochs + 1):
                t0 = time.time()

                if self.hem_enabled and epoch > hem_start:
                    self._rebuild_loader_hem(cw)

                hem_ids, hem_scores = [], []
                train_loss  = self._train_epoch(optimizer, hem_ids, hem_scores)
                val_metrics = self._eval_epoch()
                scheduler.step()

                if self.hem_enabled and hem_ids:
                    self.hem.update(hem_ids, hem_scores)

                macro_dice = val_metrics["macro"]["dice"]
                self._log(epoch, train_loss, val_metrics,
                          scheduler.get_last_lr()[0], time.time()-t0)

                if macro_dice > self.best_score:
                    self.best_score = macro_dice
                    self._save("best")
                    logger.info(f"  ★ New best  Dice={macro_dice:.4f}")

        self._save("final")
        self._save_history()
        logger.info(f"\n✅ Done. Best macro Dice={self.best_score:.4f}")

    # ── DataLoader rebuild helpers ────────────────────────────

    def _rebuild_loader(self, classes, cw):
        from dataset.drywall_dataset import MinClassBatchSampler
        from torch.utils.data import WeightedRandomSampler

        bs = self.cfg["training"]["batch_size"]
        nw = self.cfg["training"]["num_workers"]

        if "taping" in classes:
            sampler = MinClassBatchSampler(
                self.train_ds, bs, cw, minority_class="taping", min_per_batch=1
            )
            self.train_loader = DataLoader(
                self.train_ds, batch_sampler=sampler,
                num_workers=nw, pin_memory=True,
            )
        else:
            weights = torch.tensor(
                [cw.get(self.train_ds.df.iloc[i]["type"], 1.0)
                 for i in range(len(self.train_ds))], dtype=torch.float,
            )
            sampler = WeightedRandomSampler(weights, len(self.train_ds), replacement=True)
            self.train_loader = DataLoader(
                self.train_ds, batch_size=bs, sampler=sampler,
                num_workers=nw, pin_memory=True, drop_last=True,
            )

    def _rebuild_loader_hem(self, cw):
        sampler = self.hem.get_sampler()
        bs = self.cfg["training"]["batch_size"]
        nw = self.cfg["training"]["num_workers"]
        self.train_loader = DataLoader(
            self.train_ds, batch_size=bs, sampler=sampler,
            num_workers=nw, pin_memory=True, drop_last=True,
        )

    # ── Train epoch ───────────────────────────────────────────

    def _train_epoch(self, optimizer, hem_ids, hem_scores):
        self.model.train()
        total_loss = 0.0
        grad_acc   = self.cfg["training"]["grad_accumulation"]
        threshold  = self.cfg["training"].get("mask_threshold", 0.5)
        optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="  train", leave=False)
        for step, batch in enumerate(pbar):
            images       = batch["image"].to(self.model.device, dtype=self.model.dtype)
            masks        = batch["mask"].to(self.model.device)
            input_points = batch["input_points"].to(self.model.device)
            input_labels = batch["input_labels"].to(self.model.device)

            with autocast(enabled=(self.model.dtype == torch.float16)):
                pred_all = self.model.forward_all_masks(
                    images, input_points, input_labels,
                    target_size=tuple(masks.shape[-2:]),
                )
                loss, breakdown, best_pred = topk_mask_loss(
                    pred_all, masks, self.criterion, k=self.top_k
                )
                loss = loss / grad_acc

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_acc == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                for i, img_id in enumerate(batch["image_id"]):
                    d = dice_score(best_pred[i:i+1], masks[i:i+1], threshold).item()
                    hem_ids.append(img_id)
                    hem_scores.append(d)

            total_loss += breakdown["total"]
            pbar.set_postfix({"loss": f"{breakdown['total']:.4f}"})

        return total_loss / max(len(self.train_loader), 1)

    # ── Eval epoch ────────────────────────────────────────────

    def _eval_epoch(self):
        self.model.eval()
        acc       = MetricAccumulator(["crack", "taping"])
        threshold = self.cfg["training"].get("mask_threshold", 0.5)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  val  ", leave=False):
                images       = batch["image"].to(self.model.device, dtype=self.model.dtype)
                masks        = batch["mask"].to(self.model.device)
                input_points = batch["input_points"].to(self.model.device)
                input_labels = batch["input_labels"].to(self.model.device)
                labels       = batch["label"]

                with autocast(enabled=(self.model.dtype == torch.float16)):
                    pred = self.model(
                        images, input_points, input_labels,
                        target_size=tuple(masks.shape[-2:]),
                    )

                for i, label in enumerate(labels):
                    acc.update(pred[i:i+1], masks[i:i+1], label, threshold)

        return acc.compute()

    # ── Logging / checkpointing ───────────────────────────────

    def _log(self, epoch, train_loss, val_metrics, lr, elapsed):
        self.history["train_loss"].append(train_loss)
        self.history["val_dice_crack"].append(val_metrics["crack"]["dice"])
        self.history["val_dice_taping"].append(val_metrics["taping"]["dice"])
        self.history["val_dice_macro"].append(val_metrics["macro"]["dice"])
        self.history["val_miou_crack"].append(val_metrics["crack"]["miou"])
        self.history["val_miou_taping"].append(val_metrics["taping"]["miou"])
        self.history["val_miou_macro"].append(val_metrics["macro"]["miou"])
        self.history["lr"].append(lr)
        logger.info(
            f"Epoch {epoch:03d} │ loss={train_loss:.4f}  "
            f"crack={val_metrics['crack']['dice']:.4f}  "
            f"taping={val_metrics['taping']['dice']:.4f}  "
            f"miou={val_metrics['macro']['miou']:.4f}  "
            f"lr={lr:.2e}  t={elapsed:.0f}s"
        )

    def _save(self, tag):
        self.model.save(self.ckpt_dir / tag)

    def _save_history(self):
        with open(self.ckpt_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)