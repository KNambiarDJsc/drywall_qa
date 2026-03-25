"""training/trainer.py
Curriculum learning trainer with:
  • 3-phase curriculum (class filter + resolution + encoder freeze)
  • Hard Example Mining (HEM) — dynamic weight boost after each epoch
  • Multi-mask supervision — loss over top-K SAM3 mask outputs
  • Smart encoder unfreeze at Phase 3
  • torch.compile support
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

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
# Multi-mask supervision helper
# ─────────────────────────────────────────────────────────────

def topk_mask_loss(
    pred_all: torch.Tensor,   # [B, K, H, W] — all K mask outputs from SAM3
    target: torch.Tensor,     # [B, 1, H, W]
    criterion: HybridLoss,
    k: int = 3,
) -> tuple[torch.Tensor, dict]:
    """
    Compute loss for each of K masks and take the minimum per sample.
    Forces SAM3's "best mask" prediction path toward the GT.
    """
    B, K_actual, H, W = pred_all.shape
    k = min(k, K_actual)
    losses = []
    for i in range(k):
        l, _ = criterion(pred_all[:, i:i+1, :, :], target)
        losses.append(l.unsqueeze(0))
    # Min over K masks — pick the best prediction
    stacked = torch.cat(losses)
    loss = stacked.min()
    # For metrics use the best-scoring mask
    best_idx = stacked.argmin().item()
    best_pred = pred_all[:, best_idx:best_idx+1, :, :]
    _, breakdown = criterion(best_pred, target)
    return loss, breakdown, best_pred


# ─────────────────────────────────────────────────────────────
# Encoder freeze / unfreeze
# ─────────────────────────────────────────────────────────────

def _set_encoder_frozen(model, frozen: bool) -> None:
    """Freeze/unfreeze image encoder (keep mask decoder + DoRA always trainable)."""
    sam3_base = model.sam3.base_model if hasattr(model.sam3, "base_model") else model.sam3
    if hasattr(sam3_base, "image_encoder"):
        for p in sam3_base.image_encoder.parameters():
            p.requires_grad_(not frozen)
        # But always keep DoRA adapter params trainable
        for name, p in model.sam3.named_parameters():
            if "lora_" in name or "dora_" in name:
                p.requires_grad_(True)
    status = "FROZEN" if frozen else "UNFROZEN (last layers)"
    logger.info(f"  Encoder: {status}")


def _unfreeze_last_encoder_layers(model, n_layers: int = 4) -> None:
    """Selectively unfreeze the last N transformer blocks in the encoder."""
    sam3_base = model.sam3.base_model if hasattr(model.sam3, "base_model") else model.sam3
    if not hasattr(sam3_base, "image_encoder"):
        return
    blocks = getattr(sam3_base.image_encoder, "blocks", [])
    if not blocks:
        return
    for block in blocks[-n_layers:]:
        for p in block.parameters():
            p.requires_grad_(True)
    logger.info(f"  Unfroze last {n_layers} encoder blocks")


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class DrywallTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        train_dataset,
        cfg: dict,
        checkpoint_dir: str | Path = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.train_ds     = train_dataset
        self.cfg = cfg
        self.ckpt_dir = ensure_dir(checkpoint_dir)

        loss_cfg = cfg.get("loss", {})
        self.criterion = HybridLoss(
            focal_weight=loss_cfg.get("focal_weight", 0.35),
            tversky_weight=loss_cfg.get("tversky_weight", 0.55),
            boundary_weight=loss_cfg.get("boundary_weight", 0.10),
            focal_alpha=loss_cfg.get("focal_alpha", 0.50),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.70),
            tversky_beta=loss_cfg.get("tversky_beta", 0.30),
        )

        self.top_k = cfg["training"].get("top_k_masks", 3)
        self.scaler = GradScaler(enabled=(model.dtype == torch.float16))

        # HEM
        hem_cfg = cfg.get("hem", {})
        self.hem_enabled = hem_cfg.get("enabled", True)
        self.hem = HardExampleMiner(
            train_dataset,
            class_weights=cfg.get("class_weights", {"crack": 1.0, "taping": 5.4}),
            dice_threshold=hem_cfg.get("dice_threshold", 0.45),
            boost_factor=hem_cfg.get("boost_factor", 3.0),
        ) if self.hem_enabled else None

        # torch.compile
        if cfg["model"].get("torch_compile", False):
            logger.info("torch.compile() enabled")
            self.model.sam3 = torch.compile(self.model.sam3)

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_dice_crack": [], "val_dice_taping": [], "val_dice_macro": [],
            "val_miou_crack": [], "val_miou_taping": [], "val_miou_macro": [],
            "lr": [],
        }
        self.best_score = 0.0

    # ─────────────────────────────────────────────────────────
    # Phase 0 — baseline
    # ─────────────────────────────────────────────────────────

    def run_baseline(self) -> dict:
        logger.info("─" * 55)
        logger.info("Phase 0 — Zero-shot baseline")
        metrics = self._eval_epoch()
        logger.info(
            f"Baseline  Dice crack={metrics['crack']['dice']:.4f}  "
            f"taping={metrics['taping']['dice']:.4f}  "
            f"macro={metrics['macro']['dice']:.4f}"
        )
        return metrics

    # ─────────────────────────────────────────────────────────
    # Main curriculum training loop
    # ─────────────────────────────────────────────────────────

    def train(self) -> None:
        curriculum = self.cfg["training"]["curriculum"]

        for phase_cfg in curriculum:
            name          = phase_cfg["name"]
            epochs        = phase_cfg["epochs"]
            lr            = phase_cfg["lr"]
            classes       = phase_cfg.get("classes", ["crack", "taping"])
            img_size      = phase_cfg.get("image_size", self.cfg["data"]["image_size"])
            freeze_enc    = phase_cfg.get("freeze_encoder", True)
            hem_start     = phase_cfg.get("hem_start_epoch", 999)

            logger.info(f"\n{'═'*55}\n{name}")
            logger.info(f"  epochs={epochs}  lr={lr}  classes={classes}  "
                        f"img_size={img_size}  freeze_enc={freeze_enc}")

            # Curriculum adjustments
            self.train_ds.reload_with_classes(classes)
            self.train_ds.set_image_size(img_size)

            # Encoder freeze strategy
            _set_encoder_frozen(self.model, freeze_enc)
            if not freeze_enc:
                _unfreeze_last_encoder_layers(self.model, n_layers=4)

            # Rebuild sampler (class composition may have changed)
            cw = self.cfg.get("class_weights", {"crack": 1.0, "taping": 5.4})
            weights = torch.tensor(
                [cw.get(self.train_ds.df.iloc[i]["type"], 1.0) for i in range(len(self.train_ds))],
                dtype=torch.float,
            )
            from torch.utils.data import WeightedRandomSampler
            from dataset.drywall_dataset import MinClassBatchSampler

            if "taping" in classes:
                sampler = MinClassBatchSampler(
                    self.train_ds, self.cfg["training"]["batch_size"], cw,
                    minority_class="taping", min_per_batch=1,
                )
                self.train_loader = DataLoader(
                    self.train_ds,
                    batch_sampler=sampler,
                    num_workers=self.cfg["training"]["num_workers"],
                    pin_memory=True,
                )
            else:
                weights = torch.tensor(
                    [cw.get(self.train_ds.df.iloc[i]["type"], 1.0) for i in range(len(self.train_ds))],
                    dtype=torch.float,
                )
                sampler = WeightedRandomSampler(weights, len(self.train_ds), replacement=True)
                self.train_loader = DataLoader(
                    self.train_ds,
                    batch_size=self.cfg["training"]["batch_size"],
                    sampler=sampler,
                    num_workers=self.cfg["training"]["num_workers"],
                    pin_memory=True, drop_last=True,
                )

            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=self.cfg["training"].get("weight_decay", 1e-4),
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

            for epoch in range(1, epochs + 1):
                t0 = time.time()

                # ── HEM: update sampler after hem_start_epoch ──
                if self.hem_enabled and epoch > hem_start:
                    sampler = self.hem.get_sampler()
                    self.train_loader = DataLoader(
                        self.train_ds,
                        batch_size=self.cfg["training"]["batch_size"],
                        sampler=sampler,
                        num_workers=self.cfg["training"]["num_workers"],
                        pin_memory=True, drop_last=True,
                    )

                hem_ids, hem_scores = [], []
                train_loss = self._train_epoch(optimizer, hem_ids, hem_scores)
                val_metrics = self._eval_epoch()
                scheduler.step()

                # Update HEM weights for next epoch
                if self.hem_enabled and hem_ids:
                    self.hem.update(hem_ids, hem_scores)

                macro_dice = val_metrics["macro"]["dice"]
                self._log_epoch(epoch, train_loss, val_metrics, scheduler.get_last_lr()[0], time.time()-t0)

                if macro_dice > self.best_score:
                    self.best_score = macro_dice
                    self._save_checkpoint("best")
                    logger.info(f"  ★ New best  macro Dice={macro_dice:.4f}")

        self._save_checkpoint("final")
        self._save_history()
        logger.info(f"\n✅ Training complete. Best macro Dice={self.best_score:.4f}")

    # ─────────────────────────────────────────────────────────
    # Train epoch
    # ─────────────────────────────────────────────────────────

    def _train_epoch(
        self, optimizer, hem_ids: list, hem_scores: list
    ) -> float:
        self.model.train()
        total_loss = 0.0
        grad_acc = self.cfg["training"]["grad_accumulation"]
        threshold = self.cfg["training"].get("mask_threshold", 0.5)
        optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="  train", leave=False)
        for step, batch in enumerate(pbar):
            images  = batch["image"].to(self.model.device, dtype=self.model.dtype)
            masks   = batch["mask"].to(self.model.device)
            prompts = batch["prompt"]

            with autocast(enabled=(self.model.dtype == torch.float16)):
                # Get ALL mask outputs for multi-mask supervision
                pred_all = self.model.forward_all_masks(images, prompts)
                if pred_all.shape[-2:] != masks.shape[-2:]:
                    pred_all = F.interpolate(
                        pred_all.view(-1, 1, *pred_all.shape[-2:]),
                        size=masks.shape[-2:], mode="bilinear", align_corners=False
                    ).view(pred_all.shape[0], -1, *masks.shape[-2:])

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

            # Collect per-sample Dice for HEM
            with torch.no_grad():
                for i, img_id in enumerate(batch["image_id"]):
                    d = dice_score(
                        best_pred[i:i+1], masks[i:i+1], threshold
                    ).item()
                    hem_ids.append(img_id)
                    hem_scores.append(d)

            total_loss += breakdown["total"]
            pbar.set_postfix({"loss": f"{breakdown['total']:.4f}"})

        return total_loss / max(len(self.train_loader), 1)

    # ─────────────────────────────────────────────────────────
    # Eval epoch
    # ─────────────────────────────────────────────────────────

    def _eval_epoch(self) -> dict:
        self.model.eval()
        acc = MetricAccumulator(["crack", "taping"])
        threshold = self.cfg["training"].get("mask_threshold", 0.5)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  val  ", leave=False):
                images  = batch["image"].to(self.model.device, dtype=self.model.dtype)
                masks   = batch["mask"].to(self.model.device)
                prompts = batch["prompt"]
                labels  = batch["label"]

                with autocast(enabled=(self.model.dtype == torch.float16)):
                    pred = self.model(images, prompts)
                    if pred.shape[-2:] != masks.shape[-2:]:
                        pred = F.interpolate(
                            pred, size=masks.shape[-2:], mode="bilinear", align_corners=False
                        )

                for i, label in enumerate(labels):
                    acc.update(pred[i:i+1], masks[i:i+1], label, threshold)

        return acc.compute()

    # ─────────────────────────────────────────────────────────
    # Logging / checkpointing
    # ─────────────────────────────────────────────────────────

    def _log_epoch(self, epoch, train_loss, val_metrics, lr, elapsed):
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

    def _save_checkpoint(self, tag: str) -> None:
        self.model.save(self.ckpt_dir / tag)

    def _save_history(self) -> None:
        with open(self.ckpt_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)