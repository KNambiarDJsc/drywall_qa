"""train.py — Main training entry point (SAM ViT-H + DoRA).
Usage:
  python train.py --config configs/config.yaml
"""

from __future__ import annotations
import argparse
import os
import sys
import json
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent))
from utils.common import set_seed, get_logger
from dataset.drywall_dataset import build_dataloaders
from models.sam_dora import SAMDoRA
from training.trainer import DrywallTrainer
from utils.visualize import plot_training_curves

logger = get_logger("train")


# ─────────────────────────────────────────────────────────────
# HuggingFace login
# ─────────────────────────────────────────────────────────────

def hf_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("✅ HuggingFace login successful")
        except Exception as e:
            logger.warning(f"HF login failed: {e} — trying anonymous")
    else:
        logger.warning("HF_TOKEN not set — attempting anonymous download")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--metadata",      default="dataset/metadata.csv")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    # HF login before any from_pretrained()
    hf_login()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    logger.info("=" * 60)
    logger.info("Drywall QA — SAM ViT-H + DoRA")
    logger.info(f"Seed: {cfg['seed']} | Batch: {cfg['training']['batch_size']}")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, train_ds = build_dataloaders(
        metadata_csv=args.metadata,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        num_workers=cfg["training"]["num_workers"],
        class_weights=cfg["class_weights"],
        synthetic_crack_prob=cfg["augmentation"]["crack"]["synthetic_crack_prob"],
        guarantee_taping=True,
    )

    # ── Model ─────────────────────────────────────────────────
    model = SAMDoRA(
        model_name=cfg["model"]["name"],
        precision=cfg["model"]["precision"],
        lora_r=cfg["dora"]["r"],
        lora_alpha=cfg["dora"]["lora_alpha"],
        lora_dropout=cfg["dora"]["lora_dropout"],
        use_dora=cfg["dora"]["use_dora"],
    )

    # ── Trainer ───────────────────────────────────────────────
    trainer = DrywallTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        cfg=cfg,
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
    )

    # Phase 0 — baseline
    if not args.skip_baseline:
        baseline = trainer.run_baseline()
        ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "baseline_metrics.json", "w") as f:
            json.dump(baseline, f, indent=2)

    # Phases 1–3 — fine-tune
    trainer.train()

    # Plot training curves
    plot_training_curves(
        trainer.history,
        save_path=Path(cfg["output"]["viz_dir"]) / "training_curves.png",
    )
    logger.info("Done. ✅")


if __name__ == "__main__":
    main()