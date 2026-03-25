"""train.py — Main training entry point.
Usage:
  python train.py --config configs/config.yaml
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml
import json

sys.path.append(str(Path(__file__).parent))
from utils.common import set_seed, get_logger
from dataset.drywall_dataset import build_dataloaders
from models.sam3_dora import SAM3DoRA
from training.trainer import DrywallTrainer
from utils.visualize import plot_training_curves

logger = get_logger("train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--metadata", default="dataset/metadata.csv",
                        help="Path to metadata.csv (output of preprocessing)")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    logger.info("=" * 60)
    logger.info("Drywall QA — SAM3 + DoRA Fine-tune")
    logger.info(f"Seed: {cfg['seed']} | Batch: {cfg['training']['batch_size']}")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, train_ds = build_dataloaders(
        metadata_csv=args.metadata,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        num_workers=cfg["training"]["num_workers"],
        class_weights=cfg["class_weights"],
        prompt_dropout_prob=cfg["prompt_engine"]["dropout_prob"],
        prompt_noise_prob=cfg["prompt_engine"]["noise_prob"],
        synthetic_crack_prob=cfg["augmentation"]["crack"]["synthetic_crack_prob"],
        guarantee_taping=True,
    )

    # ── Model ─────────────────────────────────────────────────
    model = SAM3DoRA(
        model_name=cfg["model"]["name"],
        precision=cfg["model"]["precision"],
        lora_r=cfg["dora"]["r"],
        lora_alpha=cfg["dora"]["lora_alpha"],
        lora_dropout=cfg["dora"]["lora_dropout"],
        use_dora=cfg["dora"]["use_dora"],
        target_modules=cfg["dora"]["target_modules"],
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

    # Phase 0: baseline
    if not args.skip_baseline:
        baseline = trainer.run_baseline()
        with open(Path(cfg["output"]["checkpoint_dir"]) / "baseline_metrics.json", "w") as f:
            json.dump(baseline, f, indent=2)

    # Phase 1 + 2: fine-tune
    trainer.train()

    # ── Plot curves ───────────────────────────────────────────
    plot_training_curves(
        trainer.history,
        save_path=Path(cfg["output"]["viz_dir"]) / "training_curves.png",
    )
    logger.info("Done. ✅")


if __name__ == "__main__":
    main()