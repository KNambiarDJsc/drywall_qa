"""train.py — Grounded SAM 2 + DoRA training entry point."""

from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent))
from utils.common import set_seed, get_logger, get_device
from dataset.drywall_dataset import build_dataloaders
from models.grounded_sam2 import GroundedSAM2
from training.trainer import DrywallTrainer
from utils.visualize import plot_training_curves

logger = get_logger("train")


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("✅ HuggingFace login successful")
        except Exception as e:
            logger.warning(f"HF login failed: {e}")
    else:
        logger.warning("HF_TOKEN not set — attempting anonymous download")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--metadata",      default="dataset/metadata.csv")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    hf_login()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    logger.info("=" * 60)
    logger.info("Drywall QA — Grounded SAM 2 + DoRA")
    logger.info(f"DINO: {cfg['model']['grounding_dino']} (frozen)")
    logger.info(f"SAM2: {cfg['model']['sam2']} (DoRA fine-tuned)")
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
    model = GroundedSAM2(cfg=cfg, device=get_device())

    # ── Trainer ───────────────────────────────────────────────
    trainer = DrywallTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        cfg=cfg,
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
    )

    if not args.skip_baseline:
        baseline = trainer.run_baseline()
        ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "baseline_metrics.json", "w") as f:
            json.dump(baseline, f, indent=2)

    trainer.train()

    plot_training_curves(
        trainer.history,
        save_path=Path(cfg["output"]["viz_dir"]) / "training_curves.png",
    )
    logger.info("Done ✅")


if __name__ == "__main__":
    main()