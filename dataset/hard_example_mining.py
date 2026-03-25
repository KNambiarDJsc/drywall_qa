"""dataset/hard_example_mining.py
Hard Example Mining (HEM) sampler.

After each epoch the trainer calls `update(image_ids, dice_scores)`.
Samples with Dice < threshold get their sampling weight multiplied
by `boost_factor`. On the next epoch the DataLoader draws from this
updated distribution — model is forced to focus on failures.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from utils.common import get_logger

logger = get_logger("hem")


class HardExampleMiner:
    """
    Maintains a per-sample weight vector and rebuilds a
    WeightedRandomSampler on demand.

    Workflow:
        miner = HardExampleMiner(dataset, class_weights, ...)
        sampler = miner.get_sampler()           # epoch 1 — plain class weighting

        # after epoch 1 training loop:
        miner.update(image_ids, dice_scores)
        sampler = miner.get_sampler()           # epoch 2 — HEM-boosted
    """

    def __init__(
        self,
        dataset,                         # DrywallDataset
        class_weights: Dict[str, float], # {"crack": 1.0, "taping": 5.4}
        dice_threshold: float = 0.45,
        boost_factor: float = 3.0,
    ):
        self.dataset = dataset
        self.dice_threshold = dice_threshold
        self.boost_factor = boost_factor
        n = len(dataset)

        # Base weights from class imbalance
        self.base_weights = np.array(
            [class_weights.get(dataset.df.iloc[i]["type"], 1.0) for i in range(n)],
            dtype=np.float32,
        )
        # HEM multiplier — starts at 1.0 for all
        self.hem_multiplier = np.ones(n, dtype=np.float32)

        # Map image_id → dataset index for fast lookup
        self._id_to_idx: Dict[str, int] = {
            Path(dataset.df.iloc[i]["image_path"]).stem: i for i in range(n)
        }

        self._n_boosted_last = 0

    # ─────────────────────────────────────────────────────────
    # Update after an epoch
    # ─────────────────────────────────────────────────────────

    def update(self, image_ids: List[str], dice_scores: List[float]) -> None:
        """
        Call after each training epoch.
        image_ids: list of image stem strings (batch["image_id"])
        dice_scores: per-sample Dice computed during that forward pass
        """
        hard_count = 0
        for img_id, d in zip(image_ids, dice_scores):
            idx = self._id_to_idx.get(img_id)
            if idx is None:
                continue
            if d < self.dice_threshold:
                self.hem_multiplier[idx] = self.boost_factor
                hard_count += 1
            else:
                # Decay back towards 1 smoothly
                self.hem_multiplier[idx] = max(
                    1.0, self.hem_multiplier[idx] * 0.8
                )

        self._n_boosted_last = hard_count
        logger.info(
            f"[HEM] boosted {hard_count}/{len(image_ids)} samples "
            f"(Dice < {self.dice_threshold})"
        )

    # ─────────────────────────────────────────────────────────
    # Build sampler
    # ─────────────────────────────────────────────────────────

    def get_sampler(self) -> WeightedRandomSampler:
        combined = self.base_weights * self.hem_multiplier
        return WeightedRandomSampler(
            weights=torch.from_numpy(combined),
            num_samples=len(self.dataset),
            replacement=True,
        )

    @property
    def n_boosted(self) -> int:
        return self._n_boosted_last