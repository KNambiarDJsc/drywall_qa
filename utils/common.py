import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Union

def set_seed(seed: int = 42) -> None:
    """Fully deterministic seeding for torch, numpy, python, cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stem(path: Union[str, Path]) -> str:
    return Path(path).stem

def to_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Float mask → uint8 PNG mask with values {0, 255}."""
    binary = (mask > threshold).astype(np.uint8) * 255
    return binary


def resize_mask_to_original(
    pred_mask: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Nearest-neighbour upsample mask to original image size."""
    import cv2
    return cv2.resize(
        pred_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]