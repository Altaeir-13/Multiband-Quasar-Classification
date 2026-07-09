from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for training")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> str:
    require_torch()
    if device_name != "auto":
        return device_name
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


class EarlyStopping:
    """Track validation improvements and stop after patience epochs."""

    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.bad_epochs = 0
            return True
        improved = score < self.best_score if self.mode == "min" else score > self.best_score
        if improved:
            self.best_score = score
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience
