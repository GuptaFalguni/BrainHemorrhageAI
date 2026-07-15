"""Learning-rate scheduler factory for BrainHemorrhageAI training."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> LRScheduler | None:
    """Build a learning-rate scheduler from training configuration.

    Args:
        optimizer: Optimizer controlled by the scheduler.
        config: Training configuration dictionary.

    Returns:
        Scheduler instance, or ``None`` when scheduling is disabled.
    """
    scheduler_name = str(config.get("scheduler", "cosine")).lower()
    if scheduler_name in {"none", "null", "disabled"}:
        return None

    if scheduler_name == "cosine":
        t_max = int(config.get("epochs", 1))
        return CosineAnnealingLR(optimizer, T_max=t_max)

    raise ValueError(
        f"Unsupported scheduler: {scheduler_name}. Version 1 supports 'cosine' only."
    )
