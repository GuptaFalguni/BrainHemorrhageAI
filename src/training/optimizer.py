"""Optimizer factory for BrainHemorrhageAI training."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import AdamW, Optimizer


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> Optimizer:
    """Build an optimizer from training configuration.

    Args:
        model: Model whose parameters will be optimized.
        config: Training configuration dictionary.

    Returns:
        Configured PyTorch optimizer.
    """
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}. Version 1 supports 'adamw' only.")
