"""Loss function factory for BrainHemorrhageAI training."""

from __future__ import annotations

from typing import Any

from monai.losses import DiceCELoss


def build_loss(config: dict[str, Any]) -> DiceCELoss:
    """Build the Version 1 segmentation loss.

    Args:
        config: Training configuration dictionary.

    Returns:
        Configured ``DiceCELoss`` instance.
    """
    loss_name = str(config.get("loss_name", "dice_ce")).lower()
    if loss_name != "dice_ce":
        raise ValueError(f"Unsupported loss_name: {loss_name}. Version 1 supports 'dice_ce' only.")

    return DiceCELoss(
        include_background=bool(config.get("include_background", True)),
        to_onehot_y=True,
        softmax=True,
        lambda_dice=float(config.get("lambda_dice", 1.0)),
        lambda_ce=float(config.get("lambda_ce", 1.0)),
    )
