"""Checkpoint persistence and model-selection helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evaluation.metrics import macro_dice

logger = logging.getLogger(__name__)

CHECKPOINT_SELECTION_METRIC = "val_macro_dice"
MACRO_DICE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CheckpointSelectionResult:
    """Outcome of a best-checkpoint comparison."""

    is_best: bool
    previous_best: float
    current_score: float
    reason: str


@dataclass(frozen=True)
class CheckpointBundle:
    """Loaded checkpoint contents for resume and evaluation."""

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    epoch: int
    val_metrics: dict[str, Any]
    metadata: dict[str, Any]


def _validate_state_dict(state_dict: dict[str, Any], name: str) -> None:
    if not isinstance(state_dict, dict):
        raise TypeError(f"{name} must be a dictionary.")
    if not state_dict:
        raise ValueError(f"{name} cannot be empty.")


def should_save_best_checkpoint(
    current_val_macro_dice: float,
    best_val_macro_dice: float,
) -> CheckpointSelectionResult:
    """Decide whether the current epoch qualifies as best.

    Selection uses validation macro Dice only. Test metrics must never
    influence this function.
    """
    current = float(current_val_macro_dice)
    previous = float(best_val_macro_dice)
    is_best = current > previous + MACRO_DICE_TOLERANCE
    reason = (
        "validation macro Dice improved"
        if is_best
        else "validation macro Dice did not improve"
    )
    return CheckpointSelectionResult(
        is_best=is_best,
        previous_best=previous,
        current_score=current,
        reason=reason,
    )


def build_checkpoint_payload(
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None,
    epoch: int,
    val_metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a serializable checkpoint dictionary."""
    _validate_state_dict(model_state_dict, "model_state_dict")
    if CHECKPOINT_SELECTION_METRIC not in val_metrics:
        raise KeyError(
            f"val_metrics must include '{CHECKPOINT_SELECTION_METRIC}' for checkpoint policy."
        )
    return {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "epoch": int(epoch),
        "val_metrics": val_metrics,
        "metadata": metadata or {},
        "selection_metric": CHECKPOINT_SELECTION_METRIC,
        "selection_score": float(val_metrics[CHECKPOINT_SELECTION_METRIC]),
    }


def save_checkpoint(path: Path | str, payload: dict[str, Any]) -> None:
    """Save a checkpoint payload to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    logger.info("Saved checkpoint: %s", destination)


def save_best_checkpoint(
    path: Path | str,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None,
    epoch: int,
    val_metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    current_best_score: float = float("-inf"),
) -> CheckpointSelectionResult:
    """Save ``best_model.pt`` only when validation macro Dice improves."""
    if CHECKPOINT_SELECTION_METRIC not in val_metrics:
        raise KeyError(f"val_metrics must contain '{CHECKPOINT_SELECTION_METRIC}'.")

    decision = should_save_best_checkpoint(
        val_metrics[CHECKPOINT_SELECTION_METRIC],
        current_best_score,
    )
    if decision.is_best:
        payload = build_checkpoint_payload(
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            epoch=epoch,
            val_metrics=val_metrics,
            metadata=metadata,
        )
        save_checkpoint(path, payload)
    return decision


def save_last_checkpoint(
    path: Path | str,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None,
    epoch: int,
    val_metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the latest epoch checkpoint without selection logic."""
    payload = build_checkpoint_payload(
        model_state_dict=model_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        epoch=epoch,
        val_metrics=val_metrics,
        metadata=metadata,
    )
    save_checkpoint(path, payload)


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {source} is not a dictionary.")
    return payload


def resume_training(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> CheckpointBundle:
    """Load checkpoint fields required to resume training."""
    payload = load_checkpoint(path, map_location=map_location)
    required = ("model_state_dict", "epoch", "val_metrics")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {missing}")

    return CheckpointBundle(
        model_state_dict=payload["model_state_dict"],
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        epoch=int(payload["epoch"]),
        val_metrics=dict(payload["val_metrics"]),
        metadata=dict(payload.get("metadata", {})),
    )


def macro_dice_from_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_background: bool = True,
) -> float:
    """Compute validation macro Dice for checkpoint logging."""
    return float(macro_dice(prediction, target, ignore_background=ignore_background).item())
