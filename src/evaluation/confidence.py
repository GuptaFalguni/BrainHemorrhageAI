"""Deterministic softmax confidence utilities for segmentation outputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from evaluation.metrics import BACKGROUND_CLASS, HEMORRHAGE_CLASSES, NUM_CLASSES


def _validate_probability_tensor(probabilities: torch.Tensor) -> None:
    """Ensure probabilities have shape ``(C, ...)`` and valid range."""
    if probabilities.ndim < 2:
        raise ValueError(
            f"probabilities must have shape (C, ...), got {tuple(probabilities.shape)}."
        )
    if probabilities.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {probabilities.shape[0]}."
        )
    if torch.isnan(probabilities).any():
        raise ValueError("probabilities contain NaN values.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("probabilities must lie in [0, 1].")


def softmax_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """Return per-voxel max softmax confidence.

    Args:
        probabilities: Softmax probabilities with shape ``(C, H, W)`` or ``(C, D, H, W)``.

    Returns:
        Tensor of per-voxel confidence values with spatial shape only.
    """
    _validate_probability_tensor(probabilities)
    return probabilities.max(dim=0).values


def average_softmax_confidence(
    probabilities: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean max-softmax confidence over selected voxels.

    Args:
        probabilities: Softmax probabilities ``(C, ...)``.
        mask: Optional boolean mask selecting voxels. When ``None``, all voxels are used.

    Returns:
        Scalar mean confidence.
    """
    confidence = softmax_confidence(probabilities)
    if mask is None:
        return confidence.mean()
    if mask.shape != confidence.shape:
        raise ValueError(
            f"mask shape {mask.shape} must match confidence shape {confidence.shape}."
        )
    selected = confidence[mask]
    if selected.numel() == 0:
        return torch.tensor(0.0, dtype=confidence.dtype, device=confidence.device)
    return selected.mean()


def class_confidence(
    probabilities: torch.Tensor,
    predicted_labels: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    """Return mean confidence for voxels predicted as ``class_index``.

    When no voxels are assigned to the class, returns ``0.0``.
    """
    _validate_probability_tensor(probabilities)
    confidence = softmax_confidence(probabilities)
    class_mask = predicted_labels == class_index
    return average_softmax_confidence(probabilities, mask=class_mask)


def all_class_confidences(
    probabilities: torch.Tensor,
    predicted_labels: torch.Tensor,
    class_indices: Iterable[int] = HEMORRHAGE_CLASSES,
) -> dict[int, float]:
    """Return class confidence for each requested label."""
    return {
        class_index: float(class_confidence(probabilities, predicted_labels, class_index).item())
        for class_index in class_indices
    }


def hemorrhage_voxel_mask(predicted_labels: torch.Tensor) -> torch.Tensor:
    """Return boolean mask where predicted label is a hemorrhage subtype."""
    mask = torch.zeros_like(predicted_labels, dtype=torch.bool)
    for class_index in HEMORRHAGE_CLASSES:
        mask |= predicted_labels == class_index
    return mask


def study_confidence(
    probabilities: torch.Tensor,
    predicted_labels: torch.Tensor,
) -> torch.Tensor:
    """Return mean confidence over predicted hemorrhage voxels (classes 1-5).

    If no hemorrhage voxels are predicted, returns ``0.0``.
    """
    bleed_mask = hemorrhage_voxel_mask(predicted_labels)
    return average_softmax_confidence(probabilities, mask=bleed_mask)


def confidence_histogram(
    probabilities: torch.Tensor,
    mask: torch.Tensor | None = None,
    bins: int = 20,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a histogram of per-voxel softmax confidence.

    Args:
        probabilities: Softmax probabilities ``(C, ...)``.
        mask: Optional voxel mask.
        bins: Number of histogram bins.
        value_range: Minimum and maximum confidence values.

    Returns:
        Tuple ``(counts, bin_edges)`` as NumPy arrays.
    """
    confidence = softmax_confidence(probabilities)
    if mask is not None:
        confidence = confidence[mask]
    values = confidence.detach().cpu().numpy().ravel()
    counts, bin_edges = np.histogram(values, bins=bins, range=value_range)
    return counts.astype(np.int64), bin_edges.astype(np.float64)


def background_excluded_confidence(
    probabilities: torch.Tensor,
    predicted_labels: torch.Tensor,
) -> torch.Tensor:
    """Return mean confidence excluding voxels predicted as background."""
    non_background = predicted_labels != BACKGROUND_CLASS
    return average_softmax_confidence(probabilities, mask=non_background)
