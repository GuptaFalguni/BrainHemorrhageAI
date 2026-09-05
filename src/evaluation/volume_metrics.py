"""Native-space clinical volume metrics for BHSD segmentation."""

from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

SUBTYPE_LABELS: dict[int, str] = {
    1: "EDH",
    2: "SDH",
    3: "SAH",
    4: "IPH",
    5: "IVH",
}
HEMORRHAGE_LABELS: tuple[int, ...] = tuple(SUBTYPE_LABELS.keys())


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert a NumPy array or PyTorch tensor to NumPy."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _validate_spacing(spacing: Sequence[float]) -> tuple[float, float, float]:
    """Validate native voxel spacing in millimeters."""
    if len(spacing) != 3:
        raise ValueError(f"spacing must have 3 values (x, y, z), got {len(spacing)}.")
    x_spacing, y_spacing, z_spacing = (float(value) for value in spacing)
    if x_spacing <= 0 or y_spacing <= 0 or z_spacing <= 0:
        raise ValueError(f"spacing values must be positive, got {spacing}.")
    return x_spacing, y_spacing, z_spacing


def voxel_volume_mm3(spacing: Sequence[float]) -> float:
    """Return native voxel volume in mm³ from ``(x, y, z)`` spacing."""
    x_spacing, y_spacing, z_spacing = _validate_spacing(spacing)
    return x_spacing * y_spacing * z_spacing


def calculate_volume_ml(
    mask: ArrayLike,
    spacing: Sequence[float],
    label: int,
) -> float:
    """Calculate native-space volume in mL for one label.

    Args:
        mask: 3D integer label volume in native grid (not resampled spacing).
        spacing: Original voxel spacing ``(x, y, z)`` in millimeters.
        label: Class index to measure.

    Returns:
        Volume in milliliters.
    """
    mask_array = _to_numpy(mask)
    voxel_count = int(np.sum(mask_array == label))
    volume_mm3 = voxel_count * voxel_volume_mm3(spacing)
    return volume_mm3 / 1000.0


def per_subtype_volume(
    mask: ArrayLike,
    spacing: Sequence[float],
    labels: Sequence[int] = HEMORRHAGE_LABELS,
) -> dict[int, float]:
    """Return native-space mL volume for each hemorrhage subtype.

    Args:
        mask: 3D integer label volume on the native grid.
        spacing: Original ``(x, y, z)`` spacing in millimeters.
        labels: Subtype labels to include.

    Returns:
        Mapping from label index to volume in mL.
    """
    return {label: calculate_volume_ml(mask, spacing, label) for label in labels}


def absolute_volume_error_ml(
    predicted_volume_ml: float,
    ground_truth_volume_ml: float,
) -> float:
    """Return absolute volume error in mL."""
    return abs(float(predicted_volume_ml) - float(ground_truth_volume_ml))


def relative_volume_error(
    predicted_volume_ml: float,
    ground_truth_volume_ml: float,
    epsilon: float = 1e-6,
) -> float:
    """Return relative volume error ``|pred - gt| / max(gt, epsilon)``."""
    denominator = max(abs(float(ground_truth_volume_ml)), epsilon)
    return absolute_volume_error_ml(predicted_volume_ml, ground_truth_volume_ml) / denominator


def per_subtype_volume_error(
    predicted_mask: ArrayLike,
    ground_truth_mask: ArrayLike,
    spacing: Sequence[float],
    labels: Sequence[int] = HEMORRHAGE_LABELS,
) -> dict[int, dict[str, float]]:
    """Compare predicted and ground-truth subtype volumes on native spacing.

    Returns:
        Mapping ``label -> {pred_ml, gt_ml, abs_error_ml, rel_error}``.
    """
    if _to_numpy(predicted_mask).shape != _to_numpy(ground_truth_mask).shape:
        raise ValueError("predicted_mask and ground_truth_mask must share the same shape.")

    results: dict[int, dict[str, float]] = {}
    for label in labels:
        pred_ml = calculate_volume_ml(predicted_mask, spacing, label)
        gt_ml = calculate_volume_ml(ground_truth_mask, spacing, label)
        results[label] = {
            "pred_ml": pred_ml,
            "gt_ml": gt_ml,
            "abs_error_ml": absolute_volume_error_ml(pred_ml, gt_ml),
            "rel_error": relative_volume_error(pred_ml, gt_ml),
        }
    return results


def total_hemorrhage_volume_ml(
    mask: ArrayLike,
    spacing: Sequence[float],
    labels: Sequence[int] = HEMORRHAGE_LABELS,
) -> float:
    """Sum native-space mL across hemorrhage subtypes."""
    volumes = per_subtype_volume(mask, spacing, labels=labels)
    return float(sum(volumes.values()))


def format_subtype_name(label: int) -> str:
    """Return human-readable subtype name for a label index."""
    return SUBTYPE_LABELS.get(label, f"label_{label}")


def subtype_volume_table(
    volume_map: Mapping[int, float],
) -> list[dict[str, float | int | str]]:
    """Convert a label->mL mapping into serializable table rows."""
    rows: list[dict[str, float | int | str]] = []
    for label, volume_ml in sorted(volume_map.items()):
        rows.append(
            {
                "label": label,
                "subtype": format_subtype_name(label),
                "volume_ml": float(volume_ml),
            }
        )
    return rows
