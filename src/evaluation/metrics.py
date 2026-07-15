"""Reusable segmentation metrics on PyTorch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

NUM_CLASSES = 6
BACKGROUND_CLASS = 0
HEMORRHAGE_CLASSES: tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_SMOOTH = 1e-7


@dataclass(frozen=True)
class BinaryConfusion:
    """Confusion counts for a one-versus-rest binary segmentation."""

    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    tn: torch.Tensor


@dataclass(frozen=True)
class SegmentationMetrics:
    """Per-class and aggregated segmentation metrics."""

    dice: torch.Tensor
    iou: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    sensitivity: torch.Tensor
    specificity: torch.Tensor
    macro_dice: torch.Tensor
    micro_dice: torch.Tensor


def _validate_inputs(prediction: torch.Tensor, target: torch.Tensor) -> None:
    """Ensure prediction and target are compatible integer label maps."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}."
        )
    if prediction.dtype not in (torch.int64, torch.int32, torch.long):
        raise TypeError(f"prediction must be integer class indices, got {prediction.dtype}.")
    if target.dtype not in (torch.int64, torch.int32, torch.long):
        raise TypeError(f"target must be integer class indices, got {target.dtype}.")


def _class_indices(
    num_classes: int,
    ignore_background: bool,
) -> tuple[int, ...]:
    """Return class indices included in aggregated metrics."""
    classes = tuple(range(num_classes))
    if ignore_background:
        return tuple(index for index in classes if index != BACKGROUND_CLASS)
    return classes


def binary_confusion(
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> BinaryConfusion:
    """Compute TP, FP, FN, TN for binary masks."""
    prediction = prediction_mask.bool()
    target = target_mask.bool()
    tp = torch.logical_and(prediction, target).sum(dtype=torch.float64)
    fp = torch.logical_and(prediction, torch.logical_not(target)).sum(dtype=torch.float64)
    fn = torch.logical_and(torch.logical_not(prediction), target).sum(dtype=torch.float64)
    tn = torch.logical_and(torch.logical_not(prediction), torch.logical_not(target)).sum(
        dtype=torch.float64
    )
    return BinaryConfusion(tp=tp, fp=fp, fn=fn, tn=tn)


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """Compute Dice for binary or multi-class label maps.

    For multi-class inputs, returns per-class Dice with shape ``(num_classes,)``.
    """
    _validate_inputs(prediction, target)
    num_classes = int(max(prediction.max().item(), target.max().item())) + 1
    scores: list[torch.Tensor] = []
    for class_index in range(num_classes):
        pred_mask = prediction == class_index
        target_mask = target == class_index
        confusion = binary_confusion(pred_mask, target_mask)
        numerator = 2.0 * confusion.tp
        denominator = 2.0 * confusion.tp + confusion.fp + confusion.fn + smooth
        scores.append(numerator / denominator)
    return torch.stack(scores)


def _resolve_num_classes(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
) -> int:
    """Return explicit class count, or infer from label maxima."""
    if num_classes is not None:
        return int(num_classes)
    return int(max(prediction.max().item(), target.max().item())) + 1


def iou_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = DEFAULT_SMOOTH,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Compute IoU (Jaccard) per class."""
    _validate_inputs(prediction, target)
    resolved = _resolve_num_classes(prediction, target, num_classes)
    scores: list[torch.Tensor] = []
    for class_index in range(resolved):
        pred_mask = prediction == class_index
        target_mask = target == class_index
        confusion = binary_confusion(pred_mask, target_mask)
        union = confusion.tp + confusion.fp + confusion.fn + smooth
        scores.append(confusion.tp / union)
    return torch.stack(scores)


def precision_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = DEFAULT_SMOOTH,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Compute precision per class."""
    _validate_inputs(prediction, target)
    resolved = _resolve_num_classes(prediction, target, num_classes)
    scores: list[torch.Tensor] = []
    for class_index in range(resolved):
        confusion = binary_confusion(
            prediction == class_index,
            target == class_index,
        )
        scores.append(confusion.tp / (confusion.tp + confusion.fp + smooth))
    return torch.stack(scores)


def recall_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = DEFAULT_SMOOTH,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Compute recall (sensitivity) per class."""
    _validate_inputs(prediction, target)
    resolved = _resolve_num_classes(prediction, target, num_classes)
    scores: list[torch.Tensor] = []
    for class_index in range(resolved):
        confusion = binary_confusion(
            prediction == class_index,
            target == class_index,
        )
        scores.append(confusion.tp / (confusion.tp + confusion.fn + smooth))
    return torch.stack(scores)


def sensitivity_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = DEFAULT_SMOOTH,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Compute sensitivity per class (alias of recall)."""
    return recall_score(prediction, target, smooth=smooth, num_classes=num_classes)


def specificity_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = DEFAULT_SMOOTH,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Compute specificity per class."""
    _validate_inputs(prediction, target)
    resolved = _resolve_num_classes(prediction, target, num_classes)
    scores: list[torch.Tensor] = []
    for class_index in range(resolved):
        confusion = binary_confusion(
            prediction == class_index,
            target == class_index,
        )
        scores.append(confusion.tn / (confusion.tn + confusion.fp + smooth))
    return torch.stack(scores)


def per_class_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """Return Dice for each class index in ``[0, num_classes)``."""
    _validate_inputs(prediction, target)
    scores: list[torch.Tensor] = []
    for class_index in range(num_classes):
        confusion = binary_confusion(
            prediction == class_index,
            target == class_index,
        )
        numerator = 2.0 * confusion.tp
        denominator = 2.0 * confusion.tp + confusion.fp + confusion.fn + smooth
        scores.append(numerator / denominator)
    return torch.stack(scores)


def macro_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_background: bool = True,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """Average per-class Dice over selected classes."""
    dice = per_class_dice(prediction, target, num_classes=num_classes, smooth=smooth)
    selected = _class_indices(num_classes, ignore_background)
    return dice[list(selected)].mean()


def micro_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_background: bool = True,
    smooth: float = DEFAULT_SMOOTH,
) -> torch.Tensor:
    """Compute Dice from aggregated intersections across selected classes."""
    _validate_inputs(prediction, target)
    selected = _class_indices(num_classes, ignore_background)
    total_intersection = torch.tensor(0.0, dtype=torch.float64, device=prediction.device)
    total_pred = torch.tensor(0.0, dtype=torch.float64, device=prediction.device)
    total_target = torch.tensor(0.0, dtype=torch.float64, device=prediction.device)

    for class_index in selected:
        pred_mask = prediction == class_index
        target_mask = target == class_index
        total_intersection += torch.logical_and(pred_mask, target_mask).sum(dtype=torch.float64)
        total_pred += pred_mask.sum(dtype=torch.float64)
        total_target += target_mask.sum(dtype=torch.float64)

    numerator = 2.0 * total_intersection
    denominator = total_pred + total_target + smooth
    return (numerator / denominator).to(dtype=torch.float32)


def compute_segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_background: bool = True,
    smooth: float = DEFAULT_SMOOTH,
) -> SegmentationMetrics:
    """Compute the full segmentation metric bundle."""
    dice = per_class_dice(prediction, target, num_classes=num_classes, smooth=smooth)
    return SegmentationMetrics(
        dice=dice,
        iou=iou_score(prediction, target, smooth=smooth, num_classes=num_classes),
        precision=precision_score(prediction, target, smooth=smooth, num_classes=num_classes),
        recall=recall_score(prediction, target, smooth=smooth, num_classes=num_classes),
        sensitivity=sensitivity_score(prediction, target, smooth=smooth, num_classes=num_classes),
        specificity=specificity_score(prediction, target, smooth=smooth, num_classes=num_classes),
        macro_dice=macro_dice(
            prediction,
            target,
            num_classes=num_classes,
            ignore_background=ignore_background,
            smooth=smooth,
        ),
        micro_dice=micro_dice(
            prediction,
            target,
            num_classes=num_classes,
            ignore_background=ignore_background,
            smooth=smooth,
        ),
    )


def mean_over_classes(
    values: torch.Tensor,
    class_indices: Iterable[int],
) -> torch.Tensor:
    """Average a per-class tensor over the given indices."""
    indices = list(class_indices)
    return values[indices].mean()
