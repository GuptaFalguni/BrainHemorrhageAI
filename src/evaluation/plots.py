"""Non-interactive PNG plots for experiment reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.volume_metrics import SUBTYPE_LABELS


def _ensure_parent(path: Path | str) -> Path:
    """Create parent directories for an output file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _save_figure(fig: plt.Figure, output_path: Path | str) -> Path:
    """Save and close a Matplotlib figure."""
    destination = _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_loss_curve(
    epochs: Sequence[int],
    train_loss: Sequence[float],
    val_loss: Sequence[float] | None = None,
    output_path: Path | str = "loss_curve.png",
) -> Path:
    """Plot training (and optional validation) loss over epochs."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_loss, label="train loss", linewidth=2)
    if val_loss is not None:
        axis.plot(epochs, val_loss, label="val loss", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training Loss Curve")
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _save_figure(fig, output_path)


def plot_dice_curve(
    epochs: Sequence[int],
    macro_dice: Sequence[float],
    output_path: Path | str = "dice_curve.png",
) -> Path:
    """Plot validation macro Dice over epochs."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, macro_dice, label="val macro Dice", linewidth=2, color="#2a9d8f")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Macro Dice")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Validation Macro Dice")
    axis.grid(True, alpha=0.3)
    axis.legend()
    return _save_figure(fig, output_path)


def plot_per_class_dice(
    class_dice: Mapping[int, float],
    output_path: Path | str = "per_class_dice.png",
    title: str = "Per-Class Dice",
) -> Path:
    """Plot per-class Dice as a bar chart."""
    labels = [SUBTYPE_LABELS.get(label, f"C{label}") for label in sorted(class_dice.keys())]
    values = [float(class_dice[label]) for label in sorted(class_dice.keys())]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, color="#457b9d")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Dice")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    return _save_figure(fig, output_path)


def plot_volume_error(
    subtype_errors: Mapping[int, float],
    output_path: Path | str = "volume_error.png",
    title: str = "Absolute Volume Error (mL)",
) -> Path:
    """Plot absolute volume error per hemorrhage subtype."""
    labels = [SUBTYPE_LABELS.get(label, f"C{label}") for label in sorted(subtype_errors.keys())]
    values = [float(subtype_errors[label]) for label in sorted(subtype_errors.keys())]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, color="#e76f51")
    axis.set_ylabel("Absolute error (mL)")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    return _save_figure(fig, output_path)


def plot_confidence_histogram(
    counts: Sequence[int] | np.ndarray,
    bin_edges: Sequence[float] | np.ndarray,
    output_path: Path | str = "confidence_histogram.png",
    title: str = "Softmax Confidence Histogram",
) -> Path:
    """Plot a precomputed confidence histogram."""
    counts_array = np.asarray(counts, dtype=np.float64)
    edges_array = np.asarray(bin_edges, dtype=np.float64)
    widths = np.diff(edges_array)
    centers = edges_array[:-1] + widths / 2.0

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(centers, counts_array, width=widths * 0.9, align="center", color="#6a4c93")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Voxel count")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    return _save_figure(fig, output_path)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]] | np.ndarray,
    class_labels: Sequence[str],
    output_path: Path | str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> Path:
    """Plot a confusion matrix heatmap."""
    data = np.asarray(matrix, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(data, interpolation="nearest", cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_xticks(range(len(class_labels)))
    axis.set_yticks(range(len(class_labels)))
    axis.set_xticklabels(class_labels, rotation=45, ha="right")
    axis.set_yticklabels(class_labels)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save_figure(fig, output_path)


def plot_lr_curve(
    epochs: Sequence[int],
    learning_rates: Sequence[float],
    output_path: Path | str = "lr_curve.png",
    title: str = "Learning Rate Schedule",
) -> Path:
    """Plot learning rate over epochs."""
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, learning_rates, linewidth=2, color="#264653")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Learning Rate")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    return _save_figure(fig, output_path)


def plot_volume_error_distribution(
    volume_errors: Sequence[float],
    output_path: Path | str = "volume_error_distribution.png",
    title: str = "Total Hemorrhage Volume Error Distribution (mL)",
) -> Path:
    """Plot a histogram of per-scan total volume errors."""
    values = np.asarray(volume_errors, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=min(20, max(5, len(values))), color="#e76f51", edgecolor="white")
    axis.set_xlabel("Absolute volume error (mL)")
    axis.set_ylabel("Scan count")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)
    return _save_figure(fig, output_path)


def plot_prediction_example(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path | str,
    title: str = "Prediction Example",
) -> Path:
    """Plot center-slice intensity, ground truth, and prediction side by side."""
    if image.ndim == 3:
        display_image = image[1]
    else:
        display_image = image

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display_image, cmap="gray")
    axes[0].set_title("Center slice")
    axes[0].axis("off")

    axes[1].imshow(ground_truth, cmap="tab10", vmin=0, vmax=5)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="tab10", vmin=0, vmax=5)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.suptitle(title)
    return _save_figure(fig, output_path)


def plot_scan_examples(
    examples: Sequence[dict[str, object]],
    output_path: Path | str,
    title: str,
) -> Path:
    """Plot a grid of prediction examples for best or worst scans."""
    count = len(examples)
    fig, axes = plt.subplots(count, 3, figsize=(12, 3.5 * count))
    if count == 1:
        axes = np.array([axes])

    for row_index, example in enumerate(examples):
        image = np.asarray(example["image"])
        ground_truth = np.asarray(example["ground_truth"])
        prediction = np.asarray(example["prediction"])
        display_image = image[1] if image.ndim == 3 else image

        axes[row_index, 0].imshow(display_image, cmap="gray")
        axes[row_index, 0].set_ylabel(str(example.get("filename", "")))
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(ground_truth, cmap="tab10", vmin=0, vmax=5)
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(prediction, cmap="tab10", vmin=0, vmax=5)
        axes[row_index, 2].axis("off")

    for column, label in enumerate(["Center slice", "Ground truth", "Prediction"]):
        axes[0, column].set_title(label)

    fig.suptitle(title)
    return _save_figure(fig, output_path)
