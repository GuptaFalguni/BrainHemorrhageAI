"""Publication-quality overlay figures for inference outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from evaluation.volume_metrics import HEMORRHAGE_LABELS, SUBTYPE_LABELS
from preprocessing.preprocess_volume import load_preprocessing_config, resolve_clip_bounds

AXIAL_AXIS = 2
SAGITTAL_AXIS = 0
CORONAL_AXIS = 1

LABEL_COLORS: dict[int, str] = {
    1: "#ff0000",  # EDH red
    2: "#ffff00",  # SDH yellow
    3: "#00ffff",  # SAH cyan
    4: "#00ff00",  # IPH green
    5: "#ff00ff",  # IVH magenta
}


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _brain_window_ct(raw_hu: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Convert HU volume to display-ready grayscale in [0, 1]."""
    clipped = np.clip(raw_hu, clip_min, clip_max)
    scale = clip_max - clip_min
    if scale <= 0:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - clip_min) / scale).astype(np.float32)


def _label_colormap() -> mcolors.ListedColormap:
    colors = ["#00000000"] + [LABEL_COLORS[label] for label in range(1, 6)]
    return mcolors.ListedColormap(colors)


def _masked_overlay(ax: plt.Axes, ct_slice: np.ndarray, label_slice: np.ndarray, alpha: float = 0.55) -> None:
    ax.imshow(ct_slice.T, cmap="gray", origin="lower")
    masked = np.ma.masked_where(label_slice == 0, label_slice)
    ax.imshow(masked.T, cmap=_label_colormap(), origin="lower", alpha=alpha, vmin=0, vmax=5)
    ax.axis("off")


def _error_slice(prediction_slice: np.ndarray, ground_truth_slice: np.ndarray) -> np.ndarray:
    """Build an error map: 1 false positive, 2 false negative, 3 correct foreground."""
    error = np.zeros_like(prediction_slice, dtype=np.int32)
    pred_fg = np.isin(prediction_slice, list(HEMORRHAGE_LABELS))
    gt_fg = np.isin(ground_truth_slice, list(HEMORRHAGE_LABELS))
    error[np.logical_and(pred_fg, np.logical_not(gt_fg))] = 1
    error[np.logical_and(gt_fg, np.logical_not(pred_fg))] = 2
    error[np.logical_and(pred_fg, gt_fg)] = 3
    return error


def _display_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == AXIAL_AXIS:
        return volume[:, :, index]
    if axis == SAGITTAL_AXIS:
        return volume[index, :, :]
    return volume[:, index, :]


def _legend_handles(labels_present: list[int]) -> list[plt.Line2D]:
    handles: list[plt.Line2D] = []
    for label in labels_present:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=LABEL_COLORS[label],
                markersize=10,
                label=f"{label} = {SUBTYPE_LABELS[label]}",
            )
        )
    return handles


def render_overlay_figures(
    raw_hu: np.ndarray,
    prediction_volume: np.ndarray,
    ground_truth_volume: np.ndarray | None,
    spacing: tuple[float, float, float],
    largest_slice: int,
    preprocessing_config_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Render required overlay PNG artifacts."""
    config = load_preprocessing_config(preprocessing_config_path)
    clip_min, clip_max = resolve_clip_bounds(config)
    ct_display = _brain_window_ct(raw_hu, clip_min, clip_max)

    pred_slice = prediction_volume[:, :, largest_slice]
    ct_slice = ct_display[:, :, largest_slice]
    labels_present = sorted(
        int(value) for value in np.unique(prediction_volume) if int(value) in HEMORRHAGE_LABELS
    )
    if ground_truth_volume is not None:
        gt_labels = sorted(
            int(value) for value in np.unique(ground_truth_volume) if int(value) in HEMORRHAGE_LABELS
        )
        labels_present = sorted(set(labels_present) | set(gt_labels))

    overlay_path = _ensure_parent(output_dir / "prediction_overlay.png")
    has_gt = ground_truth_volume is not None
    columns = 5 if has_gt else 3
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4.5))
    if columns == 1:
        axes = [axes]

    panel_index = 0
    axes[panel_index].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[panel_index].set_title("Original CT")
    axes[panel_index].axis("off")
    panel_index += 1

    if has_gt:
        gt_slice = ground_truth_volume[:, :, largest_slice]
        _masked_overlay(axes[panel_index], ct_slice, gt_slice)
        axes[panel_index].set_title("Ground Truth Overlay")
        panel_index += 1

    _masked_overlay(axes[panel_index], ct_slice, pred_slice)
    axes[panel_index].set_title("Prediction Overlay")
    panel_index += 1

    if has_gt:
        error_map = _error_slice(pred_slice, gt_slice)
        error_cmap = mcolors.ListedColormap(["#00000000", "#ff7f00", "#1f77b4", "#2ca02c"])
        axes[panel_index].imshow(ct_slice.T, cmap="gray", origin="lower")
        masked_error = np.ma.masked_where(error_map == 0, error_map)
        axes[panel_index].imshow(masked_error.T, cmap=error_cmap, origin="lower", alpha=0.65, vmin=0, vmax=3)
        axes[panel_index].set_title("Error Overlay")
        axes[panel_index].axis("off")
        panel_index += 1

    axes[panel_index].imshow(pred_slice.T, cmap=_label_colormap(), origin="lower", vmin=0, vmax=5)
    axes[panel_index].set_title("Multi-class Overlay")
    axes[panel_index].axis("off")

    fig.suptitle(f"Largest hemorrhage slice (axial index {largest_slice})")
    legend = _legend_handles(labels_present)
    if legend:
        fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 5), frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    overlay_3d_path = _ensure_parent(output_dir / "prediction_overlay_3d.png")
    center_x = prediction_volume.shape[0] // 2
    center_y = prediction_volume.shape[1] // 2
    mip = np.max(ct_display, axis=AXIAL_AXIS)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axial_ct = _display_slice(ct_display, AXIAL_AXIS, largest_slice)
    axial_pred = _display_slice(prediction_volume, AXIAL_AXIS, largest_slice)
    _masked_overlay(axes[0, 0], axial_ct, axial_pred)
    axes[0, 0].set_title(f"Axial (z={largest_slice})")

    sagittal_ct = _display_slice(ct_display, SAGITTAL_AXIS, center_x)
    sagittal_pred = _display_slice(prediction_volume, SAGITTAL_AXIS, center_x)
    _masked_overlay(axes[0, 1], sagittal_ct, sagittal_pred)
    axes[0, 1].set_title(f"Sagittal (x={center_x})")

    coronal_ct = _display_slice(ct_display, CORONAL_AXIS, center_y)
    coronal_pred = _display_slice(prediction_volume, CORONAL_AXIS, center_y)
    _masked_overlay(axes[1, 0], coronal_ct, coronal_pred)
    axes[1, 0].set_title(f"Coronal (y={center_y})")

    axes[1, 1].imshow(mip.T, cmap="gray", origin="lower")
  # MIP of CT only for anatomical context
    axes[1, 1].set_title("Maximum Intensity Projection")
    axes[1, 1].axis("off")

    fig.suptitle("Orthogonal views and MIP")
    fig.tight_layout()
    fig.savefig(overlay_3d_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "prediction_overlay": overlay_path,
        "prediction_overlay_3d": overlay_3d_path,
    }
