"""Markdown, CSV, and JSON report writers for inference outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np

from evaluation.volume_metrics import (
    HEMORRHAGE_LABELS,
    calculate_volume_ml,
    format_subtype_name,
    total_hemorrhage_volume_ml,
    voxel_volume_mm3,
)


def save_prediction_nifti(
    prediction_volume: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
) -> Path:
    """Save integer label prediction as NIfTI with preserved affine."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nifti = nib.Nifti1Image(prediction_volume.astype(np.int16), affine)
    nib.save(nifti, str(output_path))
    return output_path


def save_probabilities_npz(
    prediction_volume: np.ndarray,
    probability_volume: np.ndarray,
    spacing: tuple[float, float, float],
    affine: np.ndarray,
    output_path: Path,
) -> Path:
    """Save stacked softmax probabilities and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prediction=prediction_volume.astype(np.int16),
        probabilities=probability_volume.astype(np.float32),
        spacing=np.array(spacing, dtype=np.float64),
        affine=affine.astype(np.float64),
    )
    return output_path


def write_volume_report_csv(
    prediction_volume: np.ndarray,
    spacing: tuple[float, float, float],
    output_path: Path,
) -> Path:
    """Write per-subtype native-space volume table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_volume = total_hemorrhage_volume_ml(prediction_volume, spacing)
    voxel_mm3 = voxel_volume_mm3(spacing)

    rows: list[dict[str, Any]] = []
    for label in HEMORRHAGE_LABELS:
        voxel_count = int(np.sum(prediction_volume == label))
        volume_ml = calculate_volume_ml(prediction_volume, spacing, label)
        percent = (volume_ml / total_volume * 100.0) if total_volume > 0 else 0.0
        rows.append(
            {
                "subtype": format_subtype_name(label),
                "label": label,
                "voxel_count": voxel_count,
                "volume_ml": round(volume_ml, 6),
                "percent_of_total_hemorrhage": round(percent, 4),
            }
        )

    rows.append(
        {
            "subtype": "TOTAL",
            "label": "",
            "voxel_count": int(sum(row["voxel_count"] for row in rows)),
            "volume_ml": round(total_volume, 6),
            "percent_of_total_hemorrhage": 100.0 if total_volume > 0 else 0.0,
        }
    )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subtype", "label", "voxel_count", "volume_ml", "percent_of_total_hemorrhage"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_confidence_json(
    study_confidence: float,
    class_confidences: Mapping[int, float],
    mean_softmax_confidence: float,
    confidence_histogram: Mapping[str, list[float] | list[int]],
    output_path: Path,
) -> Path:
    """Write confidence statistics for frontend consumption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_confidence": study_confidence,
        "per_class_confidence": {str(label): float(value) for label, value in class_confidences.items()},
        "mean_softmax_confidence": mean_softmax_confidence,
        "confidence_histogram": {
            "counts": list(confidence_histogram["counts"]),
            "bin_edges": list(confidence_histogram["bin_edges"]),
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def write_prediction_summary(
    output_path: Path,
    scan_name: str,
    shape: tuple[int, ...],
    spacing: tuple[float, float, float],
    processing_time_sec: float,
    device: str,
    model_name: str,
    checkpoint_path: Path,
    classes_present: list[int],
    total_volume_ml: float,
    per_class_volume_ml: Mapping[int, float],
    study_confidence: float,
    class_confidences: Mapping[int, float],
    mean_softmax_confidence: float,
    largest_slice: int,
    ground_truth_metrics: Any | None,
    artifact_paths: Mapping[str, Path],
) -> Path:
    """Write machine-readable prediction summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "scan_name": scan_name,
        "shape": list(shape),
        "spacing": list(spacing),
        "processing_time_sec": round(processing_time_sec, 4),
        "device": device,
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "classes_present": classes_present,
        "total_volume_ml": round(total_volume_ml, 6),
        "per_class_volume_ml": {str(label): round(float(value), 6) for label, value in per_class_volume_ml.items()},
        "confidence": {
            "study_confidence": study_confidence,
            "per_class_confidence": {str(label): float(value) for label, value in class_confidences.items()},
            "mean_softmax_confidence": mean_softmax_confidence,
        },
        "largest_slice": largest_slice,
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    if ground_truth_metrics is not None:
        payload["ground_truth_metrics"] = {
            "macro_dice": ground_truth_metrics.macro_dice,
            "micro_dice": ground_truth_metrics.micro_dice,
            "total_volume_error_ml": ground_truth_metrics.total_volume_error_ml,
            "per_class": {
                str(class_index): metrics
                for class_index, metrics in ground_truth_metrics.per_class.items()
            },
            "confusion_matrix": ground_truth_metrics.confusion_matrix,
            "volume_errors": {
                str(label): values for label, values in ground_truth_metrics.volume_errors.items()
            },
        }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def write_prediction_report(
    output_path: Path,
    scan_name: str,
    input_path: Path,
    checkpoint_path: Path,
    model_name: str,
    device: str,
    processing_time_sec: float,
    shape: tuple[int, ...],
    spacing: tuple[float, float, float],
    classes_present: list[int],
    per_class_volume_ml: Mapping[int, float],
    total_volume_ml: float,
    study_confidence: float,
    class_confidences: Mapping[int, float],
    mean_softmax_confidence: float,
    largest_slice: int,
    ground_truth_metrics: Any | None,
    artifact_paths: Mapping[str, Path],
) -> Path:
    """Write human-readable markdown prediction report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtype_names = [format_subtype_name(label) for label in classes_present] or ["None"]

    lines = [
        "# Prediction Report",
        "",
        "## Scan Summary",
        f"- Scan name: `{scan_name}`",
        f"- Input path: `{input_path}`",
        f"- Shape: `{shape}`",
        f"- Spacing (mm): `{spacing}`",
        f"- Largest hemorrhage axial slice: `{largest_slice}`",
        "",
        "## Model",
        f"- Model: `{model_name}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Device: `{device}`",
        f"- Runtime: `{processing_time_sec:.2f}` seconds",
        "",
        "## Detected Hemorrhage Subtypes",
        f"- Classes present: {', '.join(subtype_names)}",
        f"- Total hemorrhage volume: **{total_volume_ml:.4f} mL**",
        "",
        "## Volume Table",
        "| Subtype | Volume (mL) |",
        "|---------|------------:|",
    ]
    for label in sorted(per_class_volume_ml.keys()):
        lines.append(f"| {format_subtype_name(label)} | {per_class_volume_ml[label]:.4f} |")
    lines.append(f"| **Total** | **{total_volume_ml:.4f}** |")

    lines.extend(
        [
            "",
            "## Confidence Table",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Study confidence | {study_confidence:.6f} |",
            f"| Mean softmax confidence | {mean_softmax_confidence:.6f} |",
        ]
    )
    for label, value in sorted(class_confidences.items()):
        lines.append(f"| {format_subtype_name(label)} confidence | {value:.6f} |")

    lines.extend(
        [
            "",
            "## Generated Artifacts",
        ]
    )
    for name, path in sorted(artifact_paths.items()):
        lines.append(f"- `{name}`: `{path}`")

    if ground_truth_metrics is not None:
        lines.extend(
            [
                "",
                "## Ground Truth Comparison",
                f"- Macro Dice: **{ground_truth_metrics.macro_dice:.6f}**",
                f"- Micro Dice: {ground_truth_metrics.micro_dice:.6f}",
                f"- Total volume error: {ground_truth_metrics.total_volume_error_ml:.4f} mL",
                "",
                "### Per-Class Metrics",
                "| Class | Dice | IoU | Precision | Recall | Sensitivity | Specificity |",
                "|------:|-----:|----:|----------:|-------:|------------:|------------:|",
            ]
        )
        for class_index in sorted(ground_truth_metrics.per_class.keys()):
            metrics = ground_truth_metrics.per_class[class_index]
            subtype = "BG" if class_index == 0 else format_subtype_name(class_index)
            lines.append(
                f"| {class_index} ({subtype}) | {metrics['dice']:.4f} | {metrics['iou']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
