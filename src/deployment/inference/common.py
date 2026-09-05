"""Shared helpers for deployment inference backends (reuse existing modules)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.confidence import (
    all_class_confidences,
    average_softmax_confidence,
    confidence_histogram,
    study_confidence,
)
from evaluation.volume_metrics import (
    HEMORRHAGE_LABELS,
    per_subtype_volume,
    total_hemorrhage_volume_ml,
)
from inference.overlay import render_overlay_figures
from inference.pipeline import PROJECT_ROOT, scan_output_name
from inference.report import (
    save_confidence_json,
    save_prediction_nifti,
    save_probabilities_npz,
    write_prediction_report,
    write_prediction_summary,
    write_volume_report_csv,
)

DEFAULT_INFERENCE_ROOT = PROJECT_ROOT / "reports" / "inference"


def artifact_output_dir(
    scan_name: str,
    model_id: str,
    *,
    root: Path | None = None,
    day: date | None = None,
) -> Path:
    """Return ``reports/inference/YYYY-MM-DD/scan_name/model_id``."""
    base = root or DEFAULT_INFERENCE_ROOT
    stamp = (day or date.today()).isoformat()
    path = base / stamp / scan_name / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def classes_present(prediction_mask: np.ndarray) -> list[int]:
    unique = {int(value) for value in np.unique(prediction_mask)}
    return sorted(label for label in unique if label in HEMORRHAGE_LABELS)


def largest_hemorrhage_slice(prediction_mask: np.ndarray) -> int:
    depth = prediction_mask.shape[2]
    best_slice = 0
    best_count = -1
    for slice_index in range(depth):
        count = int(np.sum(np.isin(prediction_mask[:, :, slice_index], list(HEMORRHAGE_LABELS))))
        if count > best_count:
            best_count = count
            best_slice = slice_index
    return best_slice


def compute_volumes(
    prediction_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, Any]:
    per_class = per_subtype_volume(prediction_mask, spacing)
    total = total_hemorrhage_volume_ml(prediction_mask, spacing)
    return {
        "total_volume_ml": float(total),
        "per_class_volume_ml": {int(k): float(v) for k, v in per_class.items()},
        "classes_present": classes_present(prediction_mask),
        "spacing": tuple(float(v) for v in spacing),
    }


def compute_confidence(
    prediction_mask: np.ndarray,
    probability_volume: np.ndarray,
    *,
    bins: int = 20,
) -> dict[str, Any]:
    pred_tensor = torch.from_numpy(prediction_mask.astype(np.int64))
    prob_tensor = torch.from_numpy(probability_volume.astype(np.float32))
    study_conf = float(study_confidence(prob_tensor, pred_tensor).item())
    class_conf = all_class_confidences(prob_tensor, pred_tensor)
    mean_conf = float(average_softmax_confidence(prob_tensor).item())
    counts, bin_edges = confidence_histogram(prob_tensor, bins=bins)
    return {
        "study_confidence": study_conf,
        "per_class_confidence": {int(k): float(v) for k, v in class_conf.items()},
        "mean_softmax_confidence": mean_conf,
        "confidence_histogram": {
            "counts": counts.astype(int).tolist(),
            "bin_edges": bin_edges.astype(float).tolist(),
        },
    }


def write_standard_artifacts(
    *,
    output_dir: Path,
    scan_name: str,
    ct_path: Path,
    checkpoint_path: Path,
    model_name: str,
    device: str,
    processing_time_sec: float,
    prediction_mask: np.ndarray,
    probability_volume: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, ...],
    spacing: tuple[float, float, float],
    volumes: dict[str, Any],
    confidence: dict[str, Any],
    overlay_paths: dict[str, Path],
) -> dict[str, Path]:
    """Persist the canonical artifact set used by every backend."""
    artifacts: dict[str, Path] = {}
    artifacts["prediction"] = save_prediction_nifti(
        prediction_mask,
        affine,
        output_dir / "prediction.nii.gz",
    )
    mask_alias = output_dir / "prediction_mask.nii.gz"
    shutil.copy2(artifacts["prediction"], mask_alias)
    artifacts["prediction_mask"] = mask_alias

    artifacts["probabilities"] = save_probabilities_npz(
        prediction_mask,
        probability_volume,
        spacing,
        affine,
        output_dir / "probabilities.npz",
    )
    artifacts["volume_report"] = write_volume_report_csv(
        prediction_mask,
        spacing,
        output_dir / "volume_report.csv",
    )
    artifacts["confidence"] = save_confidence_json(
        study_confidence=float(confidence["study_confidence"]),
        class_confidences=confidence["per_class_confidence"],
        mean_softmax_confidence=float(confidence["mean_softmax_confidence"]),
        confidence_histogram=confidence["confidence_histogram"],
        output_path=output_dir / "confidence.json",
    )
    artifacts.update(overlay_paths)
    if "prediction_overlay" in overlay_paths:
        overlay_alias = output_dir / "overlay.png"
        shutil.copy2(overlay_paths["prediction_overlay"], overlay_alias)
        artifacts["overlay"] = overlay_alias

    artifacts["prediction_summary"] = write_prediction_summary(
        output_path=output_dir / "prediction_summary.json",
        scan_name=scan_name,
        shape=shape,
        spacing=spacing,
        processing_time_sec=processing_time_sec,
        device=device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        classes_present=list(volumes["classes_present"]),
        total_volume_ml=float(volumes["total_volume_ml"]),
        per_class_volume_ml=volumes["per_class_volume_ml"],
        study_confidence=float(confidence["study_confidence"]),
        class_confidences=confidence["per_class_confidence"],
        mean_softmax_confidence=float(confidence["mean_softmax_confidence"]),
        largest_slice=largest_hemorrhage_slice(prediction_mask),
        ground_truth_metrics=None,
        artifact_paths=artifacts,
    )
    artifacts["prediction_report"] = write_prediction_report(
        output_path=output_dir / "prediction_report.md",
        scan_name=scan_name,
        input_path=ct_path,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
        processing_time_sec=processing_time_sec,
        shape=shape,
        spacing=spacing,
        classes_present=list(volumes["classes_present"]),
        per_class_volume_ml=volumes["per_class_volume_ml"],
        total_volume_ml=float(volumes["total_volume_ml"]),
        study_confidence=float(confidence["study_confidence"]),
        class_confidences=confidence["per_class_confidence"],
        mean_softmax_confidence=float(confidence["mean_softmax_confidence"]),
        largest_slice=largest_hemorrhage_slice(prediction_mask),
        ground_truth_metrics=None,
        artifact_paths=artifacts,
    )
    return artifacts


def render_overlays(
    *,
    raw_hu: np.ndarray,
    prediction_mask: np.ndarray,
    spacing: tuple[float, float, float],
    output_dir: Path,
    ground_truth: np.ndarray | None = None,
    largest_slice: int | None = None,
    preprocessing_config_path: Path | None = None,
) -> dict[str, Path]:
    """Delegate overlay rendering to ``inference.overlay``."""
    slice_index = largest_slice if largest_slice is not None else largest_hemorrhage_slice(prediction_mask)
    return render_overlay_figures(
        raw_hu=raw_hu,
        prediction_volume=prediction_mask,
        ground_truth_volume=ground_truth,
        spacing=spacing,
        largest_slice=slice_index,
        preprocessing_config_path=preprocessing_config_path
        or (PROJECT_ROOT / "config" / "preprocessing.yaml"),
        output_dir=output_dir,
    )


__all__ = [
    "DEFAULT_INFERENCE_ROOT",
    "PROJECT_ROOT",
    "artifact_output_dir",
    "classes_present",
    "compute_confidence",
    "compute_volumes",
    "largest_hemorrhage_slice",
    "render_overlays",
    "scan_output_name",
    "write_standard_artifacts",
]
