"""Production inference pipeline for single-scan CT hemorrhage segmentation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from dataset.slice_extraction import volume_to_slice_samples
from evaluation.checkpoint import load_checkpoint
from evaluation.confidence import (
    all_class_confidences,
    average_softmax_confidence,
    confidence_histogram,
    study_confidence,
)
from evaluation.evaluate_model import ScanBuffer, stack_scan_predictions
from evaluation.metrics import HEMORRHAGE_CLASSES, NUM_CLASSES, compute_segmentation_metrics
from evaluation.plots import plot_confusion_matrix, plot_prediction_example
from evaluation.volume_metrics import (
    HEMORRHAGE_LABELS,
    per_subtype_volume,
    per_subtype_volume_error,
    total_hemorrhage_volume_ml,
)
from .overlay import render_overlay_figures
from .report import (
    save_confidence_json,
    save_prediction_nifti,
    save_probabilities_npz,
    write_prediction_report,
    write_prediction_summary,
    write_volume_report_csv,
)
from models.monai_unet import build_model
from preprocessing.preprocess_volume import (
    PreprocessedVolume,
    apply_intensity_clip,
    apply_normalization,
    extract_native_metadata,
    load_preprocessing_config,
    prepare_mask,
    preprocess_volume,
    resolve_clip_bounds,
    validate_preprocessed,
)
from training.config_loader import load_model_config, resolve_device, should_use_amp

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

PROJECT_ROOT = _PROJECT_ROOT
DEFAULT_PREPROCESSING_CONFIG = PROJECT_ROOT / "config" / "preprocessing.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config" / "model.yaml"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "inference"
CLASS_LABELS = ["BG", "EDH", "SDH", "SAH", "IPH", "IVH"]
BHSD_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "images"
BHSD_MASKS_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "ground truths"


@dataclass(frozen=True)
class GroundTruthMetrics:
    """Optional segmentation metrics when a reference mask is available."""

    macro_dice: float
    micro_dice: float
    per_class: dict[int, dict[str, float]]
    confusion_matrix: list[list[int]]
    total_volume_error_ml: float
    volume_errors: dict[int, dict[str, float]]


@dataclass
class InferenceResult:
    """Structured inference output for one scan."""

    scan_name: str
    output_dir: Path
    prediction_volume: np.ndarray
    probability_volume: np.ndarray
    spacing: tuple[float, float, float]
    shape: tuple[int, ...]
    affine: np.ndarray
    classes_present: list[int]
    per_class_volume_ml: dict[int, float]
    total_volume_ml: float
    study_confidence: float
    class_confidences: dict[int, float]
    mean_softmax_confidence: float
    confidence_histogram: dict[str, list[float]]
    largest_slice: int
    processing_time_sec: float
    device: str
    model_name: str
    checkpoint_path: Path
    ground_truth_metrics: GroundTruthMetrics | None = None
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def scan_output_name(input_path: Path) -> str:
    """Return directory-safe scan name from a NIfTI path."""
    name = input_path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return input_path.stem


def _is_bhsd_dataset_scan(input_path: Path) -> bool:
    """Return whether the input lives in the locked BHSD image directory."""
    try:
        input_path.resolve().relative_to(BHSD_IMAGES_DIR.resolve())
        return True
    except ValueError:
        return False


def preprocess_nifti_scan(
    input_path: Path | str,
    mask_path: Path | str | None = None,
    preprocessing_config_path: Path | str | None = None,
) -> tuple[PreprocessedVolume, np.ndarray]:
    """Load and preprocess one CT volume using the locked preprocessing pipeline.

    Returns the preprocessed volume and the raw HU array for visualization.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CT not found: {input_path}")

    if mask_path is None and _is_bhsd_dataset_scan(input_path):
        volume = preprocess_volume(input_path.name, config_path=Path(preprocessing_config_path) if preprocessing_config_path else None)
        raw_hu = nib.load(str(input_path)).get_fdata().astype(np.float32)
        return volume, raw_hu

    config = load_preprocessing_config(Path(preprocessing_config_path) if preprocessing_config_path else None)
    image_nii = nib.load(str(input_path))
    image_hu = image_nii.get_fdata().astype(np.float32)
    original_spacing, original_shape, affine = extract_native_metadata(image_nii)

    if mask_path is not None:
        mask_nii = nib.load(str(mask_path))
        mask_raw = mask_nii.get_fdata()
        if tuple(int(value) for value in mask_raw.shape[:3]) != original_shape:
            raise ValueError(
                f"Mask shape {mask_raw.shape[:3]} does not match image shape {original_shape}."
            )
        processed_mask = prepare_mask(mask_raw)
    else:
        processed_mask = np.zeros(original_shape, dtype=np.int64)

    clip_min, clip_max = resolve_clip_bounds(config)
    clipped = apply_intensity_clip(image_hu, clip_min, clip_max)
    normalized = apply_normalization(clipped, config)
    validate_preprocessed(normalized, processed_mask, original_shape, original_spacing)

    volume = PreprocessedVolume(
        processed_image=normalized,
        processed_mask=processed_mask,
        original_spacing=original_spacing,
        original_shape=original_shape,
        affine=affine,
        filename=input_path.name,
    )
    return volume, image_hu


def _find_largest_hemorrhage_slice(prediction_volume: np.ndarray) -> int:
    """Return axial slice index with the largest predicted hemorrhage footprint."""
    depth = prediction_volume.shape[2]
    best_slice = 0
    best_count = -1
    for slice_index in range(depth):
        slice_labels = prediction_volume[:, :, slice_index]
        count = int(np.sum(np.isin(slice_labels, list(HEMORRHAGE_LABELS))))
        if count > best_count:
            best_count = count
            best_slice = slice_index
    return best_slice


def _classes_present(prediction_volume: np.ndarray) -> list[int]:
    """Return sorted hemorrhage class indices present in the prediction."""
    unique = set(int(value) for value in np.unique(prediction_volume))
    return sorted(label for label in unique if label in HEMORRHAGE_LABELS)


class InferencePipeline:
    """Run trained MONAI U-Net inference on labelled or unlabelled CT scans."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        model_config_path: Path | str | None = None,
        preprocessing_config_path: Path | str | None = None,
        device: str | None = "auto",
        batch_size: int = 4,
        reports_dir: Path | str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = PROJECT_ROOT / self.checkpoint_path
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.preprocessing_config_path = Path(preprocessing_config_path or DEFAULT_PREPROCESSING_CONFIG)
        self.model_config_path = Path(model_config_path or DEFAULT_MODEL_CONFIG)
        self.reports_dir = Path(reports_dir or DEFAULT_REPORTS_DIR)
        self.batch_size = max(1, int(batch_size))

        self.device = resolve_device(device or "auto")
        self.checkpoint_payload = load_checkpoint(self.checkpoint_path, map_location=self.device)
        metadata = self.checkpoint_payload.get("metadata", {})
        self.model_config = metadata.get("model_config") or load_model_config(self.model_config_path)
        training_config = metadata.get("training_config") or {}
        self.use_amp = should_use_amp(training_config, self.device)

        self.model = build_model(self.model_config)
        self.model.load_state_dict(self.checkpoint_payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _run_model(self, volume: PreprocessedVolume) -> ScanBuffer:
        """Run slice-wise inference and reconstruct a scan buffer."""
        samples = volume_to_slice_samples(volume)
        buffer = ScanBuffer(
            filename=volume.filename,
            depth=volume.processed_image.shape[2],
            original_shape=volume.original_shape,
            original_spacing=volume.original_spacing,
            affine=volume.affine,
        )

        with torch.no_grad():
            for start in range(0, len(samples), self.batch_size):
                batch_samples = samples[start : start + self.batch_size]
                images = torch.from_numpy(
                    np.stack([sample.image for sample in batch_samples], axis=0)
                ).to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)

                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)

                for index, sample in enumerate(batch_samples):
                    buffer.add_slice(
                        slice_index=sample.slice_index,
                        prediction=predictions[index].cpu().numpy().astype(np.int64),
                        ground_truth=sample.label.astype(np.int64),
                        image=sample.image,
                        probabilities=probabilities[index].cpu().numpy(),
                    )

        if not buffer.is_complete():
            raise RuntimeError(
                f"Incomplete slice collection for {volume.filename}: "
                f"{len(buffer.pred_slices)}/{buffer.depth} slices."
            )
        return buffer

    def _compute_confidence(
        self,
        prediction_volume: np.ndarray,
        probability_volume: np.ndarray,
        bins: int = 20,
    ) -> tuple[float, dict[int, float], float, dict[str, list[float]]]:
        """Compute study, per-class, and histogram confidence statistics."""
        pred_tensor = torch.from_numpy(prediction_volume)
        prob_tensor = torch.from_numpy(probability_volume)
        study_conf = float(study_confidence(prob_tensor, pred_tensor).item())
        class_conf = all_class_confidences(prob_tensor, pred_tensor)
        mean_conf = float(average_softmax_confidence(prob_tensor).item())
        counts, bin_edges = confidence_histogram(prob_tensor, bins=bins)
        histogram = {
            "counts": counts.astype(int).tolist(),
            "bin_edges": bin_edges.astype(float).tolist(),
        }
        return study_conf, class_conf, mean_conf, histogram

    def _save_failure_visualizations(
        self,
        buffer: ScanBuffer,
        largest_slice: int,
        ground_truth_metrics: GroundTruthMetrics,
        output_dir: Path,
    ) -> None:
        """Save optional failure-analysis figures when a reference mask is provided."""
        plot_prediction_example(
            buffer.image_slices[largest_slice],
            buffer.gt_slices[largest_slice],
            buffer.pred_slices[largest_slice],
            output_path=output_dir / "failure_prediction_example.png",
            title="Failure Visualization — Representative Slice",
        )
        plot_confusion_matrix(
            ground_truth_metrics.confusion_matrix,
            CLASS_LABELS,
            output_path=output_dir / "failure_confusion_matrix.png",
            title="Failure Visualization — Confusion Matrix",
        )

    def _compute_ground_truth_metrics(
        self,
        prediction_volume: np.ndarray,
        ground_truth_volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> GroundTruthMetrics:
        """Compute optional reference-mask metrics using evaluation modules."""
        pred_tensor = torch.from_numpy(prediction_volume.astype(np.int64))
        target_tensor = torch.from_numpy(ground_truth_volume.astype(np.int64))
        metrics = compute_segmentation_metrics(pred_tensor, target_tensor, ignore_background=True)

        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        pred_flat = prediction_volume.reshape(-1)
        target_flat = ground_truth_volume.reshape(-1)
        for gt_label, pred_label in zip(target_flat, pred_flat, strict=True):
            confusion[int(gt_label), int(pred_label)] += 1

        volume_errors = per_subtype_volume_error(prediction_volume, ground_truth_volume, spacing)
        pred_total = total_hemorrhage_volume_ml(prediction_volume, spacing)
        gt_total = total_hemorrhage_volume_ml(ground_truth_volume, spacing)

        per_class: dict[int, dict[str, float]] = {}
        for class_index in range(NUM_CLASSES):
            per_class[class_index] = {
                "dice": float(metrics.dice[class_index].item()),
                "iou": float(metrics.iou[class_index].item()),
                "precision": float(metrics.precision[class_index].item()),
                "recall": float(metrics.recall[class_index].item()),
                "sensitivity": float(metrics.sensitivity[class_index].item()),
                "specificity": float(metrics.specificity[class_index].item()),
            }

        return GroundTruthMetrics(
            macro_dice=float(metrics.macro_dice.item()),
            micro_dice=float(metrics.micro_dice.item()),
            per_class=per_class,
            confusion_matrix=confusion.tolist(),
            total_volume_error_ml=abs(pred_total - gt_total),
            volume_errors=volume_errors,
        )

    def predict(
        self,
        input_path: Path | str,
        mask_path: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> InferenceResult:
        """Run inference on one CT scan and write production artifacts."""
        start_time = time.perf_counter()
        input_path = Path(input_path)
        scan_name = scan_output_name(input_path)
        output_dir = Path(output_dir) if output_dir else self.reports_dir / scan_name
        output_dir.mkdir(parents=True, exist_ok=True)

        volume, raw_hu = preprocess_nifti_scan(
            input_path=input_path,
            mask_path=mask_path,
            preprocessing_config_path=self.preprocessing_config_path,
        )
        buffer = self._run_model(volume)
        prediction_volume, ground_truth_volume = stack_scan_predictions(buffer)
        probability_volume = buffer.stack_volume(buffer.prob_slices, channel_first=True)

        spacing = volume.original_spacing
        classes_present = _classes_present(prediction_volume)
        per_class_volume = per_subtype_volume(prediction_volume, spacing)
        total_volume = total_hemorrhage_volume_ml(prediction_volume, spacing)
        largest_slice = _find_largest_hemorrhage_slice(prediction_volume)
        study_conf, class_conf, mean_conf, histogram = self._compute_confidence(
            prediction_volume,
            probability_volume,
        )

        has_ground_truth = mask_path is not None
        ground_truth_metrics = None
        if has_ground_truth:
            ground_truth_metrics = self._compute_ground_truth_metrics(
                prediction_volume,
                ground_truth_volume,
                spacing,
            )
            self._save_failure_visualizations(
                buffer=buffer,
                largest_slice=largest_slice,
                ground_truth_metrics=ground_truth_metrics,
                output_dir=output_dir,
            )

        processing_time = time.perf_counter() - start_time
        model_name = str(self.model_config.get("model_name", "monai_2d_unet"))

        artifact_paths: dict[str, Path] = {}
        artifact_paths["prediction_nifti"] = save_prediction_nifti(
            prediction_volume,
            volume.affine,
            output_dir / "prediction.nii.gz",
        )
        artifact_paths["probabilities"] = save_probabilities_npz(
            prediction_volume,
            probability_volume,
            spacing,
            volume.affine,
            output_dir / "probabilities.npz",
        )
        artifact_paths["volume_report"] = write_volume_report_csv(
            prediction_volume,
            spacing,
            output_dir / "volume_report.csv",
        )
        artifact_paths["confidence"] = save_confidence_json(
            study_confidence=study_conf,
            class_confidences=class_conf,
            mean_softmax_confidence=mean_conf,
            confidence_histogram=histogram,
            output_path=output_dir / "confidence.json",
        )

        overlay_paths = render_overlay_figures(
            raw_hu=raw_hu,
            prediction_volume=prediction_volume,
            ground_truth_volume=ground_truth_volume if has_ground_truth else None,
            spacing=spacing,
            largest_slice=largest_slice,
            preprocessing_config_path=self.preprocessing_config_path,
            output_dir=output_dir,
        )
        artifact_paths.update(overlay_paths)

        summary_path = write_prediction_summary(
            output_path=output_dir / "prediction_summary.json",
            scan_name=scan_name,
            shape=volume.original_shape,
            spacing=spacing,
            processing_time_sec=processing_time,
            device=str(self.device),
            model_name=model_name,
            checkpoint_path=self.checkpoint_path,
            classes_present=classes_present,
            total_volume_ml=total_volume,
            per_class_volume_ml=per_class_volume,
            study_confidence=study_conf,
            class_confidences=class_conf,
            mean_softmax_confidence=mean_conf,
            largest_slice=largest_slice,
            ground_truth_metrics=ground_truth_metrics,
            artifact_paths=artifact_paths,
        )
        artifact_paths["prediction_summary"] = summary_path

        report_path = write_prediction_report(
            output_path=output_dir / "prediction_report.md",
            scan_name=scan_name,
            input_path=input_path,
            checkpoint_path=self.checkpoint_path,
            model_name=model_name,
            device=str(self.device),
            processing_time_sec=processing_time,
            shape=volume.original_shape,
            spacing=spacing,
            classes_present=classes_present,
            per_class_volume_ml=per_class_volume,
            total_volume_ml=total_volume,
            study_confidence=study_conf,
            class_confidences=class_conf,
            mean_softmax_confidence=mean_conf,
            largest_slice=largest_slice,
            ground_truth_metrics=ground_truth_metrics,
            artifact_paths=artifact_paths,
        )
        artifact_paths["prediction_report"] = report_path

        logger.info(
            "Inference complete for %s — classes=%s, total_volume=%.3f mL, output=%s",
            scan_name,
            classes_present,
            total_volume,
            output_dir,
        )

        return InferenceResult(
            scan_name=scan_name,
            output_dir=output_dir,
            prediction_volume=prediction_volume,
            probability_volume=probability_volume,
            spacing=spacing,
            shape=volume.original_shape,
            affine=volume.affine,
            classes_present=classes_present,
            per_class_volume_ml=per_class_volume,
            total_volume_ml=total_volume,
            study_confidence=study_conf,
            class_confidences=class_conf,
            mean_softmax_confidence=mean_conf,
            confidence_histogram=histogram,
            largest_slice=largest_slice,
            processing_time_sec=processing_time,
            device=str(self.device),
            model_name=model_name,
            checkpoint_path=self.checkpoint_path,
            ground_truth_metrics=ground_truth_metrics,
            artifact_paths=artifact_paths,
        )
