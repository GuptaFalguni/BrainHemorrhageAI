"""Post-training baseline model evaluation on the locked test split."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.amp import autocast

from dataset.bhsd_dataset import load_split_filenames
from dataset.dataloader import build_dataloader, load_training_config
from evaluation.checkpoint import CHECKPOINT_SELECTION_METRIC, load_checkpoint
from evaluation.confidence import (
    all_class_confidences,
    study_confidence,
)
from evaluation.experiment_logger import collect_environment_info, compute_split_hash
from evaluation.metrics import (
    HEMORRHAGE_CLASSES,
    NUM_CLASSES,
    binary_confusion,
    compute_segmentation_metrics,
)
from evaluation.plots import (
    plot_confusion_matrix,
    plot_confidence_histogram,
    plot_per_class_dice,
    plot_prediction_example,
    plot_scan_examples,
    plot_volume_error,
    plot_volume_error_distribution,
)
from evaluation.volume_metrics import (
    SUBTYPE_LABELS,
    absolute_volume_error_ml,
    format_subtype_name,
    per_subtype_volume_error,
    total_hemorrhage_volume_ml,
)
from models.monai_unet import build_model
from training.config_loader import load_model_config, load_yaml_config, resolve_device, should_use_amp
from training.seed import set_seed

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

PROJECT_ROOT = _PROJECT_ROOT
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "config" / "evaluation.yaml"
CLASS_LABELS = ["BG", "EDH", "SDH", "SAH", "IPH", "IVH"]


@dataclass
class MetricAccumulator:
    """Aggregate global confusion counts and confidence histogram."""

    num_classes: int = NUM_CLASSES
    tp: torch.Tensor = field(default_factory=lambda: torch.zeros(NUM_CLASSES, dtype=torch.float64))
    fp: torch.Tensor = field(default_factory=lambda: torch.zeros(NUM_CLASSES, dtype=torch.float64))
    fn: torch.Tensor = field(default_factory=lambda: torch.zeros(NUM_CLASSES, dtype=torch.float64))
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))
    confidence_counts: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=np.int64))
    confidence_bin_edges: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.0, 21))

    def update_batch(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        probabilities: torch.Tensor,
        confidence_bins: int,
    ) -> None:
        """Update accumulators from one batch."""
        for class_index in range(self.num_classes):
            confusion = binary_confusion(
                prediction == class_index,
                target == class_index,
            )
            self.tp[class_index] += confusion.tp.cpu()
            self.fp[class_index] += confusion.fp.cpu()
            self.fn[class_index] += confusion.fn.cpu()

        pred_flat = prediction.reshape(-1).cpu().numpy()
        target_flat = target.reshape(-1).cpu().numpy()
        for gt_label, pred_label in zip(target_flat, pred_flat, strict=True):
            self.confusion_matrix[int(gt_label), int(pred_label)] += 1

        confidence = probabilities.max(dim=1).values.detach().cpu().numpy().ravel()
        counts, bin_edges = np.histogram(confidence, bins=confidence_bins, range=(0.0, 1.0))
        if self.confidence_counts.shape[0] != counts.shape[0]:
            self.confidence_counts = np.zeros_like(counts)
            self.confidence_bin_edges = bin_edges
        self.confidence_counts += counts.astype(np.int64)

    def per_class_dice(self, smooth: float = 1e-7) -> torch.Tensor:
        """Compute per-class Dice from accumulated counts."""
        numerator = 2.0 * self.tp
        denominator = 2.0 * self.tp + self.fp + self.fn + smooth
        return (numerator / denominator).to(dtype=torch.float32)

    def per_class_iou(self, smooth: float = 1e-7) -> torch.Tensor:
        """Compute per-class IoU from accumulated counts."""
        union = self.tp + self.fp + self.fn + smooth
        return (self.tp / union).to(dtype=torch.float32)

    def per_class_precision(self, smooth: float = 1e-7) -> torch.Tensor:
        """Compute per-class precision from accumulated counts."""
        return (self.tp / (self.tp + self.fp + smooth)).to(dtype=torch.float32)

    def per_class_recall(self, smooth: float = 1e-7) -> torch.Tensor:
        """Compute per-class recall from accumulated counts."""
        return (self.tp / (self.tp + self.fn + smooth)).to(dtype=torch.float32)

    def per_class_specificity(self, smooth: float = 1e-7) -> torch.Tensor:
        """Compute per-class specificity from accumulated counts."""
        total_voxels = float(self.confusion_matrix.sum())
        specificity_values: list[torch.Tensor] = []
        for class_index in range(self.num_classes):
            tp = float(self.tp[class_index])
            fp = float(self.fp[class_index])
            fn = float(self.fn[class_index])
            tn = total_voxels - tp - fp - fn
            specificity_values.append(
                torch.tensor(tn / (tn + fp + smooth), dtype=torch.float32)
            )
        return torch.stack(specificity_values)

    def macro_dice(self) -> float:
        """Compute macro Dice over hemorrhage classes."""
        dice = self.per_class_dice()
        return float(dice[list(HEMORRHAGE_CLASSES)].mean().item())


@dataclass
class ScanBuffer:
    """Collect slice-wise predictions until a full scan is available."""

    filename: str
    depth: int
    original_shape: tuple[int, ...]
    original_spacing: tuple[float, float, float]
    affine: np.ndarray
    pred_slices: dict[int, np.ndarray] = field(default_factory=dict)
    gt_slices: dict[int, np.ndarray] = field(default_factory=dict)
    image_slices: dict[int, np.ndarray] = field(default_factory=dict)
    prob_slices: dict[int, np.ndarray] = field(default_factory=dict)

    def add_slice(
        self,
        slice_index: int,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        image: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Store one slice prediction."""
        self.pred_slices[int(slice_index)] = prediction
        self.gt_slices[int(slice_index)] = ground_truth
        self.image_slices[int(slice_index)] = image
        self.prob_slices[int(slice_index)] = probabilities

    def is_complete(self) -> bool:
        """Return whether all axial slices have been collected."""
        return len(self.pred_slices) == self.depth

    def stack_volume(self, slices: dict[int, np.ndarray], channel_first: bool = False) -> np.ndarray:
        """Stack 2D slices into a 3D volume along the axial axis."""
        sample = next(iter(slices.values()))
        if channel_first:
            num_classes = sample.shape[0]
            volume = np.zeros((num_classes, *self.original_shape), dtype=sample.dtype)
            for slice_index in range(self.depth):
                volume[:, :, :, slice_index] = slices[slice_index]
            return volume

        volume = np.zeros(self.original_shape, dtype=sample.dtype)
        for slice_index in range(self.depth):
            volume[:, :, slice_index] = slices[slice_index]
        return volume


@dataclass
class ScanEvaluationResult:
    """Evaluation metrics for one scan."""

    filename: str
    macro_dice: float
    per_class_dice: dict[int, float]
    study_confidence: float
    class_confidences: dict[int, float]
    total_volume_error_ml: float
    volume_errors: dict[int, dict[str, float]]
    representative_slice: int
    image_slice: np.ndarray
    gt_slice: np.ndarray
    pred_slice: np.ndarray


def _extract_spacing(batch: dict[str, Any], sample_index: int) -> tuple[float, float, float]:
    """Extract native spacing for one sample from a collated batch."""
    components = batch["original_spacing"]
    return tuple(float(components[axis][sample_index].item()) for axis in range(3))


def _extract_shape(batch: dict[str, Any], sample_index: int) -> tuple[int, ...]:
    """Extract native volume shape for one sample from a collated batch."""
    components = batch["original_shape"]
    return tuple(int(components[axis][sample_index].item()) for axis in range(3))


def _extract_filename(filenames: list[str] | tuple[str, ...], sample_index: int) -> str:
    """Return filename string for one batch item."""
    filename = filenames[sample_index]
    if isinstance(filename, bytes):
        return filename.decode("utf-8")
    return str(filename)


def load_evaluation_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load evaluation YAML configuration."""
    return load_yaml_config(path or DEFAULT_EVAL_CONFIG)


def generate_run_id(prefix: str = "eval") -> str:
    """Create a timestamp-based evaluation run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def _write_training_config_override(
    eval_config: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write a temporary dataloader config merged from training settings."""
    training_config = load_training_config(PROJECT_ROOT / eval_config["training_config"])
    training_config["batch_size"] = int(eval_config.get("batch_size", training_config.get("batch_size", 4)))
    training_config["num_workers"] = int(eval_config.get("num_workers", training_config.get("num_workers", 0)))
    training_config["pin_memory"] = bool(eval_config.get("pin_memory", training_config.get("pin_memory", False)))
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(training_config, handle, sort_keys=False)
    return output_path


def stack_scan_predictions(buffer: ScanBuffer) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked prediction and ground-truth volumes."""
    prediction = buffer.stack_volume(buffer.pred_slices)
    ground_truth = buffer.stack_volume(buffer.gt_slices)
    return prediction, ground_truth


def _evaluate_scan(buffer: ScanBuffer) -> ScanEvaluationResult:
    """Compute scan-level metrics after all slices are collected."""
    prediction_volume, ground_truth_volume = stack_scan_predictions(buffer)
    spacing = buffer.original_spacing

    pred_tensor = torch.from_numpy(prediction_volume)
    target_tensor = torch.from_numpy(ground_truth_volume)
    scan_metrics = compute_segmentation_metrics(pred_tensor, target_tensor, ignore_background=True)

    prob_volume = buffer.stack_volume(buffer.prob_slices, channel_first=True)
    pred_tensor_volume = torch.from_numpy(prediction_volume)
    probs_tensor = torch.from_numpy(prob_volume)
    study_conf = float(study_confidence(probs_tensor, pred_tensor_volume).item())
    class_conf = all_class_confidences(probs_tensor, pred_tensor_volume)

    volume_errors = per_subtype_volume_error(prediction_volume, ground_truth_volume, spacing)
    pred_total = total_hemorrhage_volume_ml(prediction_volume, spacing)
    gt_total = total_hemorrhage_volume_ml(ground_truth_volume, spacing)
    total_error = absolute_volume_error_ml(pred_total, gt_total)

    representative_slice = buffer.depth // 2
    return ScanEvaluationResult(
        filename=buffer.filename,
        macro_dice=float(scan_metrics.macro_dice.item()),
        per_class_dice={
            class_index: float(scan_metrics.dice[class_index].item())
            for class_index in range(NUM_CLASSES)
        },
        study_confidence=study_conf,
        class_confidences=class_conf,
        total_volume_error_ml=total_error,
        volume_errors=volume_errors,
        representative_slice=representative_slice,
        image_slice=buffer.image_slices[representative_slice],
        gt_slice=buffer.gt_slices[representative_slice],
        pred_slice=buffer.pred_slices[representative_slice],
    )


class ModelEvaluator:
    """Run baseline model evaluation on the locked test split."""

    def __init__(
        self,
        eval_config: dict[str, Any],
        checkpoint_path: Path | str | None = None,
        run_dir: Path | str | None = None,
    ) -> None:
        self.eval_config = eval_config
        self.checkpoint_path = Path(checkpoint_path or eval_config["checkpoint_path"])
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = PROJECT_ROOT / self.checkpoint_path

        run_id = eval_config.get("run_id") or generate_run_id()
        reports_dir = PROJECT_ROOT / str(eval_config.get("reports_dir", "reports/evaluation"))
        self.run_dir = Path(run_dir) if run_dir else reports_dir / str(run_id)
        self.plots_dir = self.run_dir / "plots"
        self.predictions_dir = self.run_dir / "predictions"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        training_config = load_training_config(PROJECT_ROOT / eval_config["training_config"])
        self.seed = int(training_config.get("seed", 42))
        set_seed(self.seed)

        self.device = resolve_device(str(eval_config.get("device", "auto")))
        self.use_amp = should_use_amp(eval_config, self.device)
        self.confidence_bins = int(eval_config.get("confidence_bins", 20))
        self.top_k = int(eval_config.get("top_k", 10))

        self.checkpoint_payload = load_checkpoint(self.checkpoint_path, map_location=self.device)
        metadata = self.checkpoint_payload.get("metadata", {})
        model_config = metadata.get("model_config") or load_model_config(
            PROJECT_ROOT / eval_config["model_config"]
        )
        training_config_snapshot = metadata.get("training_config") or training_config

        self.model = build_model(model_config)
        self.model.load_state_dict(self.checkpoint_payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.model_config = model_config
        self.training_config = training_config_snapshot
        self.environment = collect_environment_info(seed=self.seed)

    def _build_dataloader(self, temp_config_path: Path):
        """Build the test split dataloader."""
        split = str(self.eval_config.get("split", "test"))
        if split != "test":
            raise ValueError("Baseline evaluation must use the locked test split only.")
        return build_dataloader(
            split="test",
            training_config_path=temp_config_path,
            preprocessing_config_path=PROJECT_ROOT / self.eval_config["preprocessing_config"],
        )

    def run_inference(self, temp_config_path: Path) -> tuple[MetricAccumulator, list[ScanEvaluationResult]]:
        """Run inference and compute scan-level results."""
        test_loader = self._build_dataloader(temp_config_path)
        accumulator = MetricAccumulator()
        scan_buffers: dict[str, ScanBuffer] = {}
        scan_results: list[ScanEvaluationResult] = []

        max_batches = self.eval_config.get("max_batches")

        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader, start=1):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                filenames = batch["filename"]
                slice_indices = batch["slice_index"]
                depths = batch["volume_depth"]
                affines = batch["affine"]

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)

                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)
                targets = labels

                accumulator.update_batch(
                    predictions,
                    targets,
                    probabilities,
                    confidence_bins=self.confidence_bins,
                )

                batch_size = images.shape[0]
                for sample_index in range(batch_size):
                    filename = _extract_filename(filenames, sample_index)
                    slice_index = int(slice_indices[sample_index].item())
                    depth = int(depths[sample_index].item())
                    spacing_tuple = _extract_spacing(batch, sample_index)
                    shape_tuple = _extract_shape(batch, sample_index)
                    affine = affines[sample_index].cpu().numpy()

                    if filename not in scan_buffers:
                        scan_buffers[filename] = ScanBuffer(
                            filename=filename,
                            depth=depth,
                            original_shape=shape_tuple,
                            original_spacing=spacing_tuple,
                            affine=affine,
                        )

                    buffer = scan_buffers[filename]
                    buffer.add_slice(
                        slice_index=slice_index,
                        prediction=predictions[sample_index].cpu().numpy().astype(np.int64),
                        ground_truth=targets[sample_index].cpu().numpy().astype(np.int64),
                        image=images[sample_index].cpu().numpy(),
                        probabilities=probabilities[sample_index].cpu().numpy(),
                    )

                    if buffer.is_complete():
                        result = _evaluate_scan(buffer)
                        scan_results.append(result)
                        if bool(self.eval_config.get("save_predictions", True)):
                            self._save_scan_prediction(buffer, result)
                        del scan_buffers[filename]

                if max_batches is not None and batch_index >= int(max_batches):
                    break

        for filename, buffer in list(scan_buffers.items()):
            if buffer.is_complete():
                result = _evaluate_scan(buffer)
                scan_results.append(result)
                if bool(self.eval_config.get("save_predictions", True)):
                    self._save_scan_prediction(buffer, result)

        return accumulator, scan_results

    def _save_scan_prediction(self, buffer: ScanBuffer, result: ScanEvaluationResult) -> None:
        """Save stacked prediction volume for one scan."""
        prediction_volume, ground_truth_volume = stack_scan_predictions(buffer)
        output_path = self.predictions_dir / f"{buffer.filename}.npz"
        np.savez_compressed(
            output_path,
            prediction=prediction_volume,
            ground_truth=ground_truth_volume,
            spacing=np.array(buffer.original_spacing),
            affine=buffer.affine,
            macro_dice=result.macro_dice,
        )

    def _global_metrics(self, accumulator: MetricAccumulator) -> dict[str, Any]:
        """Compute global test metrics from accumulated counts."""
        dice = accumulator.per_class_dice()
        iou = accumulator.per_class_iou()
        precision = accumulator.per_class_precision()
        recall = accumulator.per_class_recall()
        sensitivity = recall.clone()
        specificity = accumulator.per_class_specificity()

        hemorrhage = list(HEMORRHAGE_CLASSES)
        micro_tp = accumulator.tp[hemorrhage].sum()
        micro_fp = accumulator.fp[hemorrhage].sum()
        micro_fn = accumulator.fn[hemorrhage].sum()
        return {
            "macro_dice": accumulator.macro_dice(),
            "micro_dice": float(
                (2.0 * micro_tp) / (2.0 * micro_tp + micro_fp + micro_fn + 1e-7)
            ),
            "per_class": {
                str(class_index): {
                    "dice": float(dice[class_index].item()),
                    "iou": float(iou[class_index].item()),
                    "precision": float(precision[class_index].item()),
                    "recall": float(recall[class_index].item()),
                    "sensitivity": float(sensitivity[class_index].item()),
                    "specificity": float(specificity[class_index].item()),
                }
                for class_index in range(NUM_CLASSES)
            },
            "confusion_matrix": accumulator.confusion_matrix.tolist(),
            "confidence_histogram": {
                "counts": accumulator.confidence_counts.tolist(),
                "bin_edges": accumulator.confidence_bin_edges.tolist(),
            },
        }

    def _failure_analysis(self, scan_results: list[ScanEvaluationResult]) -> dict[str, Any]:
        """Identify best, worst, volume-error, and low-confidence scans."""
        if not scan_results:
            return {
                "worst_dice_scans": [],
                "best_dice_scans": [],
                "largest_volume_error_scans": [],
                "lowest_confidence_scans": [],
            }

        sorted_by_dice = sorted(scan_results, key=lambda item: item.macro_dice)
        sorted_by_volume = sorted(scan_results, key=lambda item: item.total_volume_error_ml, reverse=True)
        sorted_by_confidence = sorted(scan_results, key=lambda item: item.study_confidence)

        def _scan_summary(result: ScanEvaluationResult) -> dict[str, Any]:
            return {
                "filename": result.filename,
                "macro_dice": result.macro_dice,
                "total_volume_error_ml": result.total_volume_error_ml,
                "study_confidence": result.study_confidence,
            }

        top_k = min(self.top_k, len(scan_results))
        return {
            "worst_dice_scans": [_scan_summary(item) for item in sorted_by_dice[:top_k]],
            "best_dice_scans": [_scan_summary(item) for item in sorted_by_dice[-top_k:][::-1]],
            "largest_volume_error_scans": [_scan_summary(item) for item in sorted_by_volume[:top_k]],
            "lowest_confidence_scans": [_scan_summary(item) for item in sorted_by_confidence[:top_k]],
        }

    def _write_csv_tables(
        self,
        global_metrics: dict[str, Any],
        scan_results: list[ScanEvaluationResult],
    ) -> None:
        """Write CSV result tables."""
        per_class_path = self.run_dir / "per_class_metrics.csv"
        with per_class_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["class_index", "subtype", "dice", "iou", "precision", "recall", "sensitivity", "specificity"],
            )
            writer.writeheader()
            for class_index in range(NUM_CLASSES):
                metrics = global_metrics["per_class"][str(class_index)]
                writer.writerow(
                    {
                        "class_index": class_index,
                        "subtype": format_subtype_name(class_index) if class_index else "BG",
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "sensitivity": metrics["sensitivity"],
                        "specificity": metrics["specificity"],
                    }
                )

        volume_rows: list[dict[str, Any]] = []
        for result in scan_results:
            for label, metrics in result.volume_errors.items():
                volume_rows.append(
                    {
                        "filename": result.filename,
                        "label": label,
                        "subtype": format_subtype_name(label),
                        "pred_ml": metrics["pred_ml"],
                        "gt_ml": metrics["gt_ml"],
                        "abs_error_ml": metrics["abs_error_ml"],
                        "rel_error": metrics["rel_error"],
                    }
                )
        pd.DataFrame(volume_rows).to_csv(self.run_dir / "volume_metrics.csv", index=False)

        confidence_rows: list[dict[str, Any]] = []
        for result in scan_results:
            row: dict[str, Any] = {
                "filename": result.filename,
                "study_confidence": result.study_confidence,
            }
            for label, value in result.class_confidences.items():
                row[f"class_{label}_confidence"] = value
            confidence_rows.append(row)
        pd.DataFrame(confidence_rows).to_csv(self.run_dir / "confidence_metrics.csv", index=False)

        summary_rows = [
            {
                "filename": result.filename,
                "macro_dice": result.macro_dice,
                "total_volume_error_ml": result.total_volume_error_ml,
                "study_confidence": result.study_confidence,
            }
            for result in scan_results
        ]
        pd.DataFrame(summary_rows).to_csv(self.run_dir / "prediction_summary.csv", index=False)

    def _generate_plots(
        self,
        global_metrics: dict[str, Any],
        scan_results: list[ScanEvaluationResult],
        failure_analysis: dict[str, Any],
    ) -> None:
        """Generate evaluation plots."""
        hemorrhage_dice = {
            class_index: global_metrics["per_class"][str(class_index)]["dice"]
            for class_index in HEMORRHAGE_CLASSES
        }
        plot_per_class_dice(
            hemorrhage_dice,
            output_path=self.plots_dir / "per_class_dice.png",
            title="Test Split Per-Class Dice",
        )

        mean_subtype_errors: dict[int, float] = {}
        for label in HEMORRHAGE_CLASSES:
            errors = [result.volume_errors[label]["abs_error_ml"] for result in scan_results]
            mean_subtype_errors[label] = float(np.mean(errors)) if errors else 0.0
        plot_volume_error(
            mean_subtype_errors,
            output_path=self.plots_dir / "volume_error_by_subtype.png",
            title="Mean Absolute Volume Error by Subtype (mL)",
        )

        if scan_results:
            example = scan_results[0]
            plot_prediction_example(
                example.image_slice,
                example.gt_slice,
                example.pred_slice,
                output_path=self.plots_dir / "prediction_example.png",
                title=f"Prediction Example — {example.filename}",
            )

            total_errors = [result.total_volume_error_ml for result in scan_results]
            plot_volume_error_distribution(
                total_errors,
                output_path=self.plots_dir / "volume_error_distribution.png",
            )

        plot_confusion_matrix(
            global_metrics["confusion_matrix"],
            CLASS_LABELS,
            output_path=self.plots_dir / "confusion_matrix.png",
            title="Test Split Confusion Matrix",
        )

        histogram = global_metrics["confidence_histogram"]
        plot_confidence_histogram(
            histogram["counts"],
            histogram["bin_edges"],
            output_path=self.plots_dir / "confidence_histogram.png",
            title="Test Split Softmax Confidence Histogram",
        )

        worst_names = {item["filename"] for item in failure_analysis["worst_dice_scans"]}
        best_names = {item["filename"] for item in failure_analysis["best_dice_scans"]}
        worst_examples = [
            {
                "filename": f"{result.filename} (Dice={result.macro_dice:.3f})",
                "image": result.image_slice,
                "ground_truth": result.gt_slice,
                "prediction": result.pred_slice,
            }
            for result in scan_results
            if result.filename in worst_names
        ][: self.top_k]
        best_examples = [
            {
                "filename": f"{result.filename} (Dice={result.macro_dice:.3f})",
                "image": result.image_slice,
                "ground_truth": result.gt_slice,
                "prediction": result.pred_slice,
            }
            for result in scan_results
            if result.filename in best_names
        ][: self.top_k]

        if worst_examples:
            plot_scan_examples(
                worst_examples,
                output_path=self.plots_dir / "worst_predictions.png",
                title="Worst Dice Scans (Representative Slice)",
            )
        if best_examples:
            plot_scan_examples(
                best_examples,
                output_path=self.plots_dir / "best_predictions.png",
                title="Best Dice Scans (Representative Slice)",
            )

    def _write_evaluation_report(
        self,
        global_metrics: dict[str, Any],
        scan_results: list[ScanEvaluationResult],
        failure_analysis: dict[str, Any],
    ) -> None:
        """Write markdown evaluation report with measured values only."""
        test_filenames = load_split_filenames("test")
        checkpoint_metrics = self.checkpoint_payload.get("val_metrics", {})

        lines = [
            "# Baseline Evaluation Report",
            "",
            "## Dataset",
            f"- Split: test (locked)",
            f"- Test scans in manifest: {len(test_filenames)}",
            f"- Test scans evaluated: {len(scan_results)}",
            f"- Split hash: `{compute_split_hash()}`",
            "",
            "## Model Configuration",
            f"- Model: `{self.model_config.get('model_name', 'monai_2d_unet')}`",
            f"- Input channels: {self.model_config.get('in_channels')}",
            f"- Output channels: {self.model_config.get('out_channels')}",
            f"- Channels: {self.model_config.get('channels')}",
            f"- Strides: {self.model_config.get('strides')}",
            "",
            "## Training Configuration",
            f"- Seed: {self.seed}",
            f"- Batch size (eval): {self.eval_config.get('batch_size')}",
            f"- Optimizer: {self.training_config.get('optimizer')}",
            f"- Learning rate: {self.training_config.get('learning_rate')}",
            f"- Epochs configured: {self.training_config.get('epochs')}",
            "",
            "## Checkpoint",
            f"- Path: `{self.checkpoint_path}`",
            f"- Epoch: {self.checkpoint_payload.get('epoch', 'unknown')}",
            f"- Validation macro Dice at checkpoint: {checkpoint_metrics.get(CHECKPOINT_SELECTION_METRIC, 'unknown')}",
            "",
            "## Global Test Metrics",
            f"- Macro Dice (classes 1-5): **{global_metrics['macro_dice']:.6f}**",
            f"- Micro Dice: {global_metrics['micro_dice']:.6f}",
            "",
            "### Per-Class Metrics",
            "| Class | Subtype | Dice | IoU | Precision | Recall |",
            "|------:|---------|-----:|----:|----------:|-------:|",
        ]

        for class_index in range(NUM_CLASSES):
            metrics = global_metrics["per_class"][str(class_index)]
            subtype = "BG" if class_index == 0 else SUBTYPE_LABELS.get(class_index, str(class_index))
            lines.append(
                f"| {class_index} | {subtype} | {metrics['dice']:.4f} | {metrics['iou']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} |"
            )

        if scan_results:
            volume_errors = [result.total_volume_error_ml for result in scan_results]
            confidences = [result.study_confidence for result in scan_results]
            lines.extend(
                [
                    "",
                    "## Volume Metrics (native spacing, mL)",
                    f"- Mean total hemorrhage volume error: {float(np.mean(volume_errors)):.4f} mL",
                    f"- Median total hemorrhage volume error: {float(np.median(volume_errors)):.4f} mL",
                    f"- Max total hemorrhage volume error: {float(np.max(volume_errors)):.4f} mL",
                    "",
                    "## Confidence Statistics",
                    f"- Mean study confidence: {float(np.mean(confidences)):.4f}",
                    f"- Median study confidence: {float(np.median(confidences)):.4f}",
                    f"- Min study confidence: {float(np.min(confidences)):.4f}",
                ]
            )

        lines.extend(
            [
                "",
                "## Failure Analysis",
                f"- Worst Dice scans logged: {len(failure_analysis['worst_dice_scans'])}",
                f"- Best Dice scans logged: {len(failure_analysis['best_dice_scans'])}",
                f"- Largest volume-error scans logged: {len(failure_analysis['largest_volume_error_scans'])}",
                f"- Lowest-confidence scans logged: {len(failure_analysis['lowest_confidence_scans'])}",
                "",
                "## Observations",
                "- Metrics above are computed on the locked test split using the saved best checkpoint.",
                f"- {len(scan_results)} complete scan volumes were reconstructed from 2.5D slice predictions.",
                "- Volume errors use native voxel spacing from NIfTI metadata.",
                "",
                "## Known Limitations",
                "- Input resolution remains native 512x512; resampling is not applied.",
                "- Class weights were not used in the Version 1 baseline loss.",
                "- Softmax confidence is reported without calibration (no temperature scaling).",
                "- Evaluation uses stacked slice predictions without inverse resampling.",
                "",
                "## Future Improvements",
                "- Lock and apply isotropic resampling with inverse native-grid volume mapping.",
                "- Add hemorrhage-aware slice sampling during training.",
                "- Compare against deferred architectures (Attention U-Net, nnU-Net, SwinUNETR).",
                "- Add calibrated confidence and test-time volume error breakdown by subtype.",
            ]
        )

        (self.run_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def evaluate(self, temp_config_path: Path) -> dict[str, Any]:
        """Run the full evaluation pipeline."""
        accumulator, scan_results = self.run_inference(temp_config_path)
        global_metrics = self._global_metrics(accumulator)
        failure_analysis = self._failure_analysis(scan_results)

        payload = {
            "run_id": self.run_dir.name,
            "split": "test",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint_payload.get("epoch"),
            "checkpoint_val_macro_dice": self.checkpoint_payload.get("val_metrics", {}).get(
                CHECKPOINT_SELECTION_METRIC
            ),
            "environment": self.environment,
            "global_metrics": global_metrics,
            "failure_analysis": failure_analysis,
            "scans_evaluated": len(scan_results),
        }

        with (self.run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        with (self.run_dir / "failure_analysis.json").open("w", encoding="utf-8") as handle:
            json.dump(failure_analysis, handle, indent=2)

        self._write_csv_tables(global_metrics, scan_results)
        self._generate_plots(global_metrics, scan_results, failure_analysis)
        self._write_evaluation_report(global_metrics, scan_results, failure_analysis)
        return payload


def evaluate_model(
    eval_config_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    run_dir: Path | str | None = None,
    temp_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute baseline model evaluation.

    Args:
        eval_config_path: Path to evaluation YAML config.
        checkpoint_path: Optional override for checkpoint file.
        run_dir: Optional explicit output directory.
        temp_dir: Directory for temporary dataloader config.

    Returns:
        Evaluation results dictionary.
    """
    eval_config = load_evaluation_config(eval_config_path)
    if checkpoint_path is not None:
        eval_config = dict(eval_config)
        eval_config["checkpoint_path"] = str(checkpoint_path)

    temp_root = Path(temp_dir) if temp_dir else PROJECT_ROOT / "reports" / "evaluation" / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_root / "dataloader.yaml"
    _write_training_config_override(eval_config, temp_config_path)

    evaluator = ModelEvaluator(eval_config, checkpoint_path=checkpoint_path, run_dir=run_dir)
    return evaluator.evaluate(temp_config_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate BrainHemorrhageAI baseline model on test split.")
    parser.add_argument("--config", type=Path, default=None, help="Path to evaluation config YAML.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to best_model.pt.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional explicit output directory.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    results = evaluate_model(
        eval_config_path=args.config,
        checkpoint_path=args.checkpoint,
        run_dir=args.run_dir,
    )
    logger.info(
        "Evaluation complete — macro Dice=%.6f, scans=%s, output=%s",
        results["global_metrics"]["macro_dice"],
        results["scans_evaluated"],
        results["run_id"],
    )


if __name__ == "__main__":
    main()
