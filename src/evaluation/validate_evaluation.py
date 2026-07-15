"""
Synthetic validation for the BHSD evaluation framework.

Runs every evaluation component with known inputs and prints a summary.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluation.checkpoint import (
    CHECKPOINT_SELECTION_METRIC,
    load_checkpoint,
    resume_training,
    save_best_checkpoint,
    save_last_checkpoint,
    should_save_best_checkpoint,
)
from evaluation.confidence import (
    all_class_confidences,
    average_softmax_confidence,
    confidence_histogram,
    study_confidence,
)
from evaluation.experiment_logger import ExperimentLogger, compute_split_hash
from evaluation.metrics import (
    compute_segmentation_metrics,
    dice_score,
    iou_score,
    macro_dice,
    micro_dice,
    precision_score,
    recall_score,
    sensitivity_score,
    specificity_score,
)
from evaluation.plots import (
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_dice_curve,
    plot_loss_curve,
    plot_per_class_dice,
    plot_volume_error,
)
from evaluation.volume_metrics import (
    absolute_volume_error_ml,
    calculate_volume_ml,
    per_subtype_volume,
    per_subtype_volume_error,
    relative_volume_error,
    voxel_volume_mm3,
)


def _assert_close(actual: float, expected: float, tolerance: float = 1e-5) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {actual}.")


def validate_metrics() -> None:
    """Validate segmentation metrics on synthetic tensors."""
    target = torch.zeros(4, 4, dtype=torch.long)
    target[1:3, 1:3] = 1
    prediction = target.clone()

    perfect_dice = dice_score(prediction, target)
    _assert_close(float(perfect_dice[1].item()), 1.0)
    _assert_close(float(iou_score(prediction, target)[1].item()), 1.0)
    _assert_close(float(precision_score(prediction, target)[1].item()), 1.0)
    _assert_close(float(recall_score(prediction, target)[1].item()), 1.0)
    _assert_close(float(sensitivity_score(prediction, target)[1].item()), 1.0)
    _assert_close(float(specificity_score(prediction, target)[1].item()), 1.0)

    target_multi = torch.zeros(6, 6, dtype=torch.long)
    prediction_multi = torch.zeros(6, 6, dtype=torch.long)
    for class_index in range(1, 6):
        target_multi[class_index, 0] = class_index
        prediction_multi[class_index, 0] = class_index
    _assert_close(float(macro_dice(prediction_multi, target_multi).item()), 1.0)
    _assert_close(float(micro_dice(prediction_multi, target_multi).item()), 1.0)

    half_prediction = torch.zeros_like(target)
    half_prediction[1:3, 1:2] = 1
    half_dice = float(dice_score(half_prediction, target)[1].item())
    _assert_close(half_dice, 4.0 / 6.0, tolerance=1e-4)

    bundle = compute_segmentation_metrics(half_prediction, target)
    _assert_close(float(bundle.dice[1].item()), half_dice, tolerance=1e-4)


def validate_volume_metrics() -> None:
    """Validate native-space volume calculations."""
    spacing = (1.0, 1.0, 1.0)
    _assert_close(voxel_volume_mm3(spacing), 1.0)
    mask = np.zeros((2, 2, 2), dtype=np.int64)
    mask[0, 0, 0] = 1
    mask[0, 0, 1] = 2
    _assert_close(calculate_volume_ml(mask, spacing, label=1), 0.001)
    _assert_close(calculate_volume_ml(mask, spacing, label=2), 0.001)

    volumes = per_subtype_volume(mask, spacing)
    _assert_close(volumes[1], 0.001)
    _assert_close(volumes[2], 0.001)
    _assert_close(volumes[3], 0.0)

    _assert_close(absolute_volume_error_ml(1.2, 1.0), 0.2)
    _assert_close(relative_volume_error(1.2, 1.0), 0.2)

    pred = mask.copy()
    pred[0, 0, 0] = 2
    errors = per_subtype_volume_error(pred, mask, spacing)
    _assert_close(errors[1]["abs_error_ml"], 0.001)
    _assert_close(errors[2]["abs_error_ml"], 0.001)


def validate_confidence() -> None:
    """Validate softmax confidence utilities."""
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    probabilities = torch.softmax(logits, dim=0)
    predicted = probabilities.argmax(dim=0)

    mean_conf = float(average_softmax_confidence(probabilities).item())
    if mean_conf <= 0.5:
        raise AssertionError("Expected high average confidence on peaked softmax.")

    study = float(study_confidence(probabilities, predicted).item())
    if study <= 0.0:
        raise AssertionError("Expected positive study confidence.")

    class_scores = all_class_confidences(probabilities, predicted)
    if not class_scores:
        raise AssertionError("Expected class confidence dictionary.")

    counts, edges = confidence_histogram(probabilities)
    if counts.sum() != 4:
        raise AssertionError("Histogram voxel count mismatch.")


def validate_checkpoint() -> None:
    """Validate checkpoint save, load, and selection policy."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        best_path = root / "best_model.pt"
        last_path = root / "last_model.pt"

        state = {"layer.weight": torch.ones(2, 2)}
        optimizer_state = {"state": {}, "param_groups": []}
        metadata = {"note": "synthetic"}

        decision = save_best_checkpoint(
            best_path,
            model_state_dict=state,
            optimizer_state_dict=optimizer_state,
            epoch=1,
            val_metrics={CHECKPOINT_SELECTION_METRIC: 0.40},
            metadata=metadata,
            current_best_score=float("-inf"),
        )
        if not decision.is_best:
            raise AssertionError("First checkpoint should be saved as best.")

        decision = save_best_checkpoint(
            best_path,
            model_state_dict=state,
            optimizer_state_dict=optimizer_state,
            epoch=2,
            val_metrics={CHECKPOINT_SELECTION_METRIC: 0.35},
            metadata=metadata,
            current_best_score=0.40,
        )
        if decision.is_best:
            raise AssertionError("Lower macro Dice must not overwrite best checkpoint.")

        save_last_checkpoint(
            last_path,
            model_state_dict=state,
            optimizer_state_dict=optimizer_state,
            epoch=2,
            val_metrics={CHECKPOINT_SELECTION_METRIC: 0.35},
            metadata=metadata,
        )

        payload = load_checkpoint(best_path)
        if float(payload["selection_score"]) != 0.40:
            raise AssertionError("Best checkpoint should retain epoch-1 score.")

        bundle = resume_training(last_path)
        if bundle.epoch != 2:
            raise AssertionError("Resume should restore epoch 2.")

        no_improve = should_save_best_checkpoint(0.40, 0.40)
        if no_improve.is_best:
            raise AssertionError("Equal macro Dice should not count as improvement.")


def validate_experiment_logger() -> None:
    """Validate experiment directory creation and artifact writes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reports_dir = Path(temp_dir) / "training"
        logger = ExperimentLogger(
            run_id="validation_run",
            reports_dir=reports_dir,
            config={"batch_size": 4},
            seed=42,
        )

        logger.log_epoch(
            epoch=1,
            metrics={
                "train_loss": 0.5,
                CHECKPOINT_SELECTION_METRIC: 0.42,
            },
        )
        logger.save_training_curves(
            epochs=[1],
            train_loss=[0.5],
            val_loss=[0.6],
            val_macro_dice=[0.42],
        )
        logger.save_per_class_dice_plot({1: 0.5, 2: 0.7})
        logger.save_volume_error_plot({1: 0.3, 2: 0.1})
        logger.save_confidence_histogram_plot([1, 2, 3, 4], [0.0, 0.25, 0.5, 0.75, 1.0])
        logger.save_confusion_matrix_plot(
            [[8, 1], [2, 9]],
            class_labels=["A", "B"],
        )
        run_dir = logger.finalize(
            summary_metrics={
                CHECKPOINT_SELECTION_METRIC: 0.42,
                "best_epoch": 1,
            }
        )

        required = [
            run_dir / "config.yaml",
            run_dir / "environment.json",
            run_dir / "metrics.json",
            run_dir / "epoch_metrics.jsonl",
            run_dir / "split_manifest_hash.txt",
            run_dir / "confusion_matrix.png",
            run_dir / "plots" / "loss_curve.png",
            run_dir / "plots" / "dice_curve.png",
            run_dir / "plots" / "per_class_dice.png",
            run_dir / "plots" / "volume_error.png",
            run_dir / "plots" / "confidence_histogram.png",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise AssertionError(f"Missing experiment artifacts: {missing}")

        metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if metrics_payload["summary"][CHECKPOINT_SELECTION_METRIC] != 0.42:
            raise AssertionError("Summary metrics not persisted correctly.")

        split_hash = compute_split_hash()
        if len(split_hash) != 64:
            raise AssertionError("Split hash should be a SHA-256 hex digest.")


def validate_plots() -> None:
    """Validate standalone plot generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        outputs = [
            plot_loss_curve([1, 2], [0.8, 0.5], val_loss=[0.9, 0.6], output_path=root / "loss.png"),
            plot_dice_curve([1, 2], [0.4, 0.55], output_path=root / "dice.png"),
            plot_per_class_dice({1: 0.5, 2: 0.7}, output_path=root / "class_dice.png"),
            plot_volume_error({1: 0.2, 2: 0.4}, output_path=root / "volume.png"),
            plot_confidence_histogram([1, 2, 1], [0.0, 0.33, 0.66, 1.0], output_path=root / "conf.png"),
            plot_confusion_matrix([[5, 1], [0, 4]], ["A", "B"], output_path=root / "cm.png"),
        ]
        missing = [path for path in outputs if not path.exists()]
        if missing:
            raise AssertionError(f"Plot files missing: {missing}")


def main() -> None:
    """Run all evaluation component checks."""
    checks = [
        ("metrics", validate_metrics),
        ("volume_metrics", validate_volume_metrics),
        ("confidence", validate_confidence),
        ("checkpoint", validate_checkpoint),
        ("experiment_logger", validate_experiment_logger),
        ("plots", validate_plots),
    ]

    results: list[tuple[str, str]] = []
    for name, check in checks:
        check()
        results.append((name, "PASSED"))

    print("BHSD evaluation framework validation - PASSED")
    print()
    for name, status in results:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
