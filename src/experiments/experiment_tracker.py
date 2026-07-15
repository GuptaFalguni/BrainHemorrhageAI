"""Experiment artifact tracking and reporting for baseline runs."""

from __future__ import annotations

import csv
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evaluation.volume_metrics import SUBTYPE_LABELS, format_subtype_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "reports" / "experiments"
INDEX_CSV = EXPERIMENTS_ROOT / "index.csv"
CLASS_LABELS = ["BG", "EDH", "SDH", "SAH", "IPH", "IVH"]


def _copy_if_exists(source: Path, destination: Path) -> None:
    """Copy a file when the source exists."""
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_tree_files(sources: list[Path], destination_dir: Path) -> list[str]:
    """Copy files into destination and return relative paths."""
    copied: list[str] = []
    for source in sources:
        if source.exists():
            target = destination_dir / source.name
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                _copy_if_exists(source, target)
            copied.append(str(target.relative_to(destination_dir.parent)))
    return copied


def _most_common_misclassification(confusion_matrix: list[list[int]]) -> dict[str, Any]:
    """Find the highest off-diagonal confusion count."""
    matrix = np.array(confusion_matrix, dtype=np.int64)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    if off_diagonal.sum() == 0:
        return {"ground_truth": None, "predicted": None, "count": 0}
    flat_index = int(off_diagonal.argmax())
    gt_index, pred_index = np.unravel_index(flat_index, matrix.shape)
    return {
        "ground_truth": CLASS_LABELS[gt_index],
        "predicted": CLASS_LABELS[pred_index],
        "count": int(matrix[gt_index, pred_index]),
    }


def _failure_analysis_summary(eval_metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize failure analysis from measured evaluation metrics."""
    per_class = eval_metrics.get("global_metrics", {}).get("per_class", {})
    hemorrhage_dice = {
        int(label): values["dice"]
        for label, values in per_class.items()
        if int(label) in SUBTYPE_LABELS
    }

    worst_subtype = min(hemorrhage_dice, key=hemorrhage_dice.get) if hemorrhage_dice else None
    best_subtype = max(hemorrhage_dice, key=hemorrhage_dice.get) if hemorrhage_dice else None

    failure = eval_metrics.get("failure_analysis", {})
    largest_volume = failure.get("largest_volume_error_scans", [{}])[0] if failure else {}
    lowest_confidence = failure.get("lowest_confidence_scans", [{}])[0] if failure else {}
    confusion = eval_metrics.get("global_metrics", {}).get("confusion_matrix", [])
    misclass = _most_common_misclassification(confusion)

    return {
        "worst_dice_subtype": format_subtype_name(worst_subtype) if worst_subtype else "N/A",
        "worst_dice_value": hemorrhage_dice.get(worst_subtype) if worst_subtype else None,
        "best_dice_subtype": format_subtype_name(best_subtype) if best_subtype else "N/A",
        "best_dice_value": hemorrhage_dice.get(best_subtype) if best_subtype else None,
        "largest_volume_error_scan": largest_volume.get("filename", "N/A"),
        "largest_volume_error_ml": largest_volume.get("total_volume_error_ml"),
        "lowest_confidence_scan": lowest_confidence.get("filename", "N/A"),
        "lowest_confidence_value": lowest_confidence.get("study_confidence"),
        "most_common_misclassification": misclass,
    }


def append_experiment_index(
    row: dict[str, Any],
    index_path: Path | None = None,
) -> None:
    """Append one row to the experiment index CSV."""
    target_index = index_path or INDEX_CSV
    target_index.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "date",
        "model",
        "epochs",
        "best_val_dice",
        "test_dice",
        "checkpoint",
        "status",
        "notes",
    ]
    file_exists = target_index.exists()
    with target_index.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_experiment_report(
    experiment_dir: Path,
    experiment_config: dict[str, Any],
    preflight: dict[str, Any],
    training_summary: dict[str, Any],
    eval_results: dict[str, Any],
    training_time_sec: float,
    hardware: dict[str, Any],
    generated_reports: list[str],
) -> Path:
    """Write experiment.md with measured values only."""
    experiment_id = experiment_config.get("experiment_id", "experiment_unknown")
    eval_global = eval_results.get("global_metrics", {})
    per_class = eval_global.get("per_class", {})
    failure_summary = _failure_analysis_summary(eval_results)

    best_val_dice = training_summary.get("val_macro_dice")
    best_val_loss = training_summary.get("best_val_loss")
    test_dice = eval_global.get("macro_dice")
    hemorrhage_iou = [
        float(per_class[str(label)]["iou"])
        for label in SUBTYPE_LABELS
        if str(label) in per_class
    ]
    test_iou = float(np.mean(hemorrhage_iou)) if hemorrhage_iou else None

    confidence_hist = eval_global.get("confidence_histogram", {})
    volume_errors: list[float] = []
    volume_csv = experiment_dir / "evaluation" / "volume_metrics.csv"
    if volume_csv.exists():
        import pandas as pd

        volume_df = pd.read_csv(volume_csv)
        if "abs_error_ml" in volume_df.columns:
            volume_errors = volume_df.groupby("filename")["abs_error_ml"].sum().tolist()

    lines = [
        f"# {experiment_id}",
        "",
        "## Experiment ID",
        experiment_id,
        "",
        "## Date",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "",
        "## Git Commit",
        str(hardware.get("git_commit", "unknown")),
        "",
        "## Model",
        str(experiment_config.get("model_config", "config/model.yaml")),
        "",
        "## Dataset",
        "BHSD label_192 (192 scans, locked 70/15/15 split)",
        "",
        "## Split",
        "train / val / test from data/metadata/splits.csv (test used for final metrics only)",
        "",
        "## Configuration Files Used",
        f"- Training: `{experiment_config.get('training_config')}`",
        f"- Model: `{experiment_config.get('model_config')}`",
        f"- Preprocessing: `{experiment_config.get('preprocessing_config')}`",
        f"- Evaluation: `{experiment_config.get('evaluation_config')}`",
        f"- Experiment: `config/experiments/{experiment_id}.yaml`",
        "",
        "## Training Time",
        f"{training_time_sec:.1f} seconds ({training_time_sec / 60.0:.2f} minutes)",
        "",
        "## Hardware",
        f"- Platform: {hardware.get('platform', platform.platform())}",
        f"- Device: {preflight.get('device', 'unknown')}",
        f"- CUDA available: {preflight.get('cuda_available', False)}",
        "",
        "## Training Settings",
        f"- Epochs: {training_summary.get('epochs_completed', experiment_config.get('epochs'))}",
        f"- Learning rate: {training_summary.get('learning_rate', 'see config')}",
        f"- Batch size: {training_summary.get('batch_size', 'see config')}",
        f"- Seed: {preflight.get('seed', 'unknown')}",
        "",
        "## Best Validation Metrics",
        f"- Best validation macro Dice: {best_val_dice if best_val_dice is not None else 'N/A'}",
        f"- Best validation loss: {best_val_loss if best_val_loss is not None else 'N/A'}",
        "",
        "## Test Metrics",
        f"- Test macro Dice (classes 1-5): {test_dice if test_dice is not None else 'N/A'}",
        f"- Test macro IoU (classes 1-5): {test_iou if test_iou is not None else 'N/A'}",
        "",
        "### Per-Class Test Dice",
    ]

    for class_index in sorted(SUBTYPE_LABELS.keys()):
        metrics = per_class.get(str(class_index), {})
        dice_value = metrics.get("dice", "N/A")
        lines.append(f"- {format_subtype_name(class_index)}: {dice_value}")

    lines.extend(
        [
            "",
            "## Volume Error",
            f"- Mean total volume error (mL): {float(np.mean(volume_errors)) if volume_errors else 'N/A'}",
            f"- Median total volume error (mL): {float(np.median(volume_errors)) if volume_errors else 'N/A'}",
            "",
            "## Confidence Statistics",
            f"- Histogram bins: {len(confidence_hist.get('counts', []))}",
            "",
            "## Checkpoint Path",
            str(training_summary.get("best_checkpoint", "N/A")),
            "",
            "## Generated Reports",
        ]
    )
    for report_path in generated_reports:
        lines.append(f"- `{report_path}`")

    lines.extend(
        [
            "",
            "## Failure Analysis (Measured)",
            f"- Worst Dice subtype: {failure_summary['worst_dice_subtype']} "
            f"(Dice={failure_summary['worst_dice_value']})",
            f"- Best Dice subtype: {failure_summary['best_dice_subtype']} "
            f"(Dice={failure_summary['best_dice_value']})",
            f"- Largest volume error scan: {failure_summary['largest_volume_error_scan']} "
            f"({failure_summary['largest_volume_error_ml']} mL)",
            f"- Lowest confidence scan: {failure_summary['lowest_confidence_scan']} "
            f"(confidence={failure_summary['lowest_confidence_value']})",
            f"- Most common misclassification: "
            f"{failure_summary['most_common_misclassification']['ground_truth']} "
            f"-> {failure_summary['most_common_misclassification']['predicted']} "
            f"(count={failure_summary['most_common_misclassification']['count']})",
            "",
            "## Known Limitations",
            "- Native 512x512 slices without resampling.",
            "- Five-epoch baseline run; not converged for production use.",
            "- Softmax confidence reported without calibration.",
            "",
            "## Next Experiment Ideas",
            "- Increase epochs toward architecture default (150).",
            "- Lock input resolution after ROI analysis.",
            "- Add class-weighted DiceCE after train-set frequency analysis.",
        ]
    )

    report_path = experiment_dir / "experiment.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def assemble_experiment_folder(
    experiment_config: dict[str, Any],
    training_run_dir: Path,
    evaluation_run_dir: Path,
) -> tuple[Path, list[str]]:
    """Copy training and evaluation artifacts into the experiment folder."""
    experiment_dir = PROJECT_ROOT / str(
        experiment_config.get("experiment_output_dir", "reports/experiments/experiment_001")
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = experiment_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    def _record(path: Path) -> None:
        generated.append(str(path.relative_to(experiment_dir)))

    training_plot_sources = [
        training_run_dir / "plots" / "loss_curve.png",
        training_run_dir / "plots" / "dice_curve.png",
        training_run_dir / "plots" / "lr_curve.png",
        training_run_dir / "validation_metrics.csv",
    ]
    for source in training_plot_sources:
        if source.exists():
            target = plots_dir / source.name if source.suffix == ".png" else experiment_dir / source.name
            _copy_if_exists(source, target)
            _record(target)

    eval_artifacts = [
        evaluation_run_dir / "metrics.json",
        evaluation_run_dir / "evaluation_report.md",
        evaluation_run_dir / "failure_analysis.json",
        evaluation_run_dir / "per_class_metrics.csv",
        evaluation_run_dir / "volume_metrics.csv",
        evaluation_run_dir / "confidence_metrics.csv",
        evaluation_run_dir / "prediction_summary.csv",
        evaluation_run_dir / "plots",
        evaluation_run_dir / "predictions",
    ]
    for source in eval_artifacts:
        if not source.exists():
            continue
        if source.is_dir():
            target = eval_dir / source.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            _record(target)
        else:
            target = eval_dir / source.name
            _copy_if_exists(source, target)
            _record(target)

    with (experiment_dir / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment_config, handle, sort_keys=False)

    return experiment_dir, generated


def load_experiment_config(path: Path | str) -> dict[str, Any]:
    """Load experiment YAML configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid experiment config: {config_path}")
    return config
