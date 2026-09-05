"""Experiment logging for reproducible training runs."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from evaluation.checkpoint import CHECKPOINT_SELECTION_METRIC
from evaluation.plots import (
    plot_confusion_matrix,
    plot_confidence_histogram,
    plot_dice_curve,
    plot_loss_curve,
    plot_lr_curve,
    plot_per_class_dice,
    plot_volume_error,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"
DEFAULT_SPLITS_CSV = PROJECT_ROOT / "data" / "metadata" / "splits.csv"


def generate_run_id(prefix: str = "run") -> str:
    """Create a timestamp-based run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def compute_split_hash(splits_csv: Path | None = None) -> str:
    """Return SHA-256 hash of the split manifest file."""
    path = splits_csv or DEFAULT_SPLITS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def get_git_commit_hash(project_root: Path | None = None) -> str | None:
    """Return current git commit hash when available."""
    root = project_root or PROJECT_ROOT
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_environment_info(
    seed: int | None = None,
    splits_csv: Path | None = None,
) -> dict[str, Any]:
    """Collect runtime environment metadata for reproducibility."""
    try:
        import monai

        monai_version = monai.__version__
    except ImportError:
        monai_version = None

    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "monai_version": monai_version,
        "seed": seed,
        "git_commit": get_git_commit_hash(),
        "split_hash": compute_split_hash(splits_csv),
    }


class ExperimentLogger:
    """Manage per-run experiment artifacts under ``reports/training/<run_id>/``."""

    def __init__(
        self,
        run_id: str | None = None,
        reports_dir: Path | None = None,
        config: Mapping[str, Any] | None = None,
        seed: int | None = None,
        splits_csv: Path | None = None,
    ) -> None:
        self.run_id = run_id or generate_run_id()
        self.reports_dir = reports_dir or DEFAULT_REPORTS_DIR
        self.run_dir = self.reports_dir / self.run_id
        self.plots_dir = self.run_dir / "plots"
        self.config = dict(config or {})
        self.seed = seed
        self.splits_csv = splits_csv or DEFAULT_SPLITS_CSV

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._epoch_metrics: list[dict[str, Any]] = []
        self._summary_metrics: dict[str, Any] = {}
        self._environment = collect_environment_info(seed=seed, splits_csv=self.splits_csv)

        self.save_config()
        self.save_environment()
        self._write_split_hash_file()

    @property
    def environment(self) -> dict[str, Any]:
        """Return captured environment metadata."""
        return dict(self._environment)

    def save_config(self) -> Path:
        """Write the experiment configuration snapshot."""
        destination = self.run_dir / "config.yaml"
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)
        return destination

    def save_environment(self) -> Path:
        """Write environment metadata as JSON."""
        destination = self.run_dir / "environment.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self._environment, handle, indent=2)
        return destination

    def _write_split_hash_file(self) -> Path:
        destination = self.run_dir / "split_manifest_hash.txt"
        destination.write_text(self._environment["split_hash"] + "\n", encoding="utf-8")
        return destination

    def log_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Append one epoch's metrics to the in-memory log."""
        record = {"epoch": int(epoch), **dict(metrics)}
        self._epoch_metrics.append(record)
        self._append_epoch_jsonl(record)

    def _append_epoch_jsonl(self, record: Mapping[str, Any]) -> None:
        destination = self.run_dir / "epoch_metrics.jsonl"
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def set_summary_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Set final summary metrics for the run."""
        self._summary_metrics = dict(metrics)

    def save_metrics(self) -> Path:
        """Write summary metrics to ``metrics.json``."""
        destination = self.run_dir / "metrics.json"
        payload = {
            "run_id": self.run_id,
            "selection_metric": CHECKPOINT_SELECTION_METRIC,
            "summary": self._summary_metrics,
            "epochs_logged": len(self._epoch_metrics),
        }
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return destination

    def save_training_curves(
        self,
        epochs: list[int],
        train_loss: list[float],
        val_loss: list[float] | None = None,
        val_macro_dice: list[float] | None = None,
        learning_rates: list[float] | None = None,
    ) -> dict[str, Path]:
        """Generate loss, Dice, and learning-rate curve plots."""
        outputs: dict[str, Path] = {}
        outputs["loss_curve"] = plot_loss_curve(
            epochs,
            train_loss,
            val_loss=val_loss,
            output_path=self.plots_dir / "loss_curve.png",
        )
        if val_macro_dice is not None:
            outputs["dice_curve"] = plot_dice_curve(
                epochs,
                val_macro_dice,
                output_path=self.plots_dir / "dice_curve.png",
            )
        if learning_rates is not None:
            outputs["lr_curve"] = plot_lr_curve(
                epochs,
                learning_rates,
                output_path=self.plots_dir / "lr_curve.png",
            )
        return outputs

    def save_validation_metrics_csv(
        self,
        history: dict[str, list[float]],
        epochs: list[int],
    ) -> Path:
        """Write per-epoch validation metrics to CSV."""
        import csv

        destination = self.run_dir / "validation_metrics.csv"
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_macro_dice",
            "val_macro_iou",
            "learning_rate",
            "epoch_duration_sec",
        ]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, epoch in enumerate(epochs):
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": history["train_loss"][index],
                        "val_loss": history["val_loss"][index] if index < len(history["val_loss"]) else "",
                        "val_macro_dice": history["val_macro_dice"][index]
                        if index < len(history["val_macro_dice"])
                        else "",
                        "val_macro_iou": history["val_macro_iou"][index]
                        if index < len(history["val_macro_iou"])
                        else "",
                        "learning_rate": history["learning_rate"][index]
                        if index < len(history["learning_rate"])
                        else "",
                        "epoch_duration_sec": history["epoch_duration_sec"][index]
                        if index < len(history["epoch_duration_sec"])
                        else "",
                    }
                )
        return destination

    def save_per_class_dice_plot(self, class_dice: Mapping[int, float]) -> Path:
        """Save per-class Dice bar chart."""
        return plot_per_class_dice(
            class_dice,
            output_path=self.plots_dir / "per_class_dice.png",
        )

    def save_volume_error_plot(self, subtype_errors: Mapping[int, float]) -> Path:
        """Save absolute volume error plot."""
        return plot_volume_error(
            subtype_errors,
            output_path=self.plots_dir / "volume_error.png",
        )

    def save_confidence_histogram_plot(
        self,
        counts: list[int],
        bin_edges: list[float],
    ) -> Path:
        """Save softmax confidence histogram."""
        return plot_confidence_histogram(
            counts,
            bin_edges,
            output_path=self.plots_dir / "confidence_histogram.png",
        )

    def save_confusion_matrix_plot(
        self,
        matrix: list[list[int]],
        class_labels: list[str] | None = None,
    ) -> Path:
        """Save confusion matrix heatmap."""
        labels = class_labels or [str(index) for index in range(len(matrix))]
        return plot_confusion_matrix(
            matrix,
            labels,
            output_path=self.run_dir / "confusion_matrix.png",
        )

    def finalize(self, summary_metrics: Mapping[str, Any] | None = None) -> Path:
        """Persist summary metrics and return the run directory."""
        if summary_metrics is not None:
            self.set_summary_metrics(summary_metrics)
        self.save_metrics()
        return self.run_dir
