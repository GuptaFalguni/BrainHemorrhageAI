"""Smoke test for the post-training evaluation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import yaml

from evaluation.checkpoint import CHECKPOINT_SELECTION_METRIC, save_best_checkpoint
from evaluation.evaluate_model import evaluate_model, load_evaluation_config
from models.monai_unet import build_model
from training.config_loader import load_model_config, load_training_config


def _create_smoke_checkpoint(checkpoint_path: Path) -> None:
    """Create a synthetic best checkpoint for pipeline validation."""
    model_config = load_model_config()
    training_config = load_training_config()
    model = build_model(model_config)
    save_best_checkpoint(
        checkpoint_path,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={"state": {}, "param_groups": []},
        epoch=1,
        val_metrics={CHECKPOINT_SELECTION_METRIC: 0.25},
        metadata={
            "model_config": model_config,
            "training_config": training_config,
            "note": "synthetic smoke-test checkpoint",
        },
        current_best_score=float("-inf"),
    )


def _write_smoke_eval_config(base_config: dict[str, object], output_path: Path, checkpoint_path: Path, run_dir: Path) -> None:
    """Write temporary evaluation config for smoke testing."""
    config = dict(base_config)
    config["checkpoint_path"] = str(checkpoint_path)
    config["run_id"] = "smoke_eval"
    config["reports_dir"] = str(run_dir.parent)
    config["max_batches"] = 40
    config["max_scans"] = None
    config["save_predictions"] = True
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def run_validation() -> Path:
    """Run evaluation pipeline smoke test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint_path = temp_root / "best_model.pt"
        _create_smoke_checkpoint(checkpoint_path)

        reports_root = temp_root / "reports" / "evaluation"
        eval_config_path = temp_root / "evaluation.yaml"
        base_config = load_evaluation_config()
        _write_smoke_eval_config(
            base_config,
            eval_config_path,
            checkpoint_path=checkpoint_path,
            run_dir=reports_root / "smoke_eval",
        )

        results = evaluate_model(
            eval_config_path=eval_config_path,
            run_dir=reports_root / "smoke_eval",
            temp_dir=temp_root / ".tmp",
        )

        run_dir = reports_root / "smoke_eval"
        required = [
            run_dir / "metrics.json",
            run_dir / "per_class_metrics.csv",
            run_dir / "volume_metrics.csv",
            run_dir / "confidence_metrics.csv",
            run_dir / "prediction_summary.csv",
            run_dir / "failure_analysis.json",
            run_dir / "evaluation_report.md",
            run_dir / "plots" / "per_class_dice.png",
            run_dir / "plots" / "volume_error_distribution.png",
            run_dir / "plots" / "confusion_matrix.png",
            run_dir / "plots" / "confidence_histogram.png",
            run_dir / "plots" / "prediction_example.png",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise AssertionError(f"Missing evaluation artifacts: {missing}")

        metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if "global_metrics" not in metrics_payload:
            raise AssertionError("metrics.json missing global_metrics.")
        if metrics_payload["global_metrics"]["macro_dice"] < 0.0:
            raise AssertionError("Invalid macro Dice in evaluation output.")

        if results["scans_evaluated"] < 1:
            raise AssertionError("Expected at least one complete scan evaluation.")

        predictions = list((run_dir / "predictions").glob("*.npz"))
        if not predictions:
            raise AssertionError("Expected saved prediction files.")

        return run_dir


def main() -> None:
    """Run validation and print summary."""
    run_validation()
    print("BASELINE EVALUATION PASSED")


if __name__ == "__main__":
    main()
