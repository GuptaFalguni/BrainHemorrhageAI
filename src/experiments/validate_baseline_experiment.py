"""Smoke test for the baseline experiment pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml

from experiments.experiment_tracker import INDEX_CSV
from experiments.preflight import run_preflight_checks
from experiments.run_baseline_experiment import run_baseline_experiment
from training.config_loader import PROJECT_ROOT


def _write_smoke_experiment_config(base_config: dict, output_path: Path, temp_root: Path) -> None:
    """Write a minimal experiment config for smoke testing."""
    config = dict(base_config)
    config["experiment_id"] = "smoke_experiment"
    config["epochs"] = 1
    config["resume"] = False
    config["early_stopping_patience"] = 100
    config["checkpoint_dir"] = str(temp_root / "checkpoints" / "smoke_experiment")
    config["reports_dir"] = str(temp_root / "reports" / "training")
    config["evaluation_reports_dir"] = str(temp_root / "reports" / "evaluation")
    config["experiment_output_dir"] = str(temp_root / "reports" / "experiments" / "smoke_experiment")
    config["run_id"] = "smoke_experiment_train"
    config["evaluation_run_id"] = "smoke_experiment_eval"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def run_validation() -> None:
    """Run end-to-end baseline experiment smoke test."""
    preflight = run_preflight_checks()
    if not preflight.passed:
        raise AssertionError("Preflight checks failed.")

    base_config_path = PROJECT_ROOT / "config" / "experiments" / "experiment_001.yaml"
    with base_config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    production_index = INDEX_CSV.read_text(encoding="utf-8") if INDEX_CSV.exists() else ""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        config_path = temp_root / "experiment_smoke.yaml"
        _write_smoke_experiment_config(base_config, config_path, temp_root)

        summary = run_baseline_experiment(
            experiment_config_path=config_path,
            max_train_batches=2,
            max_val_batches=2,
            max_eval_batches=40,
            write_experiment_index=False,
        )

        experiment_dir = Path(summary["experiment_dir"])
        required = [
            experiment_dir / "experiment.md",
            experiment_dir / "experiment_config.yaml",
            experiment_dir / "plots" / "loss_curve.png",
            experiment_dir / "plots" / "dice_curve.png",
            experiment_dir / "plots" / "lr_curve.png",
            experiment_dir / "validation_metrics.csv",
            experiment_dir / "evaluation" / "metrics.json",
            experiment_dir / "evaluation" / "evaluation_report.md",
            experiment_dir / "evaluation" / "plots" / "confusion_matrix.png",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise AssertionError(f"Missing experiment artifacts: {missing}")

        checkpoint = Path(summary["best_checkpoint"])
        if not checkpoint.exists():
            raise AssertionError("Best checkpoint was not saved.")

        metrics = json.loads((experiment_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
        if "global_metrics" not in metrics:
            raise AssertionError("Evaluation metrics.json incomplete.")

        if not str(experiment_dir).startswith(str(temp_root)):
            raise AssertionError("Smoke test wrote outside the temporary directory.")

    current_index = INDEX_CSV.read_text(encoding="utf-8") if INDEX_CSV.exists() else ""
    if current_index != production_index:
        raise AssertionError("Production experiment index was modified during smoke testing.")


def main() -> None:
    """Run validation and print summary."""
    run_validation()
    print("BASELINE EXPERIMENT PIPELINE PASSED")


if __name__ == "__main__":
    main()
