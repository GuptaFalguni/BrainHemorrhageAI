"""Run the first reproducible baseline experiment (Phase 4.2)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset.dataloader import build_dataloader
from evaluation.checkpoint import CHECKPOINT_SELECTION_METRIC
from evaluation.evaluate_model import evaluate_model, load_evaluation_config
from evaluation.experiment_logger import ExperimentLogger, collect_environment_info
from experiments.experiment_tracker import (
    append_experiment_index,
    assemble_experiment_folder,
    load_experiment_config,
    write_experiment_report,
)
from experiments.preflight import PreflightError, print_preflight_report, run_preflight_checks
from models.monai_unet import build_model
from training.config_loader import (
    PROJECT_ROOT,
    load_model_config,
    load_training_config,
    load_yaml_config,
    resolve_device,
)
from training.seed import set_seed
from training.trainer import Trainer

logger = logging.getLogger(__name__)
DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "config" / "experiments" / "experiment_001.yaml"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _merge_training_config(
    base_config: dict[str, Any],
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Apply experiment overrides to the base training config."""
    merged = dict(base_config)
    override_keys = (
        "epochs",
        "resume",
        "run_id",
        "checkpoint_dir",
        "reports_dir",
        "early_stopping_patience",
        "batch_size",
        "learning_rate",
    )
    for key in override_keys:
        if key in experiment_config and experiment_config[key] is not None:
            merged[key] = experiment_config[key]
    return merged


def _build_training_snapshot(
    training_config: dict[str, Any],
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Build config snapshot for experiment logging."""
    preprocessing_path = PROJECT_ROOT / str(experiment_config.get("preprocessing_config", "config/preprocessing.yaml"))
    preprocessing_config = load_yaml_config(preprocessing_path) if preprocessing_path.exists() else {}
    return {
        "experiment": experiment_config,
        "training": training_config,
        "model": model_config,
        "preprocessing": preprocessing_config,
    }


def _training_summary_from_run(
    trainer: Trainer,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Extract training summary metrics from trainer state."""
    best_val_loss = min(trainer.history["val_loss"]) if trainer.history["val_loss"] else None
    return {
        "val_macro_dice": trainer.best_val_macro_dice,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(trainer.history["train_loss"]),
        "learning_rate": training_config.get("learning_rate"),
        "batch_size": training_config.get("batch_size"),
        "best_checkpoint": str(trainer.best_checkpoint_path),
        "last_checkpoint": str(trainer.last_checkpoint_path),
    }


def run_baseline_experiment(
    experiment_config_path: Path | str | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_eval_batches: int | None = None,
    skip_training: bool = False,
    write_experiment_index: bool = True,
    experiment_index_path: Path | None = None,
) -> dict[str, Any]:
    """Execute preflight, training, evaluation, and experiment tracking.

    Args:
        experiment_config_path: Path to experiment YAML.
        max_train_batches: Optional training batch cap (smoke tests).
        max_val_batches: Optional validation batch cap.
        max_eval_batches: Optional evaluation batch cap.
        skip_training: Skip training if checkpoint already exists.
        write_experiment_index: Append a row to the experiment index CSV.
        experiment_index_path: Optional override for the index CSV path.

    Returns:
        Dictionary with paths and measured metrics.
    """
    experiment_config = load_experiment_config(experiment_config_path or DEFAULT_EXPERIMENT_CONFIG)
    experiment_id = str(experiment_config["experiment_id"])

    preflight = run_preflight_checks()
    print_preflight_report(preflight)

    base_training = load_training_config(PROJECT_ROOT / experiment_config["training_config"])
    training_config = _merge_training_config(base_training, experiment_config)
    model_config = load_model_config(PROJECT_ROOT / experiment_config["model_config"])

    seed = int(training_config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(training_config.get("device", "auto")))

    checkpoint_dir = PROJECT_ROOT / str(training_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / str(training_config.get("best_checkpoint_name", "best_model.pt"))

    training_start = time.perf_counter()
    training_run_dir: Path | None = None
    trainer: Trainer | None = None

    if not skip_training or not best_checkpoint.exists():
        temp_training_config = checkpoint_dir / "training_merged.yaml"
        with temp_training_config.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(training_config, handle, sort_keys=False)

        train_loader = build_dataloader("train", training_config_path=temp_training_config)
        val_loader = build_dataloader("val", training_config_path=temp_training_config)
        model = build_model(model_config)

        snapshot = _build_training_snapshot(training_config, model_config, experiment_config)
        experiment_logger = ExperimentLogger(
            run_id=str(training_config.get("run_id", experiment_id)),
            reports_dir=PROJECT_ROOT / str(training_config.get("reports_dir", "reports/training")),
            config=snapshot,
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=training_config,
            model_config=model_config,
            experiment_logger=experiment_logger,
            device=device,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )

        if bool(training_config.get("resume", False)) and trainer.last_checkpoint_path.exists():
            trainer.load_checkpoint(trainer.last_checkpoint_path)

        trainer.fit()
        training_run_dir = experiment_logger.run_dir
    else:
        logger.info("Skipping training; using existing checkpoint at %s", best_checkpoint)
        training_run_dir = PROJECT_ROOT / str(training_config.get("reports_dir", "reports/training")) / str(
            training_config.get("run_id", experiment_id)
        )

    training_time_sec = time.perf_counter() - training_start

    eval_config = load_evaluation_config(PROJECT_ROOT / experiment_config["evaluation_config"])
    eval_config = dict(eval_config)
    eval_config["checkpoint_path"] = str(best_checkpoint)
    eval_config["run_id"] = str(experiment_config.get("evaluation_run_id", f"{experiment_id}_eval"))
    eval_config["reports_dir"] = str(experiment_config.get("evaluation_reports_dir", "reports/evaluation"))
    if max_eval_batches is not None:
        eval_config["max_batches"] = max_eval_batches

    temp_eval_config = checkpoint_dir / "evaluation.yaml"
    with temp_eval_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(eval_config, handle, sort_keys=False)

    eval_results = evaluate_model(
        eval_config_path=temp_eval_config,
        checkpoint_path=best_checkpoint,
        run_dir=PROJECT_ROOT / eval_config["reports_dir"] / eval_config["run_id"],
    )
    evaluation_run_dir = PROJECT_ROOT / eval_config["reports_dir"] / eval_config["run_id"]

    experiment_dir, generated_reports = assemble_experiment_folder(
        experiment_config,
        training_run_dir=training_run_dir,
        evaluation_run_dir=evaluation_run_dir,
    )

    training_summary = _training_summary_from_run(trainer, training_config) if trainer else {
        "val_macro_dice": None,
        "best_val_loss": None,
        "epochs_completed": 0,
        "learning_rate": training_config.get("learning_rate"),
        "batch_size": training_config.get("batch_size"),
        "best_checkpoint": str(best_checkpoint),
    }

    hardware = collect_environment_info(seed=seed)
    write_experiment_report(
        experiment_dir=experiment_dir,
        experiment_config=experiment_config,
        preflight={
            "device": preflight.device,
            "cuda_available": preflight.cuda_available,
            "seed": preflight.seed,
        },
        training_summary=training_summary,
        eval_results=eval_results,
        training_time_sec=training_time_sec,
        hardware=hardware,
        generated_reports=generated_reports,
    )

    test_dice = eval_results.get("global_metrics", {}).get("macro_dice")
    if write_experiment_index:
        append_experiment_index(
            {
                "experiment_id": experiment_id,
                "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "model": model_config.get("model_name", "monai_2d_unet"),
                "epochs": training_summary.get("epochs_completed", experiment_config.get("epochs")),
                "best_val_dice": training_summary.get("val_macro_dice"),
                "test_dice": test_dice,
                "checkpoint": str(best_checkpoint),
                "status": "completed",
                "notes": experiment_config.get("description", ""),
            },
            index_path=experiment_index_path,
        )

    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "training_run_dir": str(training_run_dir),
        "evaluation_run_dir": str(evaluation_run_dir),
        "best_checkpoint": str(best_checkpoint),
        "training_time_sec": training_time_sec,
        "best_val_dice": training_summary.get("val_macro_dice"),
        "test_dice": test_dice,
        "eval_results": eval_results,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run BrainHemorrhageAI baseline experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
        help="Path to experiment config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    _configure_logging()
    args = parse_args()
    try:
        summary = run_baseline_experiment(experiment_config_path=args.config)
    except PreflightError as error:
        logger.error("Pre-training check failed: %s", error)
        raise SystemExit(1) from error

    logger.info("Baseline experiment complete: %s", summary["experiment_id"])
    logger.info("Experiment folder: %s", summary["experiment_dir"])
    logger.info("Best val Dice: %s", summary["best_val_dice"])
    logger.info("Test Dice: %s", summary["test_dice"])


if __name__ == "__main__":
    main()
