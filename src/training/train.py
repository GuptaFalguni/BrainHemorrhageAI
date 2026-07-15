"""Training entry point for BrainHemorrhageAI baseline experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from dataset.dataloader import build_dataloader
from evaluation.experiment_logger import ExperimentLogger
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


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _build_experiment_config(
    training_config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge configs for experiment logging."""
    preprocessing_path = PROJECT_ROOT / str(
        training_config.get("preprocessing_config", "config/preprocessing.yaml")
    )
    preprocessing_config: dict[str, Any] = {}
    if preprocessing_path.exists():
        preprocessing_config = load_yaml_config(preprocessing_path)

    return {
        "training": training_config,
        "model": model_config,
        "preprocessing": preprocessing_config,
    }


def _build_experiment_logger(training_config: dict[str, Any], experiment_config: dict[str, Any]) -> ExperimentLogger:
    """Create the experiment logger for a training run."""
    reports_dir = PROJECT_ROOT / str(training_config.get("reports_dir", "reports/training"))
    run_id = training_config.get("run_id")
    return ExperimentLogger(
        run_id=run_id,
        reports_dir=reports_dir,
        config=experiment_config,
        seed=int(training_config.get("seed", 42)),
    )


def run_training(
    training_config_path: Path | None = None,
    model_config_path: Path | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    """Execute baseline training with optional batch limits for smoke tests.

    Args:
        training_config_path: Optional override for ``config/training.yaml``.
        model_config_path: Optional override for ``config/model.yaml``.
        max_train_batches: Optional cap on training batches per epoch.
        max_val_batches: Optional cap on validation batches per epoch.

    Returns:
        Training summary dictionary from ``Trainer.fit()``.
    """
    training_config = load_training_config(training_config_path)
    model_config = load_model_config(model_config_path)
    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(training_config.get("device", "auto")))
    logger.info("Using device: %s", device)

    train_loader = build_dataloader(
        split="train",
        training_config_path=training_config_path,
    )
    val_loader = build_dataloader(
        split="val",
        training_config_path=training_config_path,
    )

    model = build_model(model_config)
    experiment_config = _build_experiment_config(training_config, model_config)
    experiment_logger = _build_experiment_logger(training_config, experiment_config)

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

    if bool(training_config.get("resume", True)) and trainer.last_checkpoint_path.exists():
        trainer.load_checkpoint(trainer.last_checkpoint_path)

    return trainer.fit()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train BrainHemorrhageAI baseline model.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to model YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    _configure_logging()
    args = parse_args()
    summary = run_training(
        training_config_path=args.training_config,
        model_config_path=args.model_config,
    )
    logger.info("Training complete: %s", summary)


if __name__ == "__main__":
    main()
