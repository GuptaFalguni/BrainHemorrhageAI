"""Smoke test for the baseline training pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import yaml

from dataset.dataloader import build_dataloader
from evaluation.checkpoint import CHECKPOINT_SELECTION_METRIC, load_checkpoint
from evaluation.experiment_logger import ExperimentLogger
from models.monai_unet import build_model
from training.config_loader import load_model_config, load_training_config
from training.losses import build_loss
from training.optimizer import build_optimizer
from training.seed import set_seed
from training.trainer import Trainer


def _verify_single_batch_loss_decreases(device: torch.device) -> None:
    """Verify backward pass and optimizer step reduce loss on a fixed batch."""
    model_config = load_model_config()
    training_config = load_training_config()
    set_seed(int(training_config.get("seed", 42)))

    model = build_model(model_config).to(device)
    criterion = build_loss(training_config).to(device)
    optimizer = build_optimizer(model, training_config)

    images = torch.randn(2, 3, 64, 64, device=device)
    labels = torch.randint(0, 6, (2, 1, 64, 64), device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    first_loss = float(criterion(model(images), labels).item())
    first_loss_tensor = criterion(model(images), labels)
    first_loss_tensor.backward()
    optimizer.step()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    second_loss = float(criterion(model(images), labels).item())

    if second_loss >= first_loss:
        raise AssertionError(
            f"Expected loss to decrease on repeated batch, got {first_loss} -> {second_loss}."
        )


def _write_smoke_config(base_config: dict, output_path: Path, checkpoint_dir: Path, reports_dir: Path) -> None:
    """Write a temporary one-epoch smoke-test training config."""
    config = dict(base_config)
    config["epochs"] = 1
    config["early_stopping_patience"] = 1
    config["validation_interval"] = 1
    config["save_every"] = 1
    config["resume"] = False
    config["run_id"] = "smoke_test"
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["reports_dir"] = str(reports_dir)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def run_smoke_test() -> None:
    """Run an end-to-end one-epoch smoke test on real dataloaders."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _verify_single_batch_loss_decreases(device)

    base_training_config = load_training_config()
    model_config = load_model_config()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint_dir = temp_root / "checkpoints"
        reports_dir = temp_root / "reports" / "training"
        training_config_path = temp_root / "training.yaml"
        _write_smoke_config(
            base_training_config,
            training_config_path,
            checkpoint_dir=checkpoint_dir,
            reports_dir=reports_dir,
        )

        set_seed(int(base_training_config.get("seed", 42)))
        train_loader = build_dataloader("train", training_config_path=training_config_path)
        val_loader = build_dataloader("val", training_config_path=training_config_path)
        model = build_model(model_config)

        experiment_logger = ExperimentLogger(
            run_id="smoke_test",
            reports_dir=reports_dir,
            config={"training": base_training_config, "model": model_config},
            seed=int(base_training_config.get("seed", 42)),
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=load_training_config(training_config_path),
            model_config=model_config,
            experiment_logger=experiment_logger,
            device=device,
            max_train_batches=2,
            max_val_batches=2,
        )

        summary = trainer.fit()

        if trainer.first_train_batch_loss is None or trainer.last_train_batch_loss is None:
            raise AssertionError("Training loop did not record batch losses.")

        if not trainer.last_checkpoint_path.exists():
            raise AssertionError("Last checkpoint was not saved.")
        if not trainer.best_checkpoint_path.exists():
            raise AssertionError("Best checkpoint was not saved.")

        payload = load_checkpoint(trainer.best_checkpoint_path, map_location="cpu")
        if CHECKPOINT_SELECTION_METRIC not in payload["val_metrics"]:
            raise AssertionError("Checkpoint missing validation macro Dice.")

        run_dir = reports_dir / "smoke_test"
        required_artifacts = [
            run_dir / "config.yaml",
            run_dir / "environment.json",
            run_dir / "metrics.json",
            run_dir / "epoch_metrics.jsonl",
            run_dir / "plots" / "loss_curve.png",
        ]
        missing = [path for path in required_artifacts if not path.exists()]
        if missing:
            raise AssertionError(f"Missing logger artifacts: {missing}")

        metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if CHECKPOINT_SELECTION_METRIC not in metrics_payload["summary"]:
            raise AssertionError("Logger summary missing macro Dice.")

        if summary["epochs_completed"] != 1:
            raise AssertionError("Smoke test should complete exactly one epoch.")

        if trainer.last_train_batch_loss > trainer.first_train_batch_loss * 1.5:
            raise AssertionError(
                "Training loss increased substantially during smoke-test epoch."
            )


def main() -> None:
    """Run smoke test and print validation summary."""
    run_smoke_test()
    print("TRAINING PIPELINE VALIDATION PASSED")


if __name__ == "__main__":
    main()
