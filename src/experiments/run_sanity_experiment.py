"""Phase 5.0 — Baseline sanity experiment orchestration."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset.dataloader import build_dataloader
from evaluation.checkpoint import CHECKPOINT_SELECTION_METRIC
from evaluation.evaluate_model import evaluate_model, load_evaluation_config
from evaluation.experiment_logger import ExperimentLogger, collect_environment_info
from evaluation.metrics import NUM_CLASSES
from experiments.experiment_tracker import (
    append_experiment_index,
    assemble_experiment_folder,
    load_experiment_config,
    write_experiment_report,
)
from experiments.preflight import (
    IMAGES_DIR,
    MASKS_DIR,
    PreflightError,
    PreflightResult,
    print_preflight_report,
    run_preflight_checks,
)
from experiments.run_baseline_experiment import (
    _build_training_snapshot,
    _training_summary_from_run,
)
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
DEFAULT_SANITY_CONFIG = PROJECT_ROOT / "config" / "experiments" / "experiment_sanity.yaml"


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
        "validation_interval",
        "save_every",
    )
    for key in override_keys:
        if key in experiment_config and experiment_config[key] is not None:
            merged[key] = experiment_config[key]
    return merged


def _memory_summary(device: torch.device) -> str:
    """Return a concise memory usage string."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        return f"CUDA {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / (1024**3)
        return f"CPU RSS {rss:.2f} GB"
    except ImportError:
        return "CPU memory unavailable (install psutil for details)"


def _tensor_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise PreflightError when a tensor contains non-finite values."""
    if not torch.isfinite(tensor).all():
        raise PreflightError(f"Non-finite values detected in {name}.")


def run_extended_preflight() -> dict[str, str]:
    """Run standard preflight plus sanity-specific checks."""
    result = run_preflight_checks()
    checks = dict(result.checks)

    image_names = {path.name for path in IMAGES_DIR.glob("*.nii.gz")}
    mask_names = {path.name for path in MASKS_DIR.glob("*.nii.gz")}
    if image_names != mask_names:
        unmatched_images = sorted(image_names - mask_names)
        unmatched_masks = sorted(mask_names - image_names)
        raise PreflightError(
            f"Image-mask pairing failed: {len(unmatched_images)} images without masks, "
            f"{len(unmatched_masks)} masks without images."
        )
    checks["image_mask_pairing"] = f"OK ({len(image_names)} paired scans)"

    try:
        version = importlib.metadata.version("brain-hemorrhage-ai")
        checks["editable_installation"] = f"OK (brain-hemorrhage-ai {version})"
    except importlib.metadata.PackageNotFoundError as error:
        raise PreflightError(
            "Editable installation not found. Run: pip install -e ."
        ) from error

    train_loader = build_dataloader("train")
    batch = next(iter(train_loader))
    if batch["image"].dtype != torch.float32:
        raise PreflightError(f"Expected float32 images, got {batch['image'].dtype}.")
    if batch["label"].dtype != torch.long:
        raise PreflightError(f"Expected long labels, got {batch['label'].dtype}.")
    checks["dataloader"] = f"OK (batch image {tuple(batch['image'].shape)})"

    return checks


class SanityTrainer(Trainer):
    """Trainer with per-epoch sanity monitoring and progress reporting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.warnings: list[str] = []
        self.critical_issues: list[str] = []
        self.epoch_records: list[dict[str, Any]] = []
        self._epoch_weight_snapshot: dict[str, torch.Tensor] | None = None

    def _verify_gradients_one_batch(self) -> None:
        """Confirm backward pass produces finite gradients on one training batch."""
        self.model.train()
        batch = next(iter(self.train_loader))
        images, labels = self._prepare_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._forward_loss(images, labels)
        if not math.isfinite(float(loss.item())):
            raise PreflightError("Training loss is not finite on diagnostic batch.")
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                _tensor_finite(f"gradients ({name})", parameter.grad)

    def _capture_weight_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

    def _weights_changed(self, before: dict[str, torch.Tensor]) -> bool:
        for name, parameter in self.model.named_parameters():
            if not torch.equal(before[name].cpu(), parameter.detach().cpu()):
                return True
        return False

    def _optimizer_has_state(self) -> bool:
        return any(self.optimizer.state.get(parameter) for parameter in self.model.parameters())

    def _validate_prediction_classes(self) -> tuple[list[int], bool]:
        """Inspect one validation batch for valid class IDs and background-only predictions."""
        self.model.eval()
        batch = next(iter(self.val_loader))
        images, labels = self._prepare_batch(batch)
        with torch.no_grad():
            logits = self.model(images)
        predictions = logits.argmax(dim=1)
        unique = sorted(int(value) for value in torch.unique(predictions).tolist())
        invalid = [value for value in unique if value < 0 or value >= NUM_CLASSES]
        if invalid:
            raise PreflightError(f"Invalid predicted class IDs: {invalid}")
        background_only = set(unique) <= {0}
        return unique, background_only

    def _run_epoch_sanity_checks(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, Any] | None,
        weight_before: dict[str, torch.Tensor],
        checkpoint_saved: bool,
    ) -> None:
        """Verify training health after each epoch."""
        if not math.isfinite(train_loss):
            self.critical_issues.append(f"Epoch {epoch}: training loss is not finite.")

        if val_metrics is not None:
            val_loss = float(val_metrics["val_loss"])
            val_dice = float(val_metrics[CHECKPOINT_SELECTION_METRIC])
            val_iou = float(val_metrics.get("val_macro_iou", 0.0))
            if not math.isfinite(val_loss):
                self.critical_issues.append(f"Epoch {epoch}: validation loss is not finite.")
            if not math.isfinite(val_dice):
                self.critical_issues.append(f"Epoch {epoch}: validation Dice is not finite.")

            if len(self.history["train_loss"]) >= 2:
                previous = self.history["train_loss"][-2]
                if train_loss > previous * 2.0 and train_loss > 1.0:
                    self.warnings.append(
                        f"Epoch {epoch}: training loss increased sharply "
                        f"({previous:.4f} -> {train_loss:.4f})."
                    )

            if val_dice == 0.0:
                self.warnings.append(f"Epoch {epoch}: validation macro Dice is exactly zero.")

            _, background_only = self._validate_prediction_classes()
            if background_only:
                self.warnings.append(f"Epoch {epoch}: predictions contain background only.")

        if not self._weights_changed(weight_before):
            self.critical_issues.append(f"Epoch {epoch}: model weights did not change.")
        elif not self._optimizer_has_state():
            self.warnings.append(f"Epoch {epoch}: optimizer state not populated after weight update.")

        if val_metrics is not None and not checkpoint_saved:
            self.warnings.append(f"Epoch {epoch}: checkpoint was not written.")

    def _print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, Any] | None,
        learning_rate: float,
        epoch_duration: float,
        eta_seconds: float,
    ) -> None:
        """Print concise epoch progress."""
        memory = _memory_summary(self.device)
        if val_metrics is None:
            print(
                f"[Epoch {epoch}/{self.epochs}] "
                f"train_loss={train_loss:.6f} | lr={learning_rate:.2e} | "
                f"duration={epoch_duration:.1f}s | eta={eta_seconds/60:.1f}min | {memory}"
            )
            return

        print(
            f"[Epoch {epoch}/{self.epochs}] "
            f"train_loss={train_loss:.6f} | val_loss={val_metrics['val_loss']:.6f} | "
            f"val_dice={val_metrics[CHECKPOINT_SELECTION_METRIC]:.6f} | "
            f"val_iou={val_metrics.get('val_macro_iou', 0.0):.6f} | "
            f"lr={learning_rate:.2e} | duration={epoch_duration:.1f}s | "
            f"eta={eta_seconds/60:.1f}min | {memory}"
        )

    def fit(self) -> dict[str, Any]:
        """Run training with sanity monitoring."""
        self._verify_gradients_one_batch()
        stop_training = False
        epoch_durations: list[float] = []

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.perf_counter()
            weight_before = self._capture_weight_snapshot()
            train_loss = self.train_one_epoch(epoch)
            _tensor_finite("training loss", torch.tensor(train_loss))

            val_metrics: dict[str, Any] | None = None
            checkpoint_saved = False

            if epoch % self.validation_interval == 0:
                val_metrics = self.validate_one_epoch(epoch)
                if self._update_early_stopping(val_metrics):
                    logger.info("Early stopping triggered at epoch %s.", epoch)
                    stop_training = True

            epoch_duration = time.perf_counter() - epoch_start
            epoch_durations.append(epoch_duration)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.history["learning_rate"].append(learning_rate)
            self.history["epoch_duration_sec"].append(epoch_duration)

            remaining_epochs = self.epochs - epoch
            avg_duration = sum(epoch_durations) / len(epoch_durations)
            eta_seconds = remaining_epochs * avg_duration

            record_extras = {
                "learning_rate": learning_rate,
                "epoch_duration_sec": epoch_duration,
                "eta_seconds": eta_seconds,
            }
            self._log_epoch(epoch, train_loss, val_metrics, record_extras)

            if val_metrics is not None and epoch % self.save_every == 0:
                self.save_checkpoint(epoch, val_metrics)
                checkpoint_saved = (
                    self.best_checkpoint_path.exists() and self.last_checkpoint_path.exists()
                )

            self._run_epoch_sanity_checks(
                epoch, train_loss, val_metrics, weight_before, checkpoint_saved
            )
            self._print_epoch_summary(
                epoch, train_loss, val_metrics, learning_rate, epoch_duration, eta_seconds
            )

            self.epoch_records.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": None if val_metrics is None else val_metrics["val_loss"],
                    "val_macro_dice": None if val_metrics is None else val_metrics[CHECKPOINT_SELECTION_METRIC],
                    "val_macro_iou": None if val_metrics is None else val_metrics.get("val_macro_iou"),
                    "learning_rate": learning_rate,
                    "epoch_duration_sec": epoch_duration,
                    "checkpoint_saved": checkpoint_saved,
                }
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if stop_training:
                break

        if self.critical_issues:
            raise PreflightError(
                "Sanity checks failed:\n" + "\n".join(f"  - {issue}" for issue in self.critical_issues)
            )

        self._save_training_plots()
        summary = {
            CHECKPOINT_SELECTION_METRIC: self.best_val_macro_dice,
            "epochs_completed": len(self.history["train_loss"]),
            "best_checkpoint": str(self.best_checkpoint_path),
            "last_checkpoint": str(self.last_checkpoint_path),
            "warnings": list(self.warnings),
        }
        self.experiment_logger.finalize(summary_metrics=summary)
        return summary


def _loss_trend(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient data"
    delta = values[-1] - values[0]
    direction = "decreasing" if delta < 0 else "increasing" if delta > 0 else "flat"
    return f"{direction} ({values[0]:.6f} -> {values[-1]:.6f})"


def _write_training_analysis(
    experiment_dir: Path,
    trainer: SanityTrainer,
    training_time_sec: float,
    health_status: str,
    issues: list[dict[str, str]],
    recommendations: list[str],
) -> Path:
    """Write post-training analysis report."""
    best_epoch = 1
    if trainer.history["val_macro_dice"]:
        best_epoch = int(
            trainer.history["val_macro_dice"].index(max(trainer.history["val_macro_dice"])) + trainer.start_epoch
        )
    best_iou = max(trainer.history["val_macro_iou"]) if trainer.history["val_macro_iou"] else 0.0
    avg_epoch = (
        sum(trainer.history["epoch_duration_sec"]) / len(trainer.history["epoch_duration_sec"])
        if trainer.history["epoch_duration_sec"]
        else 0.0
    )

    lines = [
        "# Training Analysis — experiment_sanity",
        "",
        "## Summary",
        "",
        "- Training completed successfully",
        f"- Total training time: {training_time_sec/60:.1f} minutes ({training_time_sec:.1f} s)",
        f"- Average epoch time: {avg_epoch:.1f} s",
        f"- Best validation Dice: {trainer.best_val_macro_dice:.6f}",
        f"- Best validation IoU: {best_iou:.6f}",
        f"- Best epoch: {best_epoch}",
        "",
        "## Loss trends",
        "",
        f"- Training loss: {_loss_trend(trainer.history['train_loss'])}",
        f"- Validation loss: {_loss_trend(trainer.history['val_loss'])}",
        "",
        "## Warnings",
        "",
    ]
    if trainer.warnings:
        lines.extend(f"- {warning}" for warning in trainer.warnings)
    else:
        lines.append("- None")

    lines.extend(["", "## Pipeline health", "", f"**{health_status}**", ""])
    if issues:
        lines.extend(["", "## Issues detected", ""])
        for issue in issues:
            lines.append(f"### {issue['title']}")
            lines.append(f"- Likely cause: {issue['cause']}")
            lines.append(f"- Recommended fix: {issue['fix']}")
            lines.append("")

    lines.extend(["", "## Recommendations before long training", ""])
    lines.extend(f"- {item}" for item in recommendations)

    report_path = experiment_dir / "training_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _assess_health(
    trainer: SanityTrainer,
    eval_results: dict[str, Any],
    best_checkpoint: Path,
) -> tuple[str, list[dict[str, str]], list[str]]:
    """Determine final health status."""
    issues: list[dict[str, str]] = []
    recommendations = [
        "Review validation curves in reports/experiments/experiment_sanity/plots/.",
        "Confirm test macro Dice and volume metrics before launching Experiment 001 (150 epochs).",
        "Use GPU (Kaggle T4) for full training if CPU runtime is prohibitive.",
    ]

    if not best_checkpoint.exists():
        issues.append(
            {
                "title": "Best checkpoint missing",
                "cause": "Checkpoint save failed or validation did not run.",
                "fix": "Inspect trainer logs and checkpoint directory permissions.",
            }
        )

    test_dice = float(eval_results.get("global_metrics", {}).get("macro_dice", 0.0))
    if trainer.best_val_macro_dice <= 0.0:
        issues.append(
            {
                "title": "Validation macro Dice remained zero",
                "cause": "Model may not be learning hemorrhage classes yet, or dataset batch imbalance.",
                "fix": "Run additional sanity epochs or inspect predictions on validation slices.",
            }
        )
        recommendations.insert(0, "Investigate zero validation Dice before long training.")

    if trainer.warnings:
        recommendations.insert(
            0,
            f"Review {len(trainer.warnings)} training warning(s) listed in training_analysis.md.",
        )

    if not math.isfinite(test_dice):
        issues.append(
            {
                "title": "Test evaluation returned non-finite Dice",
                "cause": "Numerical instability or evaluation pipeline error.",
                "fix": "Re-run evaluation and inspect metrics.json.",
            }
        )

    status = "STATUS: READY FOR LONG TRAINING" if not issues else "STATUS: ISSUES DETECTED"
    return status, issues, recommendations


def run_sanity_experiment(
    experiment_config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the Phase 5.0 sanity experiment end-to-end."""
    extended_checks = run_extended_preflight()
    preflight = run_preflight_checks()
    merged_checks = {**preflight.checks, **extended_checks}
    print_preflight_report(
        PreflightResult(
            passed=True,
            checks=merged_checks,
            device=preflight.device,
            cuda_available=preflight.cuda_available,
            seed=preflight.seed,
        )
    )

    experiment_config = load_experiment_config(experiment_config_path or DEFAULT_SANITY_CONFIG)
    experiment_id = str(experiment_config["experiment_id"])

    base_training = load_training_config(PROJECT_ROOT / experiment_config["training_config"])
    training_config = _merge_training_config(base_training, experiment_config)
    model_config = load_model_config(PROJECT_ROOT / experiment_config["model_config"])

    seed = int(training_config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(training_config.get("device", "auto")))

    checkpoint_dir = PROJECT_ROOT / str(training_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / str(training_config.get("best_checkpoint_name", "best_model.pt"))

    temp_training_config = checkpoint_dir / "training_sanity.yaml"
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

    trainer = SanityTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        model_config=model_config,
        experiment_logger=experiment_logger,
        device=device,
    )

    training_start = time.perf_counter()
    trainer.fit()
    training_time_sec = time.perf_counter() - training_start
    training_run_dir = experiment_logger.run_dir

    eval_config = load_evaluation_config(PROJECT_ROOT / experiment_config["evaluation_config"])
    eval_config = dict(eval_config)
    eval_config["checkpoint_path"] = str(best_checkpoint)
    eval_config["run_id"] = str(experiment_config.get("evaluation_run_id", f"{experiment_id}_eval"))
    eval_config["reports_dir"] = str(experiment_config.get("evaluation_reports_dir", "reports/evaluation"))

    temp_eval_config = checkpoint_dir / "evaluation_sanity.yaml"
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

    training_summary = _training_summary_from_run(trainer, training_config)
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

    health_status, issues, recommendations = _assess_health(trainer, eval_results, best_checkpoint)
    analysis_path = _write_training_analysis(
        experiment_dir,
        trainer,
        training_time_sec,
        health_status,
        issues,
        recommendations,
    )

    test_dice = eval_results.get("global_metrics", {}).get("macro_dice")
    append_experiment_index(
        {
            "experiment_id": experiment_id,
            "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "model": model_config.get("model_name", "monai_2d_unet"),
            "epochs": training_summary.get("epochs_completed", experiment_config.get("epochs")),
            "best_val_dice": training_summary.get("val_macro_dice"),
            "test_dice": test_dice,
            "checkpoint": str(best_checkpoint),
            "status": "completed" if health_status == "STATUS: READY FOR LONG TRAINING" else "issues_detected",
            "notes": experiment_config.get("description", ""),
        }
    )

    print()
    print(health_status)
    if issues:
        for issue in issues:
            print(f"  - {issue['title']}: {issue['cause']} -> {issue['fix']}")
    print(f"Training analysis: {analysis_path}")

    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "best_checkpoint": str(best_checkpoint),
        "training_time_sec": training_time_sec,
        "best_val_dice": training_summary.get("val_macro_dice"),
        "test_dice": test_dice,
        "health_status": health_status,
        "warnings": trainer.warnings,
        "training_analysis": str(analysis_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BrainHemorrhageAI sanity experiment (Phase 5.0).")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SANITY_CONFIG,
        help="Path to sanity experiment YAML.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    try:
        summary = run_sanity_experiment(experiment_config_path=args.config)
    except PreflightError as error:
        logger.error("Sanity experiment aborted: %s", error)
        print(f"STATUS: ISSUES DETECTED — {error}")
        raise SystemExit(1) from error

    logger.info("Sanity experiment complete: %s", summary["experiment_id"])
    logger.info("Best val Dice: %s", summary["best_val_dice"])
    logger.info("Test Dice: %s", summary["test_dice"])
    logger.info("%s", summary["health_status"])


if __name__ == "__main__":
    main()
