"""Baseline training loop for BrainHemorrhageAI."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from monai.data import DataLoader
from torch.amp import GradScaler, autocast

from evaluation.checkpoint import (
    CHECKPOINT_SELECTION_METRIC,
    load_checkpoint,
    save_best_checkpoint,
    save_last_checkpoint,
    should_save_best_checkpoint,
)
from evaluation.experiment_logger import ExperimentLogger
from evaluation.metrics import HEMORRHAGE_CLASSES, NUM_CLASSES, binary_confusion, macro_dice
from training.config_loader import PROJECT_ROOT, should_use_amp
from training.losses import build_loss
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


@dataclass
class ValidationAccumulator:
    """Aggregate per-class confusion counts across validation batches."""

    num_classes: int = NUM_CLASSES
    tp: torch.Tensor = field(default_factory=lambda: torch.zeros(NUM_CLASSES, dtype=torch.float64))
    fp: torch.Tensor = field(default_factory=lambda: torch.zeros(NUM_CLASSES, dtype=torch.float64))
    fn: torch.Tensor = field(default_factory=lambda: torch.zeros(NUM_CLASSES, dtype=torch.float64))

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Update counts from one batch of integer label maps."""
        for class_index in range(self.num_classes):
            confusion = binary_confusion(
                prediction == class_index,
                target == class_index,
            )
            self.tp[class_index] += confusion.tp.cpu()
            self.fp[class_index] += confusion.fp.cpu()
            self.fn[class_index] += confusion.fn.cpu()

    def per_class_dice(self, smooth: float = 1e-7) -> torch.Tensor:
        """Compute per-class Dice from accumulated counts."""
        numerator = 2.0 * self.tp
        denominator = 2.0 * self.tp + self.fp + self.fn + smooth
        return (numerator / denominator).to(dtype=torch.float32)

    def macro_dice(self, ignore_background: bool = True) -> float:
        """Compute macro Dice over hemorrhage classes."""
        dice = self.per_class_dice()
        if ignore_background:
            dice = dice[list(HEMORRHAGE_CLASSES)]
        return float(dice.mean().item())

    def per_class_iou(self, smooth: float = 1e-7) -> torch.Tensor:
        """Compute per-class IoU from accumulated counts."""
        union = self.tp + self.fp + self.fn + smooth
        return (self.tp / union).to(dtype=torch.float32)

    def macro_iou(self, ignore_background: bool = True) -> float:
        """Compute macro IoU over hemorrhage classes."""
        iou = self.per_class_iou()
        if ignore_background:
            iou = iou[list(HEMORRHAGE_CLASSES)]
        return float(iou.mean().item())


class Trainer:
    """Orchestrates baseline MONAI U-Net training for BHSD."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: dict[str, Any],
        model_config: dict[str, Any],
        experiment_logger: ExperimentLogger,
        device: torch.device | None = None,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_config = training_config
        self.model_config = model_config
        self.experiment_logger = experiment_logger
        self.device = device or torch.device("cpu")
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches

        self.criterion = build_loss(training_config)
        self.optimizer = build_optimizer(model, training_config)
        self.scheduler = build_scheduler(self.optimizer, training_config)
        self.use_amp = should_use_amp(training_config, self.device)
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        self.gradient_clip = float(training_config.get("gradient_clip", 0.0))

        checkpoint_dir = PROJECT_ROOT / str(training_config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = checkpoint_dir / str(
            training_config.get("best_checkpoint_name", "best_model.pt")
        )
        self.last_checkpoint_path = checkpoint_dir / str(
            training_config.get("last_checkpoint_name", "last_model.pt")
        )

        self.start_epoch = 1
        self.epochs = int(training_config.get("epochs", 1))
        self.validation_interval = int(training_config.get("validation_interval", 1))
        self.save_every = int(training_config.get("save_every", 1))
        self.early_stopping_patience = int(training_config.get("early_stopping_patience", 20))
        self.best_val_macro_dice = float("-inf")
        self.epochs_without_improvement = 0

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_dice": [],
            "val_macro_iou": [],
            "learning_rate": [],
            "epoch_duration_sec": [],
        }

        self.model.to(self.device)
        self.criterion.to(self.device)

    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move image and label tensors to the training device."""
        images = batch["image"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        return images, labels

    def _forward_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and compute loss."""
        logits = self.model(images)
        return self.criterion(logits, labels)

    def train_one_epoch(self, epoch: int) -> float:
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        self.first_train_batch_loss = None
        self.last_train_batch_loss = None

        for batch_index, batch in enumerate(self.train_loader, start=1):
            images, labels = self._prepare_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = self._forward_loss(images, labels)

            batch_loss = float(loss.item())
            if self.first_train_batch_loss is None:
                self.first_train_batch_loss = batch_loss
            self.last_train_batch_loss = batch_loss

            self.scaler.scale(loss).backward()

            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.item())
            batch_count += 1

            if self.max_train_batches is not None and batch_index >= self.max_train_batches:
                break

        epoch_loss = running_loss / max(batch_count, 1)
        logger.info("Epoch %s train loss: %.6f", epoch, epoch_loss)
        return epoch_loss

    def validate_one_epoch(self, epoch: int) -> dict[str, Any]:
        """Validate the model for one epoch."""
        self.model.eval()
        running_loss = 0.0
        batch_count = 0
        accumulator = ValidationAccumulator()

        with torch.no_grad():
            for batch_index, batch in enumerate(self.val_loader, start=1):
                images, labels = self._prepare_batch(batch)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                running_loss += float(loss.item())
                batch_count += 1

                predictions = logits.argmax(dim=1)
                targets = labels.squeeze(1)
                accumulator.update(predictions.detach(), targets.detach())

                if self.max_val_batches is not None and batch_index >= self.max_val_batches:
                    break

        per_class_dice = accumulator.per_class_dice()
        val_macro_dice = accumulator.macro_dice(ignore_background=True)
        val_macro_iou = accumulator.macro_iou(ignore_background=True)
        val_loss = running_loss / max(batch_count, 1)

        metrics = {
            "val_loss": val_loss,
            CHECKPOINT_SELECTION_METRIC: val_macro_dice,
            "val_macro_iou": val_macro_iou,
            "per_class_dice": {
                str(class_index): float(per_class_dice[class_index].item())
                for class_index in range(NUM_CLASSES)
            },
        }
        logger.info(
            "Epoch %s val loss: %.6f, val macro Dice: %.6f, val macro IoU: %.6f",
            epoch,
            val_loss,
            val_macro_dice,
            val_macro_iou,
        )
        return metrics

    def save_checkpoint(self, epoch: int, val_metrics: dict[str, Any]) -> None:
        """Persist last and best checkpoints using evaluation helpers."""
        decision = should_save_best_checkpoint(
            val_metrics[CHECKPOINT_SELECTION_METRIC],
            self.best_val_macro_dice,
        )
        if decision.is_best:
            self.best_val_macro_dice = decision.current_score
            logger.info(
                "New best checkpoint at epoch %s (macro Dice=%.6f).",
                epoch,
                decision.current_score,
            )

        metadata = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "run_id": self.experiment_logger.run_id,
            "best_val_macro_dice": self.best_val_macro_dice,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        if scheduler_state is not None:
            metadata["scheduler_state_dict"] = scheduler_state

        save_last_checkpoint(
            self.last_checkpoint_path,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            epoch=epoch,
            val_metrics=val_metrics,
            metadata=metadata,
        )

        save_best_checkpoint(
            self.best_checkpoint_path,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            epoch=epoch,
            val_metrics=val_metrics,
            metadata=metadata,
            current_best_score=decision.previous_best,
        )

    def load_checkpoint(self, checkpoint_path: Path | str | None = None) -> None:
        """Restore model, optimizer, scheduler, and training progress."""
        path = Path(checkpoint_path or self.last_checkpoint_path)
        if not path.exists():
            logger.info("No checkpoint found at %s; starting fresh.", path)
            return

        payload = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        if payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        metadata = payload.get("metadata", {})
        scheduler_state = metadata.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.start_epoch = int(payload["epoch"]) + 1
        val_metrics = payload.get("val_metrics", {})
        if "best_val_macro_dice" in metadata:
            self.best_val_macro_dice = float(metadata["best_val_macro_dice"])
        else:
            self.best_val_macro_dice = float(
                val_metrics.get(CHECKPOINT_SELECTION_METRIC, float("-inf"))
            )
        self.epochs_without_improvement = int(metadata.get("epochs_without_improvement", 0))
        logger.info(
            "Resumed from %s at epoch %s (best macro Dice=%.6f, patience=%s).",
            path,
            self.start_epoch,
            self.best_val_macro_dice,
            self.epochs_without_improvement,
        )

    def _update_early_stopping(self, val_metrics: dict[str, Any]) -> bool:
        """Update early stopping state. Returns True when training should stop."""
        current = float(val_metrics[CHECKPOINT_SELECTION_METRIC])
        if current > self.best_val_macro_dice + 1e-6:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.early_stopping_patience

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, Any] | None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Append metrics to logger and history."""
        self.history["train_loss"].append(train_loss)
        record: dict[str, Any] = {"train_loss": train_loss}
        if extras:
            record.update(extras)
        if val_metrics is not None:
            record.update(val_metrics)
            self.history["val_loss"].append(float(val_metrics["val_loss"]))
            self.history["val_macro_dice"].append(float(val_metrics[CHECKPOINT_SELECTION_METRIC]))
            self.history["val_macro_iou"].append(float(val_metrics.get("val_macro_iou", 0.0)))
        self.experiment_logger.log_epoch(epoch, record)

    def _save_training_plots(self) -> None:
        """Generate training curve plots from accumulated history."""
        if not self.history["train_loss"]:
            return
        epochs = list(range(self.start_epoch, self.start_epoch + len(self.history["train_loss"])))
        val_loss = self.history["val_loss"] if self.history["val_loss"] else None
        val_macro_dice = self.history["val_macro_dice"] if self.history["val_macro_dice"] else None
        learning_rates = self.history["learning_rate"] if self.history["learning_rate"] else None
        self.experiment_logger.save_training_curves(
            epochs=epochs,
            train_loss=self.history["train_loss"],
            val_loss=val_loss,
            val_macro_dice=val_macro_dice,
            learning_rates=learning_rates,
        )
        self.experiment_logger.save_validation_metrics_csv(self.history, epochs)

    def fit(self) -> dict[str, Any]:
        """Run the full training loop with validation and checkpointing."""
        stop_training = False
        final_metrics: dict[str, Any] = {}
        epoch_durations: list[float] = []

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self.train_one_epoch(epoch)
            val_metrics: dict[str, Any] | None = None

            if epoch % self.validation_interval == 0:
                val_metrics = self.validate_one_epoch(epoch)
                final_metrics = val_metrics
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
            logger.info(
                "Epoch %s duration: %.1fs | LR: %.2e | ETA: %.1fs (%.1f min)",
                epoch,
                epoch_duration,
                learning_rate,
                eta_seconds,
                eta_seconds / 60.0,
            )

            record_extras = {
                "learning_rate": learning_rate,
                "epoch_duration_sec": epoch_duration,
                "eta_seconds": eta_seconds,
            }
            self._log_epoch(epoch, train_loss, val_metrics, record_extras)

            if val_metrics is not None and epoch % self.save_every == 0:
                self.save_checkpoint(epoch, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            if stop_training:
                break

        self._save_training_plots()
        summary = {
            CHECKPOINT_SELECTION_METRIC: self.best_val_macro_dice,
            "epochs_completed": len(self.history["train_loss"]),
            "best_checkpoint": str(self.best_checkpoint_path),
            "last_checkpoint": str(self.last_checkpoint_path),
        }
        self.experiment_logger.finalize(summary_metrics=summary)
        return summary
