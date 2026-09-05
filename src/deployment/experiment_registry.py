"""Experiment registry — trained-run source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_REGISTRY = PROJECT_ROOT / "config" / "experiments" / "registry.yaml"

REQUIRED_EXPERIMENT_FIELDS: frozenset[str] = frozenset(
    {
        "experiment_id",
        "model_type",
        "checkpoint_path",
        "config_path",
        "dataset",
        "epochs",
        "framework",
        "metrics",
        "publication_status",
    }
)


@dataclass(frozen=True)
class ExperimentRecord:
    """One trained experiment entry from the experiment registry."""

    experiment_id: str
    model_type: str
    checkpoint_path: Path
    config_path: Path
    dataset: str
    epochs: int
    framework: str
    metrics: Mapping[str, Any]
    publication_status: str

    @property
    def checkpoint_exists(self) -> bool:
        return self.checkpoint_path.is_file()

    @property
    def config_exists(self) -> bool:
        return self.config_path.is_file()

    def resolve_metric(self, key: str) -> Any:
        """Return a metric value by key, or raise ``KeyError``."""
        if key not in self.metrics:
            raise KeyError(f"Metric '{key}' not found for experiment '{self.experiment_id}'.")
        return self.metrics[key]


@dataclass
class ExperimentRegistry:
    """Load and validate ``config/experiments/registry.yaml``."""

    records: dict[str, ExperimentRecord] = field(default_factory=dict)
    registry_path: Path = DEFAULT_EXPERIMENT_REGISTRY
    project_root: Path = PROJECT_ROOT

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        *,
        project_root: Path | str | None = None,
    ) -> ExperimentRegistry:
        root = Path(project_root) if project_root else PROJECT_ROOT
        registry_path = Path(path) if path else root / "config" / "experiments" / "registry.yaml"
        if not registry_path.is_file():
            raise FileNotFoundError(f"Experiment registry not found: {registry_path}")

        with registry_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        if not isinstance(payload, dict) or "experiments" not in payload:
            raise ValueError(f"Experiment registry must contain an 'experiments' list: {registry_path}")

        experiments = payload["experiments"]
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("Experiment registry 'experiments' must be a non-empty list.")

        records: dict[str, ExperimentRecord] = {}
        for index, entry in enumerate(experiments):
            if not isinstance(entry, dict):
                raise ValueError(f"Experiment entry at index {index} must be a mapping.")
            missing = REQUIRED_EXPERIMENT_FIELDS - set(entry)
            if missing:
                raise ValueError(
                    f"Experiment entry at index {index} missing fields: {sorted(missing)}"
                )

            experiment_id = str(entry["experiment_id"]).strip()
            if not experiment_id:
                raise ValueError(f"Experiment entry at index {index} has empty experiment_id.")
            if experiment_id in records:
                raise ValueError(f"Duplicate experiment_id: {experiment_id}")

            metrics = entry["metrics"]
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError(f"Experiment '{experiment_id}' metrics must be a non-empty mapping.")

            epochs = entry["epochs"]
            if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
                raise ValueError(f"Experiment '{experiment_id}' epochs must be a positive integer.")

            checkpoint_path = Path(str(entry["checkpoint_path"]))
            config_path = Path(str(entry["config_path"]))
            if not checkpoint_path.is_absolute():
                checkpoint_path = root / checkpoint_path
            if not config_path.is_absolute():
                config_path = root / config_path

            records[experiment_id] = ExperimentRecord(
                experiment_id=experiment_id,
                model_type=str(entry["model_type"]),
                checkpoint_path=checkpoint_path.resolve(),
                config_path=config_path.resolve(),
                dataset=str(entry["dataset"]),
                epochs=int(epochs),
                framework=str(entry["framework"]),
                metrics=dict(metrics),
                publication_status=str(entry["publication_status"]),
            )

        return cls(records=records, registry_path=registry_path.resolve(), project_root=root.resolve())

    def get(self, experiment_id: str) -> ExperimentRecord:
        try:
            return self.records[experiment_id]
        except KeyError as exc:
            raise KeyError(f"Unknown experiment_id: {experiment_id}") from exc

    def list_ids(self) -> list[str]:
        return sorted(self.records)

    def resolve_checkpoint(self, experiment_id: str) -> Path:
        record = self.get(experiment_id)
        if not record.checkpoint_exists:
            raise FileNotFoundError(
                f"Checkpoint missing for '{experiment_id}': {record.checkpoint_path}"
            )
        return record.checkpoint_path

    def resolve_metrics(self, experiment_id: str) -> Mapping[str, Any]:
        return self.get(experiment_id).metrics

    def validate(self, *, require_checkpoints: bool = True) -> list[str]:
        """Return a list of validation error strings (empty means OK)."""
        errors: list[str] = []
        if not self.records:
            errors.append("Experiment registry contains no experiments.")
        for experiment_id, record in self.records.items():
            if require_checkpoints and not record.checkpoint_exists:
                errors.append(f"Missing checkpoint for '{experiment_id}': {record.checkpoint_path}")
            if not record.config_exists:
                errors.append(f"Missing config for '{experiment_id}': {record.config_path}")
            if "test_macro_dice" not in record.metrics:
                errors.append(f"Experiment '{experiment_id}' metrics missing test_macro_dice.")
        return errors
