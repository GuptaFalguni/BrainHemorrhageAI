"""Model registry — deployable models referencing experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from deployment.experiment_registry import ExperimentRecord, ExperimentRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "config" / "models" / "registry.yaml"

Category = Literal["interactive", "research"]

REQUIRED_MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "model_id",
        "display_name",
        "experiment_id",
        "category",
        "framework",
        "version",
        "default",
        "supports_overlay",
        "supports_volume",
        "supports_confidence",
        "supports_multiclass",
        "supports_severity",
        "supports_uncertainty",
    }
)

CAPABILITY_FIELDS: tuple[str, ...] = (
    "supports_overlay",
    "supports_volume",
    "supports_confidence",
    "supports_multiclass",
    "supports_severity",
    "supports_uncertainty",
)


@dataclass(frozen=True)
class ModelCapabilities:
    """Frontend feature flags for a deployable model."""

    supports_overlay: bool
    supports_volume: bool
    supports_confidence: bool
    supports_multiclass: bool
    supports_severity: bool
    supports_uncertainty: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "supports_overlay": self.supports_overlay,
            "supports_volume": self.supports_volume,
            "supports_confidence": self.supports_confidence,
            "supports_multiclass": self.supports_multiclass,
            "supports_severity": self.supports_severity,
            "supports_uncertainty": self.supports_uncertainty,
        }


@dataclass(frozen=True)
class ModelRecord:
    """One deployable model entry from the model registry."""

    model_id: str
    display_name: str
    experiment_id: str
    category: Category
    framework: str
    version: str
    default: bool
    capabilities: ModelCapabilities
    description: str = ""
    paper: str = ""
    training_dataset: str = ""
    macro_dice: float | None = None
    volume_error_mean_ml: float | None = None
    speed: str = ""
    limitations: str = ""
    recommended_use: str = ""
    model_card: str = ""

    def metadata(self) -> dict[str, Any]:
        """Frontend-consumable metadata dictionary."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "experiment_id": self.experiment_id,
            "category": self.category,
            "framework": self.framework,
            "version": self.version,
            "default": self.default,
            "description": self.description,
            "paper": self.paper,
            "training_dataset": self.training_dataset,
            "macro_dice": self.macro_dice,
            "volume_error_mean_ml": self.volume_error_mean_ml,
            "speed": self.speed,
            "limitations": self.limitations,
            "recommended_use": self.recommended_use,
            "model_card": self.model_card,
            "capabilities": self.capabilities.as_dict(),
        }


@dataclass
class ModelRegistry:
    """Load and validate ``config/models/registry.yaml``."""

    records: dict[str, ModelRecord] = field(default_factory=dict)
    experiment_registry: ExperimentRegistry | None = None
    registry_path: Path = DEFAULT_MODEL_REGISTRY
    project_root: Path = PROJECT_ROOT

    @classmethod
    def from_yaml(
        cls,
        path: Path | str | None = None,
        *,
        experiment_registry: ExperimentRegistry | None = None,
        project_root: Path | str | None = None,
    ) -> ModelRegistry:
        root = Path(project_root) if project_root else PROJECT_ROOT
        registry_path = Path(path) if path else root / "config" / "models" / "registry.yaml"
        if not registry_path.is_file():
            raise FileNotFoundError(f"Model registry not found: {registry_path}")

        experiments = experiment_registry or ExperimentRegistry.from_yaml(project_root=root)

        with registry_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        if not isinstance(payload, dict) or "models" not in payload:
            raise ValueError(f"Model registry must contain a 'models' list: {registry_path}")

        models = payload["models"]
        if not isinstance(models, list) or not models:
            raise ValueError("Model registry 'models' must be a non-empty list.")

        records: dict[str, ModelRecord] = {}
        for index, entry in enumerate(models):
            if not isinstance(entry, dict):
                raise ValueError(f"Model entry at index {index} must be a mapping.")
            missing = REQUIRED_MODEL_FIELDS - set(entry)
            if missing:
                raise ValueError(f"Model entry at index {index} missing fields: {sorted(missing)}")

            model_id = str(entry["model_id"]).strip()
            if not model_id:
                raise ValueError(f"Model entry at index {index} has empty model_id.")
            if model_id in records:
                raise ValueError(f"Duplicate model_id: {model_id}")

            category = str(entry["category"]).strip().lower()
            if category not in ("interactive", "research"):
                raise ValueError(
                    f"Model '{model_id}' category must be 'interactive' or 'research', got '{category}'."
                )

            for capability in CAPABILITY_FIELDS:
                if not isinstance(entry[capability], bool):
                    raise ValueError(
                        f"Model '{model_id}' field '{capability}' must be a boolean."
                    )

            if not isinstance(entry["default"], bool):
                raise ValueError(f"Model '{model_id}' field 'default' must be a boolean.")

            experiment_id = str(entry["experiment_id"]).strip()
            if experiment_id not in experiments.records:
                raise ValueError(
                    f"Model '{model_id}' references unknown experiment_id '{experiment_id}'."
                )

            capabilities = ModelCapabilities(
                supports_overlay=bool(entry["supports_overlay"]),
                supports_volume=bool(entry["supports_volume"]),
                supports_confidence=bool(entry["supports_confidence"]),
                supports_multiclass=bool(entry["supports_multiclass"]),
                supports_severity=bool(entry["supports_severity"]),
                supports_uncertainty=bool(entry["supports_uncertainty"]),
            )

            macro_dice = entry.get("macro_dice")
            volume_error = entry.get("volume_error_mean_ml")

            records[model_id] = ModelRecord(
                model_id=model_id,
                display_name=str(entry["display_name"]),
                experiment_id=experiment_id,
                category=category,  # type: ignore[arg-type]
                framework=str(entry["framework"]),
                version=str(entry["version"]),
                default=bool(entry["default"]),
                capabilities=capabilities,
                description=str(entry.get("description", "")),
                paper=str(entry.get("paper", "")),
                training_dataset=str(entry.get("training_dataset", "")),
                macro_dice=float(macro_dice) if macro_dice is not None else None,
                volume_error_mean_ml=float(volume_error) if volume_error is not None else None,
                speed=str(entry.get("speed", "")),
                limitations=str(entry.get("limitations", "")),
                recommended_use=str(entry.get("recommended_use", "")),
                model_card=str(entry.get("model_card", "")),
            )

        return cls(
            records=records,
            experiment_registry=experiments,
            registry_path=registry_path.resolve(),
            project_root=root.resolve(),
        )

    def get(self, model_id: str) -> ModelRecord:
        try:
            return self.records[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model_id: {model_id}") from exc

    def list_ids(self) -> list[str]:
        return sorted(self.records)

    def list_enabled(self) -> list[ModelRecord]:
        return [self.records[key] for key in self.list_ids()]

    def get_default(self) -> ModelRecord:
        defaults = [record for record in self.records.values() if record.default]
        if len(defaults) != 1:
            raise ValueError(
                f"Expected exactly one default model, found {len(defaults)}: "
                f"{[item.model_id for item in defaults]}"
            )
        return defaults[0]

    def resolve_experiment(self, model_id: str) -> ExperimentRecord:
        if self.experiment_registry is None:
            raise RuntimeError("ModelRegistry has no ExperimentRegistry attached.")
        model = self.get(model_id)
        return self.experiment_registry.get(model.experiment_id)

    def capabilities(self, model_id: str) -> ModelCapabilities:
        return self.get(model_id).capabilities

    def metadata(self, model_id: str) -> dict[str, Any]:
        return self.get(model_id).metadata()

    def deployment_info(self, model_id: str) -> dict[str, Any]:
        """Return deploy-facing information including checkpoint paths."""
        model = self.get(model_id)
        experiment = self.resolve_experiment(model_id)
        return {
            "model": model.metadata(),
            "experiment_id": experiment.experiment_id,
            "checkpoint_path": str(experiment.checkpoint_path),
            "config_path": str(experiment.config_path),
            "framework": experiment.framework,
            "model_type": experiment.model_type,
            "publication_status": experiment.publication_status,
            "metrics": dict(experiment.metrics),
        }

    def validate(self, *, require_checkpoints: bool = True) -> list[str]:
        """Return a list of validation error strings (empty means OK)."""
        errors: list[str] = []
        if not self.records:
            errors.append("Model registry contains no models.")

        defaults = [record.model_id for record in self.records.values() if record.default]
        if len(defaults) != 1:
            errors.append(f"Expected exactly one default model, found {len(defaults)}: {defaults}")

        if self.experiment_registry is None:
            errors.append("Model registry is missing an attached ExperimentRegistry.")
            return errors

        errors.extend(self.experiment_registry.validate(require_checkpoints=require_checkpoints))

        for model_id, model in self.records.items():
            if model.experiment_id not in self.experiment_registry.records:
                errors.append(
                    f"Model '{model_id}' references missing experiment '{model.experiment_id}'."
                )
                continue
            experiment = self.experiment_registry.get(model.experiment_id)
            if model.framework != experiment.framework:
                errors.append(
                    f"Model '{model_id}' framework '{model.framework}' does not match "
                    f"experiment framework '{experiment.framework}'."
                )
            for capability_name, value in model.capabilities.as_dict().items():
                if not isinstance(value, bool):
                    errors.append(f"Model '{model_id}' capability '{capability_name}' is not bool.")
        return errors
