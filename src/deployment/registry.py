"""Convenience loaders for deployment registries."""

from __future__ import annotations

from pathlib import Path

from deployment.experiment_registry import ExperimentRegistry
from deployment.model_registry import ModelRegistry


def load_deployment_registries(
    *,
    project_root: Path | str | None = None,
    experiment_registry_path: Path | str | None = None,
    model_registry_path: Path | str | None = None,
) -> tuple[ExperimentRegistry, ModelRegistry]:
    """Load experiment and model registries with cross-links validated at load time."""
    experiments = ExperimentRegistry.from_yaml(
        experiment_registry_path,
        project_root=project_root,
    )
    models = ModelRegistry.from_yaml(
        model_registry_path,
        experiment_registry=experiments,
        project_root=project_root,
    )
    return experiments, models
