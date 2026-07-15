"""Deployment package — registries and loaders (Phase C).

Inference types: import from ``deployment.inference`` directly to avoid
circular imports with ``src/inference``.
"""

from deployment.experiment_registry import ExperimentRecord, ExperimentRegistry
from deployment.model_registry import ModelCapabilities, ModelRecord, ModelRegistry
from deployment.registry import load_deployment_registries

__all__ = [
    "ExperimentRecord",
    "ExperimentRegistry",
    "ModelCapabilities",
    "ModelRecord",
    "ModelRegistry",
    "load_deployment_registries",
]
