"""Configuration loading utilities for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "config" / "model.yaml"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "config" / "training.yaml"


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {config_path}")
    return config


def load_model_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load model configuration."""
    return load_yaml_config(path or DEFAULT_MODEL_CONFIG)


def load_training_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load training configuration."""
    return load_yaml_config(path or DEFAULT_TRAINING_CONFIG)


def resolve_device(device_config: str | None = None) -> torch.device:
    """Resolve training device from configuration."""
    requested = (device_config or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def should_use_amp(training_config: dict[str, Any], device: torch.device) -> bool:
    """Return whether automatic mixed precision should be enabled."""
    configured = bool(training_config.get("mixed_precision", False))
    return configured and device.type == "cuda" and torch.cuda.is_available()
