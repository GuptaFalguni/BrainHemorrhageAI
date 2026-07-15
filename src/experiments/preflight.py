"""Pre-training verification checks for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "images"
MASKS_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "ground truths"
SPLITS_CSV = PROJECT_ROOT / "data" / "metadata" / "splits.csv"
PREPROCESSING_CONFIG = PROJECT_ROOT / "config" / "preprocessing.yaml"
MODEL_CONFIG = PROJECT_ROOT / "config" / "model.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "config" / "training.yaml"
EVALUATION_CONFIG = PROJECT_ROOT / "config" / "evaluation.yaml"

REQUIRED_PREPROCESSING_KEYS = (
    "clip_min",
    "clip_max",
    "normalization_method",
    "train_mean",
    "train_std",
)
REQUIRED_MODEL_KEYS = ("in_channels", "out_channels", "channels", "strides")
REQUIRED_TRAINING_KEYS = ("epochs", "learning_rate", "batch_size", "seed")


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of pre-training verification."""

    passed: bool
    checks: dict[str, str]
    device: str
    cuda_available: bool
    seed: int | None


class PreflightError(RuntimeError):
    """Raised when a mandatory pre-training check fails."""


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file."""
    if not path.exists():
        raise PreflightError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise PreflightError(f"Invalid config format: {path}")
    return config


def _check_dataset_exists(checks: dict[str, str]) -> None:
    """Verify raw dataset directories contain NIfTI files."""
    if not IMAGES_DIR.exists():
        raise PreflightError(f"Images directory not found: {IMAGES_DIR}")
    if not MASKS_DIR.exists():
        raise PreflightError(f"Masks directory not found: {MASKS_DIR}")
    image_count = len(list(IMAGES_DIR.glob("*.nii.gz")))
    mask_count = len(list(MASKS_DIR.glob("*.nii.gz")))
    if image_count == 0:
        raise PreflightError(f"No CT volumes found in {IMAGES_DIR}")
    if mask_count == 0:
        raise PreflightError(f"No masks found in {MASKS_DIR}")
    checks["dataset"] = f"OK ({image_count} images, {mask_count} masks)"


def _check_splits(checks: dict[str, str]) -> None:
    """Verify split manifest exists."""
    if not SPLITS_CSV.exists():
        raise PreflightError(f"Split manifest not found: {SPLITS_CSV}")
    checks["splits_csv"] = f"OK ({SPLITS_CSV})"


def _check_preprocessing_config(checks: dict[str, str]) -> None:
    """Verify preprocessing config contains required keys."""
    config = _load_yaml(PREPROCESSING_CONFIG)
    missing = [key for key in REQUIRED_PREPROCESSING_KEYS if key not in config]
    if missing:
        raise PreflightError(f"preprocessing.yaml missing keys: {missing}")
    checks["preprocessing_config"] = "OK"


def _check_model_config(checks: dict[str, str]) -> None:
    """Verify model config loads."""
    config = _load_yaml(MODEL_CONFIG)
    missing = [key for key in REQUIRED_MODEL_KEYS if key not in config]
    if missing:
        raise PreflightError(f"model.yaml missing keys: {missing}")
    checks["model_config"] = "OK"


def _check_training_config(checks: dict[str, str]) -> tuple[int]:
    """Verify training config loads."""
    config = _load_yaml(TRAINING_CONFIG)
    missing = [key for key in REQUIRED_TRAINING_KEYS if key not in config]
    if missing:
        raise PreflightError(f"training.yaml missing keys: {missing}")
    checks["training_config"] = "OK"
    return int(config["seed"])


def _check_evaluation_config(checks: dict[str, str]) -> None:
    """Verify evaluation config loads."""
    _load_yaml(EVALUATION_CONFIG)
    checks["evaluation_config"] = "OK"


def _ensure_output_directories(checks: dict[str, str]) -> None:
    """Create required output directories."""
    directories = [
        PROJECT_ROOT / "reports" / "training",
        PROJECT_ROOT / "reports" / "evaluation",
        PROJECT_ROOT / "reports" / "experiments",
        PROJECT_ROOT / "checkpoints",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    checks["output_folders"] = "OK"
    checks["checkpoint_directory"] = f"OK ({PROJECT_ROOT / 'checkpoints'})"


def _check_device_and_seed(checks: dict[str, str], seed: int) -> tuple[str, bool]:
    """Report CUDA availability and seed."""
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    checks["cuda_available"] = str(cuda_available)
    checks["device"] = device if device == "cuda" else "cpu (fallback)"
    checks["random_seed"] = str(seed)
    return device, cuda_available


def run_preflight_checks(seed_override: int | None = None) -> PreflightResult:
    """Run all mandatory checks before training.

    Raises:
        PreflightError: When any check fails.

    Returns:
        PreflightResult with check details.
    """
    checks: dict[str, str] = {}
    _check_dataset_exists(checks)
    _check_splits(checks)
    _check_preprocessing_config(checks)
    _check_model_config(checks)
    seed = _check_training_config(checks)
    if seed_override is not None:
        seed = seed_override
    _check_evaluation_config(checks)
    _ensure_output_directories(checks)
    device, cuda_available = _check_device_and_seed(checks, seed)

    return PreflightResult(
        passed=True,
        checks=checks,
        device=device,
        cuda_available=cuda_available,
        seed=seed,
    )


def print_preflight_report(result: PreflightResult) -> None:
    """Print a human-readable preflight summary."""
    print("Pre-training verification")
    for name, status in result.checks.items():
        print(f"  {name}: {status}")
    print(f"  selected_device: {result.device}")
