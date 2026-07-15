"""
MONAI dataset and DataLoader construction for BHSD 2.5D training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from monai.data import DataLoader, Dataset

from dataset.bhsd_dataset import SplitName, VolumeCache, _validate_split
from dataset.slice_extraction import build_slice_records
from dataset.transforms import get_train_transforms, get_val_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "config" / "training.yaml"
DEFAULT_PREPROCESSING_CONFIG = PROJECT_ROOT / "config" / "preprocessing.yaml"


def resolve_preprocessing_config_path(
    training_config: dict[str, Any],
    preprocessing_config_path: Path | None = None,
) -> Path | None:
    """Resolve preprocessing YAML path from explicit argument or training config."""
    if preprocessing_config_path is not None:
        return preprocessing_config_path
    config_ref = training_config.get("preprocessing_config")
    if not config_ref:
        return None
    path = Path(config_ref)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_training_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load dataloader settings from ``config/training.yaml``."""
    path = config_path or DEFAULT_TRAINING_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {path}")
    return config


def build_monai_dataset(
    split: SplitName,
    config_path: Path | None = None,
    preprocessing_config_path: Path | None = None,
    volume_cache: VolumeCache | None = None,
) -> Dataset:
    """Create a MONAI ``Dataset`` of 2.5D slice records for one split."""
    split = _validate_split(split)
    cache = volume_cache or VolumeCache(config_path=preprocessing_config_path)
    transform = (
        get_train_transforms(cache, config_path=preprocessing_config_path)
        if split == "train"
        else get_val_transforms(cache, config_path=preprocessing_config_path)
    )
    records = build_slice_records(split)
    return Dataset(data=records, transform=transform)


def build_dataloader(
    split: SplitName,
    training_config_path: Path | None = None,
    preprocessing_config_path: Path | None = None,
    volume_cache: VolumeCache | None = None,
) -> DataLoader:
    """Build a MONAI ``DataLoader`` for the requested split."""
    split = _validate_split(split)
    config = load_training_config(training_config_path)
    resolved_preprocessing_path = resolve_preprocessing_config_path(
        config,
        preprocessing_config_path,
    )
    dataset = build_monai_dataset(
        split=split,
        preprocessing_config_path=resolved_preprocessing_path,
        volume_cache=volume_cache,
    )

    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 0))
    shuffle = bool(config.get("shuffle_train", False)) if split == "train" else False
    pin_memory = bool(config.get("pin_memory", False))
    persistent_workers = bool(config.get("persistent_workers", False))

    if num_workers == 0 and persistent_workers:
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
