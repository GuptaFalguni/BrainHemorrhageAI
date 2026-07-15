"""
MONAI transforms for BHSD 2.5D slice loading.

Train and validation pipelines are separated for future augmentation hooks.
Version 1 uses deterministic transforms only (no augmentation).
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.transforms import Compose, EnsureTyped, MapTransform

from dataset.bhsd_dataset import VolumeCache
from dataset.slice_extraction import extract_slice_sample


class PreprocessExtractSlice(MapTransform):
    """Load a preprocessed volume and extract one 2.5D slice."""

    def __init__(
        self,
        keys: tuple[str, ...],
        volume_cache: VolumeCache,
        config_path: Path | None = None,
    ) -> None:
        super().__init__(keys)
        self.volume_cache = volume_cache
        self.config_path = config_path

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        record = dict(data)
        filename = str(record["filename"])
        slice_index = int(record["slice_index"])
        volume = self.volume_cache.get(filename, config_path=self.config_path)
        sample = extract_slice_sample(volume, slice_index)
        record.update(sample.to_dict())
        return record


def _tensorize_transforms() -> Compose:
    """Deterministic tensor conversion shared by train and validation."""
    return Compose(
        [
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.long),
        ]
    )


def get_train_transforms(
    volume_cache: VolumeCache,
    config_path: Path | None = None,
) -> Compose:
    """Build training transforms (deterministic in Version 1)."""
    return Compose(
        [
            PreprocessExtractSlice(
                keys=("filename",),
                volume_cache=volume_cache,
                config_path=config_path,
            ),
            _tensorize_transforms(),
        ]
    )


def get_val_transforms(
    volume_cache: VolumeCache,
    config_path: Path | None = None,
) -> Compose:
    """Build validation/test transforms (deterministic)."""
    return Compose(
        [
            PreprocessExtractSlice(
                keys=("filename",),
                volume_cache=volume_cache,
                config_path=config_path,
            ),
            _tensorize_transforms(),
        ]
    )
