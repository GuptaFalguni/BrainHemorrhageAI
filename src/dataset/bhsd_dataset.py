"""
Volume-level BHSD dataset: split filtering, NIfTI loading, and preprocessing.

Each sample is a native-space, intensity-preprocessed CT volume with its
segmentation mask and preserved geometry metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

from dataset.volume_cache import VolumeCache
from preprocessing.preprocess_volume import (
    IMAGES_DIR,
    VALID_LABELS,
    PreprocessedVolume,
    preprocess_volume,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = PROJECT_ROOT / "data" / "metadata" / "splits.csv"
SplitName = Literal["train", "val", "test"]
VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})

__all__ = [
    "BHSDVolumeDataset",
    "VALID_SPLITS",
    "VolumeCache",
    "VolumeSample",
    "get_volume_depth",
    "load_split_filenames",
    "load_volume_sample",
]


@dataclass(frozen=True)
class VolumeSample:
    """Intensity-preprocessed 3D volume with native geometry metadata."""

    processed_image: np.ndarray
    processed_mask: np.ndarray
    original_spacing: tuple[float, float, float]
    original_shape: tuple[int, ...]
    affine: np.ndarray
    filename: str

    @classmethod
    def from_preprocessed(cls, volume: PreprocessedVolume) -> VolumeSample:
        """Build a sample from a ``PreprocessedVolume`` without copying arrays."""
        return cls(
            processed_image=volume.processed_image,
            processed_mask=volume.processed_mask,
            original_spacing=volume.original_spacing,
            original_shape=volume.original_shape,
            affine=volume.affine,
            filename=volume.filename,
        )

    def to_preprocessed(self) -> PreprocessedVolume:
        """Convert back to the preprocessing module dataclass."""
        return PreprocessedVolume(
            processed_image=self.processed_image,
            processed_mask=self.processed_mask,
            original_spacing=self.original_spacing,
            original_shape=self.original_shape,
            affine=self.affine,
            filename=self.filename,
        )


def _validate_split(split: str) -> SplitName:
    normalized = split.strip().lower()
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}. Expected one of {sorted(VALID_SPLITS)}.")
    return normalized  # type: ignore[return-value]


@lru_cache(maxsize=1)
def _load_splits_table() -> pd.DataFrame:
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"Split manifest not found: {SPLITS_CSV}")
    table = pd.read_csv(SPLITS_CSV)
    required = {"filename", "split"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"splits.csv missing columns: {sorted(missing)}")
    return table


def load_split_filenames(split: SplitName) -> list[str]:
    """Return sorted filenames assigned to the requested split."""
    split = _validate_split(split)
    table = _load_splits_table()
    mask = table["split"].astype(str).str.lower() == split
    filenames = sorted(table.loc[mask, "filename"].astype(str).tolist())
    if not filenames:
        raise ValueError(f"No scans found for split {split!r} in {SPLITS_CSV}.")
    return filenames


def get_volume_depth(filename: str, axis: int = 2) -> int:
    """Read axial depth from a CT header without loading the full volume."""
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(f"CT not found: {image_path}")
    shape = nib.load(str(image_path)).shape
    if axis >= len(shape):
        raise ValueError(f"Axis {axis} invalid for shape {shape}.")
    return int(shape[axis])


def load_volume_sample(
    filename: str,
    config_path: Path | None = None,
) -> VolumeSample:
    """Load, preprocess, and wrap one CT/mask pair."""
    return VolumeSample.from_preprocessed(preprocess_volume(filename, config_path=config_path))


class BHSDVolumeDataset:
    """PyTorch-style dataset returning one preprocessed volume per index."""

    def __init__(
        self,
        split: SplitName,
        config_path: Path | None = None,
    ) -> None:
        self.split = _validate_split(split)
        self.config_path = config_path
        self.filenames = load_split_filenames(self.split)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> VolumeSample:
        if index < 0 or index >= len(self.filenames):
            raise IndexError(f"Index {index} out of range for split {self.split!r}.")
        return load_volume_sample(self.filenames[index], config_path=self.config_path)
