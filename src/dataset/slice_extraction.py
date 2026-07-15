"""
2.5D slice extraction from preprocessed 3D BHSD volumes.

Builds three-channel inputs (previous, current, next axial slice) and predicts
the center slice label, with edge replication per training_pipeline_design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.bhsd_dataset import get_volume_depth, load_split_filenames
from preprocessing.preprocess_volume import PreprocessedVolume

AXIAL_AXIS = 2
NUM_INPUT_CHANNELS = 3


@dataclass(frozen=True)
class SliceSample:
    """One 2.5D training/inference sample with volume traceability metadata."""

    image: np.ndarray
    label: np.ndarray
    slice_index: int
    volume_depth: int
    filename: str
    original_spacing: tuple[float, float, float]
    original_shape: tuple[int, ...]
    affine: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Convert to a MONAI-friendly dictionary."""
        return {
            "image": self.image,
            "label": self.label,
            "slice_index": self.slice_index,
            "volume_depth": self.volume_depth,
            "filename": self.filename,
            "original_spacing": self.original_spacing,
            "original_shape": self.original_shape,
            "affine": self.affine,
        }


def _neighbor_index(slice_index: int, offset: int, depth: int) -> int:
    """Return neighbor index with edge replication at volume boundaries."""
    neighbor = slice_index + offset
    if neighbor < 0:
        return 0
    if neighbor >= depth:
        return depth - 1
    return neighbor


def extract_axial_slice(volume: np.ndarray, slice_index: int, axis: int = AXIAL_AXIS) -> np.ndarray:
    """Extract one 2D axial slice from a 3D array."""
    depth = volume.shape[axis]
    if slice_index < 0 or slice_index >= depth:
        raise IndexError(f"slice_index {slice_index} out of range for depth {depth}.")
    return np.take(volume, slice_index, axis=axis)


def build_2p5d_stack(
    volume: np.ndarray,
    slice_index: int,
    axis: int = AXIAL_AXIS,
) -> np.ndarray:
    """
    Stack previous, current, and next axial slices as channels.

    Returns
    -------
    np.ndarray
        Array with shape ``(3, H, W)``.
    """
    depth = volume.shape[axis]
    indices = (
        _neighbor_index(slice_index, -1, depth),
        slice_index,
        _neighbor_index(slice_index, 1, depth),
    )
    slices = [extract_axial_slice(volume, index, axis=axis) for index in indices]
    return np.stack(slices, axis=0).astype(np.float32)


def extract_slice_sample(
    volume: PreprocessedVolume,
    slice_index: int,
    axis: int = AXIAL_AXIS,
) -> SliceSample:
    """Extract one 2.5D sample from a preprocessed volume."""
    depth = volume.processed_image.shape[axis]
    if slice_index < 0 or slice_index >= depth:
        raise IndexError(
            f"slice_index {slice_index} out of range for {volume.filename} (depth={depth})."
        )

    image = build_2p5d_stack(volume.processed_image, slice_index, axis=axis)
    label = extract_axial_slice(volume.processed_mask, slice_index, axis=axis).astype(np.int64)

    return SliceSample(
        image=image,
        label=label,
        slice_index=slice_index,
        volume_depth=depth,
        filename=volume.filename,
        original_spacing=volume.original_spacing,
        original_shape=volume.original_shape,
        affine=volume.affine,
    )


def volume_to_slice_samples(
    volume: PreprocessedVolume,
    axis: int = AXIAL_AXIS,
) -> list[SliceSample]:
    """Convert an entire volume into sequential 2.5D samples."""
    depth = volume.processed_image.shape[axis]
    return [extract_slice_sample(volume, index, axis=axis) for index in range(depth)]


def build_slice_records(
    split: str,
    axis: int = AXIAL_AXIS,
) -> list[dict[str, object]]:
    """
    Build flat slice indices for a split without loading volumes.

    Each record contains ``filename`` and ``slice_index`` for MONAI datasets.
    """
    records: list[dict[str, object]] = []
    for filename in load_split_filenames(split):  # type: ignore[arg-type]
        depth = get_volume_depth(filename, axis=axis)
        for slice_index in range(depth):
            records.append({"filename": filename, "slice_index": slice_index})
    return records
