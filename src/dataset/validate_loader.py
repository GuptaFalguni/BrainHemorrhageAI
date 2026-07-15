"""
Validate the BHSD production data loading pipeline.

Checks split loading, tensor shapes, label integrity, metadata preservation,
2.5D slice extraction (including edge slices), and DataLoader iteration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dataset.bhsd_dataset import (
    BHSDVolumeDataset,
    VALID_SPLITS,
    load_split_filenames,
    load_volume_sample,
)
from dataset.dataloader import build_dataloader, build_monai_dataset
from dataset.slice_extraction import (
    NUM_INPUT_CHANNELS,
    build_2p5d_stack,
    extract_slice_sample,
    volume_to_slice_samples,
)
from preprocessing.preprocess_volume import VALID_LABELS


def _check_volume_splits() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in sorted(VALID_SPLITS):
        dataset = BHSDVolumeDataset(split=split)  # type: ignore[arg-type]
        sample = dataset[0]
        assert sample.processed_image.shape == sample.original_shape
        assert sample.processed_mask.shape == sample.original_shape
        assert len(sample.original_spacing) == 3
        assert sample.affine.shape == (4, 4)
        labels = set(np.unique(sample.processed_mask).astype(int).tolist())
        assert labels.issubset(VALID_LABELS)
        counts[split] = len(dataset)
    return counts


def _check_slice_extraction() -> tuple[str, int, int]:
    filename = load_split_filenames("val")[0]
    volume = load_volume_sample(filename)
    preprocessed = volume.to_preprocessed()
    depth = volume.processed_image.shape[2]

    all_samples = volume_to_slice_samples(preprocessed)
    assert len(all_samples) == depth

    first = extract_slice_sample(preprocessed, 0)
    last = extract_slice_sample(preprocessed, depth - 1)

    first_stack = build_2p5d_stack(volume.processed_image, 0)
    last_stack = build_2p5d_stack(volume.processed_image, depth - 1)
    assert np.array_equal(first_stack[0], first_stack[1])
    assert np.array_equal(last_stack[1], last_stack[2])

    for sample in (first, last, all_samples[depth // 2]):
        assert sample.image.shape[0] == NUM_INPUT_CHANNELS
        assert sample.image.shape[1:] == sample.label.shape
        assert sample.slice_index >= 0
        assert sample.volume_depth == depth
        assert sample.original_shape == volume.original_shape
        assert sample.original_spacing == volume.original_spacing
        assert np.array_equal(sample.affine, volume.affine)
        assert set(np.unique(sample.label).astype(int).tolist()).issubset(VALID_LABELS)

    return filename, depth, len(all_samples)


def _check_dataloader_splits() -> dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]]:
    summary: dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]] = {}
    for split in sorted(VALID_SPLITS):
        loader = build_dataloader(split=split)  # type: ignore[arg-type]
        batch = next(iter(loader))
        image = batch["image"]
        label = batch["label"]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert image.dtype == torch.float32
        assert label.dtype == torch.long
        assert image.ndim == 4
        assert label.ndim == 3
        assert image.shape[1] == NUM_INPUT_CHANNELS
        assert image.shape[2:] == label.shape[1:]
        labels = set(torch.unique(label).tolist())
        assert labels.issubset(VALID_LABELS)
        assert "original_spacing" in batch
        assert "affine" in batch
        assert "slice_index" in batch
        summary[split] = (
            len(build_monai_dataset(split=split)),  # type: ignore[arg-type]
            tuple(int(v) for v in image.shape),
            tuple(int(v) for v in label.shape),
        )
    return summary


def main() -> None:
    volume_counts = _check_volume_splits()
    sample_name, edge_depth, slice_count = _check_slice_extraction()
    loader_summary = _check_dataloader_splits()

    print("BHSD data loader validation - PASSED")
    print()
    print("Volume splits:")
    for split, count in volume_counts.items():
        print(f"  {split}: {count} scans")
    print()
    print("Slice extraction (sample val volume):")
    print(f"  {sample_name}: depth={edge_depth}, slices={slice_count}, edge replication verified")
    print()
    print("DataLoader (first batch per split):")
    for split, (records, image_shape, label_shape) in loader_summary.items():
        print(f"  {split}: {records} slice records, image={image_shape}, label={label_shape}")


if __name__ == "__main__":
    main()
