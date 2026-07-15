"""Research utility for BHSD analysis.

Run: python -m research.inspect_dataset
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    images_dir = project_root / "data" / "raw" / "label_192" / "images"
    masks_dir = project_root / "data" / "raw" / "label_192" / "ground truths"

    image_files = sorted(images_dir.glob("*.nii.gz"))
    mask_files = sorted(masks_dir.glob("*.nii.gz"))

    image_path = image_files[0]
    mask_path = masks_dir / image_path.name

    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))

    image_data = image_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    print(f"total number of CT scans: {len(image_files)}")
    print(f"total number of masks: {len(mask_files)}")
    print(f"image filename: {image_path.name}")
    print(f"mask filename: {mask_path.name}")
    print(f"image shape: {image_data.shape}")
    print(f"voxel spacing: {image_nii.header.get_zooms()[:3]}")
    print(f"image datatype: {image_data.dtype}")
    print(f"mask datatype: {mask_data.dtype}")
    print(f"unique values inside the mask: {np.unique(mask_data)}")


if __name__ == "__main__":
    main()
