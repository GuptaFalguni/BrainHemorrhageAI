"""Research utility for BHSD analysis.

Run: python -m research.analyze_dataset
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_analysis(project_root: Path | None = None) -> SimpleNamespace:
    """Run Phase-1 dataset analysis and return structured results.

    Algorithm matches the original research script (pairing, shapes, spacing,
    label presence, voxel counts). Side-effect free except for file reads.
    """
    root = project_root or _project_root()
    images_dir = root / "data" / "raw" / "label_192" / "images"
    masks_dir = root / "data" / "raw" / "label_192" / "ground truths"

    image_files = sorted(images_dir.glob("*.nii.gz"))
    mask_files = sorted(masks_dir.glob("*.nii.gz"))
    image_names = {p.name for p in image_files}
    mask_names = {p.name for p in mask_files}

    images_without_mask = [p.name for p in image_files if p.name not in mask_names]
    masks_without_image = [p.name for p in mask_files if p.name not in image_names]

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    shapes: list[tuple[int, ...]] = []
    spacings: list[tuple[float, float, float]] = []
    labels_in_dataset: set[int] = set()
    scans_with_label: Counter[int] = Counter()
    voxel_counts: Counter[int] = Counter()

    for image_path in image_files:
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            continue

        image_nii = nib.load(str(image_path))
        mask_nii = nib.load(str(mask_path))
        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        if image_data.shape != mask_data.shape:
            shape_mismatches.append((image_path.name, image_data.shape, mask_data.shape))

        shapes.append(image_data.shape)
        spacings.append(tuple(float(z) for z in image_nii.header.get_zooms()[:3]))

        unique_labels = np.unique(mask_data.astype(np.int64))
        for label in unique_labels:
            label = int(label)
            labels_in_dataset.add(label)
            scans_with_label[label] += 1
            voxel_counts[label] += int(np.sum(mask_data == label))

    unique_shapes = sorted(set(shapes))
    return SimpleNamespace(
        image_files=image_files,
        mask_files=mask_files,
        images_without_mask=images_without_mask,
        masks_without_image=masks_without_image,
        shape_mismatches=shape_mismatches,
        shapes=shapes,
        spacings=spacings,
        labels_in_dataset=labels_in_dataset,
        scans_with_label=scans_with_label,
        voxel_counts=voxel_counts,
        unique_shapes=unique_shapes,
    )


def print_report(result: SimpleNamespace) -> None:
    """Print the same CLI report as the original analyze_dataset script."""
    image_files = result.image_files
    mask_files = result.mask_files
    images_without_mask = result.images_without_mask
    masks_without_image = result.masks_without_image
    shape_mismatches = result.shape_mismatches
    shapes = result.shapes
    spacings = result.spacings
    labels_in_dataset = result.labels_in_dataset
    scans_with_label = result.scans_with_label
    voxel_counts = result.voxel_counts
    unique_shapes = result.unique_shapes

    print(f"1. total number of image files: {len(image_files)}")
    print(f"2. total number of mask files: {len(mask_files)}")

    print("3. image-mask pairing:")
    print(f"   images without matching mask: {len(images_without_mask)}")
    for name in images_without_mask:
        print(f"   - {name}")
    print(f"   masks without matching image: {len(masks_without_image)}")
    for name in masks_without_image:
        print(f"   - {name}")

    print("4. shape verification:")
    print(f"   image-mask shape mismatches: {len(shape_mismatches)}")
    for name, image_shape, mask_shape in shape_mismatches:
        print(f"   - {name}: image {image_shape}, mask {mask_shape}")

    print(f"5. unique label values across dataset: {sorted(labels_in_dataset)}")

    print("6. scans containing each label:")
    for label in sorted(scans_with_label):
        print(f"   label {label}: {scans_with_label[label]} scans")

    shape_array = np.array(shapes)
    print("7. image shape statistics:")
    print(f"   minimum shape (per axis): {tuple(int(x) for x in shape_array.min(axis=0))}")
    print(f"   maximum shape (per axis): {tuple(int(x) for x in shape_array.max(axis=0))}")
    print(f"   unique shapes ({len(unique_shapes)}):")
    for shape in unique_shapes:
        count = shapes.count(shape)
        print(f"   - {shape}: {count} scans")

    spacing_array = np.array(spacings)
    print("8. voxel spacing statistics (mm):")
    print(f"   minimum (x, y, z): {tuple(round(x, 4) for x in spacing_array.min(axis=0))}")
    print(f"   maximum (x, y, z): {tuple(round(x, 4) for x in spacing_array.max(axis=0))}")
    print(f"   average (x, y, z): {tuple(round(x, 4) for x in spacing_array.mean(axis=0))}")

    print("9. total voxel count per label:")
    for label in sorted(voxel_counts):
        print(f"   label {label}: {voxel_counts[label]} voxels")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    paired = len(images_without_mask) == 0 and len(masks_without_image) == 0
    aligned = len(shape_mismatches) == 0
    single_shape = len(unique_shapes) == 1
    single_spacing = np.allclose(spacing_array.min(axis=0), spacing_array.max(axis=0))

    multi_class = len(labels_in_dataset - {0}) > 1
    expected_labels = {0, 1, 2, 3, 4, 5}
    present_hemorrhage_labels = sorted(labels_in_dataset - {0})
    missing_labels = sorted(expected_labels - labels_in_dataset)

    anisotropic = spacing_array.mean(axis=0)[2] / spacing_array.mean(axis=0)[0] > 3

    print(f"- Multi-class segmentation: {'yes' if multi_class else 'no'}")
    print(f"  Hemorrhage labels found: {present_hemorrhage_labels}")
    if missing_labels:
        print(f"  Labels never appearing in any scan: {missing_labels}")
    print(f"- Every subtype appears in at least one scan: {'yes' if not missing_labels else 'no'}")
    if missing_labels:
        print(f"  Missing subtypes: {missing_labels}")
    print("  Not every scan contains every subtype; see section 6 above.")
    print("- Dataset consistency:")
    print(f"  All images paired with masks: {'yes' if paired else 'no'}")
    print(f"  All image-mask shapes match: {'yes' if aligned else 'no'}")
    print(f"  Uniform volume shape: {'yes' if single_shape else 'no'} ({len(unique_shapes)} unique shapes)")
    print(f"  Uniform voxel spacing: {'yes' if single_spacing else 'no'}")
    print(
        f"- Resampling likely needed: "
        f"{'yes' if not single_spacing or not single_shape or anisotropic else 'probably not'}"
    )
    if anisotropic:
        print("  Reason: strong slice thickness anisotropy (avg z-spacing >> in-plane spacing)")
    if not single_spacing:
        print("  Reason: voxel spacing varies across scans")
    if not single_shape:
        print("  Reason: volume dimensions vary across scans")
    issues = []
    if not paired:
        issues.append("missing image-mask pairs")
    if not aligned:
        issues.append("shape mismatches between images and masks")
    if missing_labels:
        issues.append(f"labels {missing_labels} absent from entire dataset")
    if not issues:
        print("- Issues discovered: none")
    else:
        print(f"- Issues discovered: {', '.join(issues)}")


def main() -> None:
    print_report(run_analysis())


if __name__ == "__main__":
    main()
