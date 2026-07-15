"""Research utility for BHSD hemorrhage volume statistics.

Run: python -m research.calculate_volume
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    images_dir = project_root / "data" / "raw" / "label_192" / "images"
    masks_dir = project_root / "data" / "raw" / "label_192" / "ground truths"
    output_csv = project_root / "data" / "metadata" / "volume_statistics.csv"

    label_names = {1: "EDH", 2: "SDH", 3: "SAH", 4: "IPH", 5: "IVH"}

    image_files = sorted(images_dir.glob("*.nii.gz"))
    rows = []
    scan_total_volumes = []

    for scan_idx, image_path in enumerate(image_files):
        mask_path = masks_dir / image_path.name

        image_nii = nib.load(str(image_path))
        mask_data = nib.load(str(mask_path)).get_fdata()

        spacing = image_nii.header.get_zooms()[:3]
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

        scan_total_mm3 = 0.0
        scan_label_stats = []

        for label in range(1, 6):
            voxel_count = int(np.sum(mask_data == label))
            volume_mm3 = voxel_count * voxel_volume_mm3
            volume_ml = volume_mm3 / 1000.0
            scan_total_mm3 += volume_mm3

            rows.append(
                {
                    "filename": image_path.name,
                    "label": label,
                    "voxel_count": voxel_count,
                    "volume_mm3": volume_mm3,
                    "volume_ml": volume_ml,
                }
            )
            scan_label_stats.append((label, voxel_count, volume_mm3, volume_ml))

        scan_total_volumes.append(scan_total_mm3)

        if scan_idx < 5:
            print(f"--- scan {scan_idx + 1}: {image_path.name} ---")
            print(f"voxel spacing (mm): ({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f})")
            print(f"voxel volume (mm³): {voxel_volume_mm3:.4f}")
            for label, voxel_count, volume_mm3, volume_ml in scan_label_stats:
                if voxel_count > 0:
                    print(
                        f"  label {label} ({label_names[label]}): "
                        f"{voxel_count} voxels, {volume_mm3:.2f} mm³, {volume_ml:.4f} mL"
                    )
            print(f"  total hemorrhage: {scan_total_mm3:.2f} mm³, {scan_total_mm3 / 1000:.4f} mL")
            print()

    scan_totals_array = np.array(scan_total_volumes)
    print("dataset statistics (total hemorrhage per scan, labels 1-5 combined):")
    print(f"  minimum: {scan_totals_array.min():.2f} mm³ ({scan_totals_array.min() / 1000:.4f} mL)")
    print(f"  maximum: {scan_totals_array.max():.2f} mm³ ({scan_totals_array.max() / 1000:.4f} mL)")
    print(f"  average: {scan_totals_array.mean():.2f} mm³ ({scan_totals_array.mean() / 1000:.4f} mL)")
    print()

    print("average volume per hemorrhage subtype (among scans where label is present):")
    df = pd.DataFrame(rows)
    for label in range(1, 6):
        label_rows = df[(df["label"] == label) & (df["voxel_count"] > 0)]
        if len(label_rows) == 0:
            print(f"  label {label} ({label_names[label]}): no scans")
        else:
            print(
                f"  label {label} ({label_names[label]}): "
                f"{label_rows['volume_mm3'].mean():.2f} mm³ "
                f"({label_rows['volume_ml'].mean():.4f} mL) "
                f"across {len(label_rows)} scans"
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print()
    print(f"saved: {output_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
