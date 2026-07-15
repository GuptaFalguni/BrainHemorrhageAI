"""Research utility for BHSD scan visualization.

Run: python -m research.visualize_scan
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


LABEL_COLORS = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#ff7f00",
    5: "#984ea3",
}
LABEL_NAMES = {
    1: "EDH",
    2: "SDH",
    3: "SAH",
    4: "IPH",
    5: "IVH",
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    images_dir = project_root / "data" / "raw" / "label_192" / "images"
    masks_dir = project_root / "data" / "raw" / "label_192" / "ground truths"

    label_colors = LABEL_COLORS
    label_names = LABEL_NAMES

    image_files = sorted(images_dir.glob("*.nii.gz"))
    image_path = image_files[0]
    mask_path = masks_dir / image_path.name

    image_data = nib.load(str(image_path)).get_fdata()
    mask_data = nib.load(str(mask_path)).get_fdata()

    best_slice = 0
    most_hemorrhage = 0
    for slice_idx in range(mask_data.shape[2]):
        hemorrhage_voxels = np.sum(mask_data[:, :, slice_idx] > 0)
        if hemorrhage_voxels > most_hemorrhage:
            most_hemorrhage = hemorrhage_voxels
            best_slice = slice_idx

    ct_slice = image_data[:, :, best_slice]
    mask_slice = mask_data[:, :, best_slice].astype(int)
    labels_in_slice = sorted(int(v) for v in np.unique(mask_slice) if v != 0)

    low = 40 - 80 / 2
    high = 40 + 80 / 2
    ct_display = np.clip(ct_slice, low, high)
    ct_display = (ct_display - low) / (high - low)

    cmap_colors = ["#00000000"] + [label_colors[i] for i in range(1, 6)]
    mask_cmap = mcolors.ListedColormap(cmap_colors)

    print(f"scan: {image_path.name}")
    print(f"selected slice index: {best_slice}")
    print(f"labels present in that slice: {labels_in_slice}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Scan: {image_path.name}  |  Axial slice: {best_slice}")

    axes[0].imshow(ct_display.T, cmap="gray", origin="lower")
    axes[0].set_title("CT (brain window)")
    axes[0].axis("off")

    axes[1].imshow(mask_slice.T, cmap=mask_cmap, origin="lower", vmin=0, vmax=5)
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")

    axes[2].imshow(ct_display.T, cmap="gray", origin="lower")
    mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[2].imshow(mask_overlay.T, cmap=mask_cmap, origin="lower", alpha=0.55, vmin=0, vmax=5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    legend_labels = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=label_colors[label],
            markersize=10,
            label=f"{label} = {label_names[label]}",
        )
        for label in labels_in_slice
    ]
    if legend_labels:
        fig.legend(handles=legend_labels, loc="lower center", ncol=len(legend_labels), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    output_path = project_root / "reports" / "visualize_scan.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
