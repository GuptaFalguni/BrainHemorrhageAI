"""
Generate a reproducible train/validation/test split for the BHSD labeled dataset.

Implements docs/architecture/data_split_design.md:
  - 70 / 15 / 15 split
  - random seed 42
  - multi-label stratification (iterative stratification)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "images"
MASKS_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "ground truths"
SPLITS_CSV = PROJECT_ROOT / "data" / "metadata" / "splits.csv"
SPLIT_CONFIG_JSON = PROJECT_ROOT / "data" / "metadata" / "split_config.json"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
HEMORRHAGE_LABELS = (1, 2, 3, 4, 5)
LABEL_NAMES = {1: "EDH", 2: "SDH", 3: "SAH", 4: "IPH", 5: "IVH"}
STRATIFICATION_METHOD = (
    "iterative_multi_label_stratification (MultilabelStratifiedShuffleSplit)"
)


def collect_paired_filenames() -> list[str]:
    """Return sorted image filenames that have a matching mask."""
    image_names = sorted(p.name for p in IMAGES_DIR.glob("*.nii.gz"))
    mask_names = {p.name for p in MASKS_DIR.glob("*.nii.gz")}
    return [name for name in image_names if name in mask_names]


def verify_dataset_integrity(filenames: list[str]) -> None:
    """Raise ValueError if pairing or uniqueness checks fail."""
    image_names = sorted(p.name for p in IMAGES_DIR.glob("*.nii.gz"))
    mask_names = sorted(p.name for p in MASKS_DIR.glob("*.nii.gz"))

    if len(image_names) != len(set(image_names)):
        duplicates = {n for n in image_names if image_names.count(n) > 1}
        raise ValueError(f"Duplicate image filenames: {sorted(duplicates)}")

    if len(mask_names) != len(set(mask_names)):
        duplicates = {n for n in mask_names if mask_names.count(n) > 1}
        raise ValueError(f"Duplicate mask filenames: {sorted(duplicates)}")

    missing_masks = sorted(set(image_names) - set(mask_names))
    if missing_masks:
        raise ValueError(f"Images without matching masks ({len(missing_masks)}): {missing_masks[:5]} ...")

    extra_masks = sorted(set(mask_names) - set(image_names))
    if extra_masks:
        raise ValueError(f"Masks without matching images ({len(extra_masks)}): {extra_masks[:5]} ...")

    if filenames != image_names:
        raise ValueError("Paired filename list does not match the full image inventory.")


def build_multilabel_matrix(filenames: list[str]) -> np.ndarray:
    """Build binary presence matrix (n_scans x 5) for hemorrhage labels 1-5."""
    matrix = np.zeros((len(filenames), len(HEMORRHAGE_LABELS)), dtype=int)
    for row, filename in enumerate(filenames):
        mask_data = nib.load(str(MASKS_DIR / filename)).get_fdata()
        present = set(int(v) for v in np.unique(mask_data) if v in HEMORRHAGE_LABELS)
        for col, label in enumerate(HEMORRHAGE_LABELS):
            matrix[row, col] = int(label in present)
    return matrix


def create_stratified_split(
    filenames: list[str], label_matrix: np.ndarray
) -> dict[str, list[str]]:
    """
    70/15/15 split via two-step iterative multi-label stratification.

    Step 1: 70% train vs 30% holdout (val+test).
    Step 2: split holdout 50/50 into val and test (15% each overall).
    """
    indices = np.arange(len(filenames))
    holdout_ratio = VAL_RATIO + TEST_RATIO

    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=holdout_ratio, random_state=RANDOM_SEED
    )
    train_idx, holdout_idx = next(first_splitter.split(indices, label_matrix))

    holdout_labels = label_matrix[holdout_idx]
    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=RANDOM_SEED
    )
    val_local_idx, test_local_idx = next(second_splitter.split(holdout_idx, holdout_labels))

    split_names = {
        "train": [filenames[i] for i in train_idx],
        "val": [filenames[i] for i in holdout_idx[val_local_idx]],
        "test": [filenames[i] for i in holdout_idx[test_local_idx]],
    }
    return split_names


def split_assignments_to_dataframe(split_names: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        {"filename": filename, "split": split_name}
        for split_name in ("train", "val", "test")
        for filename in sorted(split_names[split_name])
    ]
    return pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)


def subtype_counts(label_matrix: np.ndarray, filenames: list[str], subset: list[str]) -> dict[int, int]:
    filename_to_row = {name: i for i, name in enumerate(filenames)}
    rows = [filename_to_row[name] for name in subset]
    subset_matrix = label_matrix[rows]
    return {label: int(subset_matrix[:, col].sum()) for col, label in enumerate(HEMORRHAGE_LABELS)}


def validate_split(
    df: pd.DataFrame,
    filenames: list[str],
    label_matrix: np.ndarray,
    split_names: dict[str, list[str]],
) -> None:
    errors: list[str] = []

    if len(df) != len(filenames):
        errors.append(f"Expected {len(filenames)} rows, got {len(df)}.")

    if df["filename"].duplicated().any():
        dupes = df.loc[df["filename"].duplicated(), "filename"].tolist()
        errors.append(f"Duplicate filenames in split output: {dupes}")

    expected_splits = {"train", "val", "test"}
    if set(df["split"].unique()) != expected_splits:
        errors.append(f"Unexpected split labels: {set(df['split'].unique())}")

    assigned = set(df["filename"])
    if assigned != set(filenames):
        errors.append("Split filenames do not match the full dataset inventory.")

    for split_name in ("val", "test"):
        counts = subtype_counts(label_matrix, filenames, split_names[split_name])
        for label in HEMORRHAGE_LABELS:
            if counts[label] == 0:
                errors.append(
                    f"Label {label} ({LABEL_NAMES[label]}) missing from {split_name} split."
                )

    if errors:
        raise ValueError("Split validation failed:\n- " + "\n- ".join(errors))


def save_artifacts(df: pd.DataFrame, split_names: dict[str, list[str]]) -> None:
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS_CSV, index=False)

    config = {
        "split_ratio": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "random_seed": RANDOM_SEED,
        "stratification_method": STRATIFICATION_METHOD,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scans": len(df),
        "train_count": len(split_names["train"]),
        "validation_count": len(split_names["val"]),
        "test_count": len(split_names["test"]),
    }
    SPLIT_CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def print_summary(
    df: pd.DataFrame,
    filenames: list[str],
    label_matrix: np.ndarray,
    split_names: dict[str, list[str]],
) -> None:
    print("=== Dataset split summary ===")
    print(f"Total scans: {len(filenames)}")
    for split_name in ("train", "val", "test"):
        subset = split_names[split_name]
        n = len(subset)
        print(f"\n{split_name}: {n} scans")
        counts = subtype_counts(label_matrix, filenames, subset)
        for label in HEMORRHAGE_LABELS:
            count = counts[label]
            pct = (100.0 * count / n) if n else 0.0
            print(f"  label {label} ({LABEL_NAMES[label]}): {count} scans ({pct:.1f}%)")

    print("\n=== Subtype coverage (val / test) ===")
    for split_name in ("val", "test"):
        counts = subtype_counts(label_matrix, filenames, split_names[split_name])
        missing = [LABEL_NAMES[l] for l in HEMORRHAGE_LABELS if counts[l] == 0]
        status = "OK" if not missing else f"MISSING: {missing}"
        print(f"  {split_name}: {status}")

    print("\n=== Integrity checks ===")
    print(f"Every filename appears exactly once: {df['filename'].is_unique and len(df) == len(filenames)}")
    print(f"Saved: {SPLITS_CSV}")
    print(f"Saved: {SPLIT_CONFIG_JSON}")


def main() -> None:
    print(f"Stratification: {STRATIFICATION_METHOD}")
    print(
        "Using iterative-stratification (iterstrat) because sklearn's stratified "
        "splitters only support single-label targets, not multi-label BHSD scans."
    )

    filenames = collect_paired_filenames()
    verify_dataset_integrity(filenames)

    label_matrix = build_multilabel_matrix(filenames)
    split_names = create_stratified_split(filenames, label_matrix)
    df = split_assignments_to_dataframe(split_names)

    validate_split(df, filenames, label_matrix, split_names)
    save_artifacts(df, split_names)
    print_summary(df, filenames, label_matrix, split_names)


if __name__ == "__main__":
    main()
