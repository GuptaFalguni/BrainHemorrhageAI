"""
Compute training-set intensity statistics for z-score normalization.

Reads TRAIN split only from data/metadata/splits.csv, applies clip/window (no normalization),
and writes train_mean / train_std to config/preprocessing.yaml without changing other keys.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.preprocess_volume import (
    DEFAULT_CONFIG_PATH,
    clip_ct_volume,
    load_preprocessing_config,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = PROJECT_ROOT / "data" / "metadata" / "splits.csv"


@dataclass(frozen=True)
class IntensityStatistics:
    mean: float
    std: float
    minimum: float
    maximum: float
    voxel_count: int
    scan_count: int


def load_train_filenames(splits_path: Path | None = None) -> list[str]:
    path = splits_path or SPLITS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    df = pd.read_csv(path)
    if "filename" not in df.columns or "split" not in df.columns:
        raise ValueError(f"Expected columns 'filename' and 'split' in {path}")

    train_df = df[df["split"] == "train"]
    if train_df.empty:
        raise ValueError("No training scans found in splits.csv.")

    filenames = sorted(train_df["filename"].astype(str).tolist())
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate filenames in training split.")

    return filenames


def accumulate_clipped_voxels(clipped: np.ndarray, state: dict[str, float]) -> None:
    """Update running sum, sum-of-squares, count, min, and max."""
    values = clipped.astype(np.float64, copy=False).ravel()
    if values.size == 0:
        return

    if np.isnan(values).any():
        raise ValueError("Clipped volume contains NaN values.")

    state["count"] += float(values.size)
    state["sum"] += float(values.sum())
    state["sum_sq"] += float(np.square(values).sum())
    state["min"] = min(state["min"], float(values.min()))
    state["max"] = max(state["max"], float(values.max()))


def compute_intensity_statistics(
    filenames: list[str],
    config_path: Path | None = None,
) -> IntensityStatistics:
    config = load_preprocessing_config(config_path)
    state = {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": math.inf, "max": -math.inf}

    for index, filename in enumerate(filenames, start=1):
        logger.info("Processing train scan %s/%s: %s", index, len(filenames), filename)
        clipped = clip_ct_volume(filename, config)
        accumulate_clipped_voxels(clipped, state)

    if state["count"] == 0:
        raise ValueError("No voxels accumulated from training scans.")

    mean = state["sum"] / state["count"]
    variance = (state["sum_sq"] / state["count"]) - (mean ** 2)
    if variance <= 0:
        raise ValueError(f"Non-positive variance computed: {variance}.")

    std = math.sqrt(variance)
    if math.isnan(mean) or math.isnan(std):
        raise ValueError("Computed mean or std is NaN.")

    return IntensityStatistics(
        mean=mean,
        std=std,
        minimum=state["min"],
        maximum=state["max"],
        voxel_count=int(state["count"]),
        scan_count=len(filenames),
    )


def update_train_stats_in_config(
    config_path: Path,
    mean: float,
    std: float,
) -> None:
    """Update train_mean and train_std in YAML, preserving comments and other keys."""
    text = config_path.read_text(encoding="utf-8")

    mean_line = f"train_mean: {mean:.8g}"
    std_line = f"train_std: {std:.8g}"

    if re.search(r"^train_mean:", text, flags=re.MULTILINE):
        text = re.sub(r"^train_mean:.*$", mean_line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + f"\n{mean_line}\n"

    if re.search(r"^train_std:", text, flags=re.MULTILINE):
        text = re.sub(r"^train_std:.*$", std_line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip() + f"\n{std_line}\n"

    config_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def validate_statistics(stats: IntensityStatistics) -> None:
    if math.isnan(stats.mean) or math.isnan(stats.std):
        raise ValueError("Statistics contain NaN values.")
    if stats.std <= 0:
        raise ValueError(f"Standard deviation must be > 0, got {stats.std}.")


def print_summary(stats: IntensityStatistics) -> None:
    print("=== Training-set intensity statistics (post-clip) ===")
    print(f"Scans processed: {stats.scan_count}")
    print(f"Total voxels: {stats.voxel_count}")
    print(f"Mean: {stats.mean:.6f}")
    print(f"Standard deviation: {stats.std:.6f}")
    print(f"Min: {stats.minimum:.6f}")
    print(f"Max: {stats.maximum:.6f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    train_filenames = load_train_filenames()
    logger.info("Loaded %s training scans from %s", len(train_filenames), SPLITS_CSV)

    stats = compute_intensity_statistics(train_filenames)
    validate_statistics(stats)
    update_train_stats_in_config(DEFAULT_CONFIG_PATH, stats.mean, stats.std)

    print_summary(stats)
    logger.info("Updated train_mean and train_std in %s", DEFAULT_CONFIG_PATH)


if __name__ == "__main__":
    main()
