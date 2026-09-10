"""
Preprocess a single BHSD CT volume and matching segmentation mask.

Intensity steps:
  native HU → clip/window → normalization (when train stats are configured)

Spatial resampling is not applied here.
Original spacing, shape, and affine are preserved for clinical volume calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "preprocessing.yaml"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "images"
MASKS_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "ground truths"

VALID_LABELS = frozenset({0, 1, 2, 3, 4, 5})
SUPPORTED_NORMALIZATION = frozenset({"zscore", "minmax", "none"})


@dataclass(frozen=True)
class PreprocessedVolume:
    """Intensity-preprocessed volume with preserved native geometry metadata."""

    processed_image: np.ndarray
    processed_mask: np.ndarray
    original_spacing: tuple[float, float, float]
    original_shape: tuple[int, ...]
    affine: np.ndarray
    filename: str


def load_preprocessing_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {path}")
    return config


def resolve_clip_bounds(config: dict[str, Any]) -> tuple[float, float]:
    clip_min = config.get("clip_min")
    clip_max = config.get("clip_max")
    if clip_min is not None and clip_max is not None:
        low, high = float(clip_min), float(clip_max)
    else:
        window_level = config.get("window_level")
        window_width = config.get("window_width")
        if window_level is None or window_width is None:
            raise ValueError(
                "Configure clip_min and clip_max, or window_level and window_width "
                "in config/preprocessing.yaml (values pending literature review)."
            )
        half = float(window_width) / 2.0
        low = float(window_level) - half
        high = float(window_level) + half

    if low >= high:
        raise ValueError(f"Invalid clip bounds: min ({low}) must be less than max ({high}).")
    return low, high


def load_volume_pair(filename: str) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    image_path = IMAGES_DIR / filename
    mask_path = MASKS_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(f"CT not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    return image_nii, mask_nii


def extract_native_metadata(image_nii: nib.Nifti1Image) -> tuple[tuple[float, float, float], tuple[int, ...], np.ndarray]:
    spacing = tuple(float(v) for v in image_nii.header.get_zooms()[:3])
    shape = tuple(int(v) for v in image_nii.shape[:3])
    affine = np.array(image_nii.affine, copy=True)
    return spacing, shape, affine


def apply_intensity_clip(image_hu: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    return np.clip(image_hu, clip_min, clip_max).astype(np.float32)


def clip_ct_volume(filename: str, config: dict[str, Any]) -> np.ndarray:
    """Load a CT volume and apply HU clip/window only (no normalization)."""
    image_nii, _ = load_volume_pair(filename)
    image_hu = image_nii.get_fdata().astype(np.float32)
    clip_min, clip_max = resolve_clip_bounds(config)
    return apply_intensity_clip(image_hu, clip_min, clip_max)


def apply_normalization(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    method = str(config.get("normalization_method", "none")).lower()
    if method not in SUPPORTED_NORMALIZATION:
        raise ValueError(f"Unsupported normalization_method: {method}")

    if method == "none":
        logger.info("Normalization skipped (normalization_method=none).")
        return image

    if method == "zscore":
        mean = config.get("train_mean")
        std = config.get("train_std")
        if mean is None or std is None:
            logger.info(
                "Normalization skipped: train_mean and train_std not set "
                "(compute from training split after literature review)."
            )
            return image
        std = float(std)
        if std <= 0:
            raise ValueError(f"train_std must be positive, got {std}.")
        return ((image - float(mean)) / std).astype(np.float32)

    clip_min, clip_max = resolve_clip_bounds(config)
    scale = clip_max - clip_min
    if scale <= 0:
        raise ValueError("Cannot apply minmax normalization with zero-width clip range.")
    normalized = (image - clip_min) / scale
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def prepare_mask(mask_hu: np.ndarray) -> np.ndarray:
    mask_int = np.rint(mask_hu).astype(np.int64)
    return mask_int


def validate_preprocessed(
    processed_image: np.ndarray,
    processed_mask: np.ndarray,
    original_shape: tuple[int, ...],
    original_spacing: tuple[float, float, float],
) -> None:
    if processed_image.shape != original_shape:
        raise ValueError(
            f"Processed image shape {processed_image.shape} != original {original_shape}."
        )
    if processed_mask.shape != original_shape:
        raise ValueError(
            f"Processed mask shape {processed_mask.shape} != original {original_shape}."
        )
    if any(value <= 0 for value in original_spacing):
        raise ValueError(f"Invalid voxel spacing (must be > 0): {original_spacing}.")

    if np.isnan(processed_image).any():
        raise ValueError("Processed image contains NaN values.")
    if np.isnan(processed_mask).any():
        raise ValueError("Processed mask contains NaN values.")

    unique_labels = set(np.unique(processed_mask).astype(int).tolist())
    invalid = unique_labels - VALID_LABELS
    if invalid:
        raise ValueError(f"Mask contains invalid labels: {sorted(invalid)}")


def preprocess_volume(
    filename: str,
    config_path: Path | None = None,
) -> PreprocessedVolume:
    """
    Load, intensity-preprocess, and validate one CT/mask pair in native space.

    Parameters
    ----------
    filename:
        Basename shared by image and mask (e.g. ``ID_xxx.nii.gz``).
    config_path:
        Path to preprocessing YAML; defaults to ``config/preprocessing.yaml``.
    """
    config = load_preprocessing_config(config_path)
    logger.info("Preprocessing %s", filename)

    image_nii, mask_nii = load_volume_pair(filename)
    image_hu = image_nii.get_fdata().astype(np.float32)
    mask_raw = mask_nii.get_fdata()

    original_spacing, original_shape, affine = extract_native_metadata(image_nii)
    if tuple(int(v) for v in mask_raw.shape[:3]) != original_shape:
        raise ValueError(
            f"Shape mismatch for {filename}: image {original_shape}, mask {mask_raw.shape[:3]}."
        )

    clip_min, clip_max = resolve_clip_bounds(config)
    logger.debug("Applying HU clip [%s, %s]", clip_min, clip_max)
    clipped = apply_intensity_clip(image_hu, clip_min, clip_max)

    normalized = apply_normalization(clipped, config)
    processed_mask = prepare_mask(mask_raw)

    validate_preprocessed(normalized, processed_mask, original_shape, original_spacing)

    logger.info(
        "Preprocessed %s — shape %s, spacing %s, labels %s",
        filename,
        original_shape,
        original_spacing,
        sorted(int(v) for v in np.unique(processed_mask)),
    )

    return PreprocessedVolume(
        processed_image=normalized,
        processed_mask=processed_mask,
        original_spacing=original_spacing,
        original_shape=original_shape,
        affine=affine,
        filename=filename,
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    _configure_logging()
    first_image = sorted(IMAGES_DIR.glob("*.nii.gz"))[0].name
    try:
        result = preprocess_volume(first_image)
        print(f"filename: {result.filename}")
        print(f"original_shape: {result.original_shape}")
        print(f"original_spacing: {result.original_spacing}")
        print(f"processed_image: dtype={result.processed_image.dtype}, "
              f"min={result.processed_image.min():.4f}, max={result.processed_image.max():.4f}")
        print(f"processed_mask: dtype={result.processed_mask.dtype}, "
              f"labels={sorted(int(v) for v in np.unique(result.processed_mask))}")
    except ValueError as error:
        logger.error("%s", error)
        logger.error(
            "Set clip_min/clip_max or window_level/window_width in config/preprocessing.yaml "
            "after literature review."
        )
        raise SystemExit(1) from error
