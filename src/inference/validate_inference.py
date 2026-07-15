"""Smoke validation for the production inference pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.metrics import NUM_CLASSES
from inference.pipeline import InferencePipeline, PROJECT_ROOT, preprocess_nifti_scan, scan_output_name
from preprocessing.preprocess_volume import IMAGES_DIR, MASKS_DIR

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACTS = (
    "prediction.nii.gz",
    "probabilities.npz",
    "prediction_overlay.png",
    "prediction_overlay_3d.png",
    "confidence.json",
    "volume_report.csv",
    "prediction_summary.json",
    "prediction_report.md",
)


def _select_labelled_scan() -> tuple[Path, Path]:
    image_files = sorted(IMAGES_DIR.glob("*.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No labelled scans found in {IMAGES_DIR}")
    image_path = image_files[0]
    mask_path = MASKS_DIR / image_path.name
    if not mask_path.exists():
        raise FileNotFoundError(f"Matching mask not found: {mask_path}")
    return image_path, mask_path


def _resolve_checkpoint() -> Path:
    candidates = [
        PROJECT_ROOT / "checkpoints" / "experiment_sanity" / "best_model.pt",
        PROJECT_ROOT / "checkpoints" / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No inference checkpoint found. Train or run the sanity experiment first.")


def validate_inference() -> None:
    """Run inference validation checks and raise on failure."""
    image_path, mask_path = _select_labelled_scan()
    checkpoint_path = _resolve_checkpoint()
    scan_name = scan_output_name(image_path)
    output_dir = PROJECT_ROOT / "reports" / "inference" / f"{scan_name}_validation"

    preprocessed, _ = preprocess_nifti_scan(image_path, mask_path=mask_path)
    input_nii = nib.load(str(image_path))

    pipeline = InferencePipeline(checkpoint_path=checkpoint_path, batch_size=2)
    result = pipeline.predict(
        input_path=image_path,
        mask_path=mask_path,
        output_dir=output_dir,
    )

    checks: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, passed, detail))
        status = "OK" if passed else "FAIL"
        message = f"  {name}: {status}"
        if detail:
            message += f" ({detail})"
        print(message)

    record(
        "preprocessing reused",
        result.shape == preprocessed.original_shape,
        f"shape={result.shape}",
    )
    record(
        "prediction shape correct",
        result.prediction_volume.shape == preprocessed.original_shape,
        str(result.prediction_volume.shape),
    )
    record(
        "affine preserved",
        np.allclose(result.affine, input_nii.affine),
    )
    record(
        "spacing preserved",
        result.spacing == preprocessed.original_spacing,
        str(result.spacing),
    )
    unique_labels = set(int(value) for value in np.unique(result.prediction_volume))
    record(
        "labels within {0..5}",
        unique_labels.issubset(set(range(NUM_CLASSES))),
        str(sorted(unique_labels)),
    )

    for artifact in REQUIRED_ARTIFACTS:
        artifact_path = output_dir / artifact
        record(f"artifact {artifact}", artifact_path.exists(), str(artifact_path))

    with (output_dir / "confidence.json").open(encoding="utf-8") as handle:
        confidence_payload = json.load(handle)
    record(
        "confidence generated",
        "study_confidence" in confidence_payload and "confidence_histogram" in confidence_payload,
    )

    with (output_dir / "volume_report.csv").open(encoding="utf-8") as handle:
        volume_lines = handle.readlines()
    record("volume calculation works", len(volume_lines) >= 2)

    with (output_dir / "prediction_summary.json").open(encoding="utf-8") as handle:
        summary_payload = json.load(handle)
    record(
        "summary generated",
        summary_payload.get("scan_name") == scan_name and "per_class_volume_ml" in summary_payload,
    )

    record(
        "optional metrics with GT",
        result.ground_truth_metrics is not None and "ground_truth_metrics" in summary_payload,
    )
    if result.ground_truth_metrics is not None:
        record(
            "optional dice available",
            0.0 <= result.ground_truth_metrics.macro_dice <= 1.0,
            f"macro_dice={result.ground_truth_metrics.macro_dice:.6f}",
        )
        record(
            "optional confusion matrix",
            len(result.ground_truth_metrics.confusion_matrix) == NUM_CLASSES,
        )

    failed = [name for name, passed, _ in checks if not passed]
    if failed:
        raise RuntimeError(f"Inference validation failed: {', '.join(failed)}")

    print(f"\nInference validation passed — output: {output_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    validate_inference()


if __name__ == "__main__":
    main()
