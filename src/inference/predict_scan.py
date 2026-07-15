"""CLI entry point for single-scan inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from inference.pipeline import InferencePipeline, PROJECT_ROOT

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run BrainHemorrhageAI inference on one CT scan.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CT NIfTI (.nii / .nii.gz).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--mask", type=Path, default=None, help="Optional ground-truth mask NIfTI.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--model-config", type=Path, default=None, help="Optional model config YAML.")
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "preprocessing.yaml",
        help="Preprocessing config YAML.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cpu/cuda).")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    return parser.parse_args()


def main() -> None:
    """Run inference from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    input_path = args.input
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    mask_path = args.mask
    if mask_path is not None and not mask_path.is_absolute():
        mask_path = (Path.cwd() / mask_path).resolve()

    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        model_config_path=args.model_config,
        preprocessing_config_path=args.preprocessing_config,
        device=args.device,
        batch_size=args.batch_size,
    )
    result = pipeline.predict(
        input_path=input_path,
        mask_path=mask_path,
        output_dir=args.output_dir,
    )
    logger.info(
        "Prediction saved to %s (classes=%s, total_volume=%.3f mL)",
        result.output_dir,
        result.classes_present,
        result.total_volume_ml,
    )


if __name__ == "__main__":
    main()
