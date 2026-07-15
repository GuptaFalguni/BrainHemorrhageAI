"""Cached inference service wrapping the production inference pipeline."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.pipeline import InferencePipeline, InferenceResult, PROJECT_ROOT, scan_output_name

logger = logging.getLogger(__name__)

API_REPORTS_DIR = PROJECT_ROOT / "reports" / "api"
UPLOADS_DIR = PROJECT_ROOT / "reports" / "api" / "uploads"

API_ARTIFACT_ALIASES: dict[str, str] = {
    "prediction_mask.nii.gz": "prediction.nii.gz",
    "overlay.png": "prediction_overlay.png",
}

REQUIRED_API_FILES = (
    "prediction_summary.json",
    "confidence.json",
    "volume_report.csv",
    "prediction_mask.nii.gz",
    "overlay.png",
    "prediction_report.md",
)

_pipeline: InferencePipeline | None = None


@dataclass(frozen=True)
class PredictionArtifacts:
    """API-facing prediction artifacts after inference completes."""

    scan_name: str
    job_id: str
    output_dir: Path
    prediction_summary: dict[str, Any]
    confidence: dict[str, Any]
    volume_report: list[dict[str, str]]
    prediction_report_md: str
    files: dict[str, Path]
    result: InferenceResult


def resolve_checkpoint_path() -> Path:
    """Resolve checkpoint from environment or project defaults."""
    env_path = os.environ.get("BRAIN_HEMORRHAGE_CHECKPOINT")
    if env_path:
        candidate = Path(env_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Checkpoint from environment not found: {candidate}")

    candidates = [
        PROJECT_ROOT / "checkpoints" / "experiment_sanity" / "best_model.pt",
        PROJECT_ROOT / "checkpoints" / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("No inference checkpoint found.")


def get_pipeline() -> InferencePipeline:
    """Return the cached inference pipeline (loaded once per process)."""
    global _pipeline
    if _pipeline is None:
        checkpoint_path = resolve_checkpoint_path()
        logger.info("Loading inference model from %s", checkpoint_path)
        _pipeline = InferencePipeline(
            checkpoint_path=checkpoint_path,
            reports_dir=API_REPORTS_DIR,
        )
    return _pipeline


def reset_pipeline_cache() -> None:
    """Clear cached pipeline (testing only)."""
    global _pipeline
    _pipeline = None


def publish_api_aliases(output_dir: Path) -> dict[str, Path]:
    """Expose API-facing filenames without changing inference outputs."""
    published: dict[str, Path] = {}
    for api_name, source_name in API_ARTIFACT_ALIASES.items():
        source = output_dir / source_name
        destination = output_dir / api_name
        if not source.exists():
            raise FileNotFoundError(f"Expected inference artifact missing: {source}")
        shutil.copy2(source, destination)
        published[api_name] = destination
    return published


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_volume_report(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_prediction(
    input_path: Path,
    mask_path: Path | None = None,
    output_dir: Path | None = None,
) -> PredictionArtifacts:
    """Run cached inference and publish API artifact aliases."""
    pipeline = get_pipeline()
    scan_name = scan_output_name(input_path)
    job_id = uuid.uuid4().hex[:12]
    destination = output_dir or (API_REPORTS_DIR / f"{scan_name}_{job_id}")
    destination.mkdir(parents=True, exist_ok=True)

    result = pipeline.predict(
        input_path=input_path,
        mask_path=mask_path,
        output_dir=destination,
    )

    alias_paths = publish_api_aliases(destination)
    summary_path = destination / "prediction_summary.json"
    confidence_path = destination / "confidence.json"
    volume_path = destination / "volume_report.csv"
    report_path = destination / "prediction_report.md"

    files = {
        "prediction_summary.json": summary_path,
        "confidence.json": confidence_path,
        "volume_report.csv": volume_path,
        "prediction_report.md": report_path,
        "prediction_mask.nii.gz": alias_paths["prediction_mask.nii.gz"],
        "overlay.png": alias_paths["overlay.png"],
    }

    return PredictionArtifacts(
        scan_name=scan_name,
        job_id=job_id,
        output_dir=destination,
        prediction_summary=_read_json(summary_path),
        confidence=_read_json(confidence_path),
        volume_report=_read_volume_report(volume_path),
        prediction_report_md=report_path.read_text(encoding="utf-8"),
        files=files,
        result=result,
    )


def save_upload(upload_name: str, content: bytes) -> Path:
    """Persist an uploaded file for inference."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload_name).name
    destination = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    destination.write_bytes(content)
    return destination
