"""Shared API v1 dependencies — single registry / pipeline instances."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical.clinical_pipeline import ClinicalPipeline
from deployment.inference.pipeline import UnifiedInferencePipeline
from deployment.model_registry import ModelRegistry
from deployment.registry import load_deployment_registries

PROJECT_ROOT = Path(__file__).resolve().parents[3]
API_V1_ROOT = PROJECT_ROOT / "reports" / "api_v1"
PREDICTIONS_ROOT = API_V1_ROOT / "predictions"
INDEX_PATH = API_V1_ROOT / "index.json"

# Filenames the API may expose (must already exist on disk).
# Keep in sync with docs/platform/api_v1_contract.md
EXPOSED_ARTIFACTS: tuple[str, ...] = (
    "input_ct.nii.gz",
    "overlay.png",
    "prediction_overlay.png",
    "prediction_overlay_3d.png",
    "prediction_mask.nii.gz",
    "prediction.nii.gz",
    "volume_report.csv",
    "confidence.json",
    "prediction_summary.json",
    "prediction_report.md",
    "clinical_report.md",
    "probabilities.npz",
    "meta.json",
)

# Canonical volume keys on UnifiedInferenceResult / PredictResponse.volumes
VOLUME_KEYS: frozenset[str] = frozenset(
    {"total_volume_ml", "per_class_volume_ml", "classes_present", "spacing"}
)

# Canonical confidence keys
CONFIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "study_confidence",
        "per_class_confidence",
        "mean_softmax_confidence",
        "confidence_histogram",
    }
)

_lock = threading.Lock()
_model_registry: ModelRegistry | None = None
_inference_pipeline: UnifiedInferencePipeline | None = None
_clinical_pipeline: ClinicalPipeline | None = None


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        with _lock:
            if _model_registry is None:
                _experiments, models = load_deployment_registries(project_root=PROJECT_ROOT)
                _model_registry = models
    return _model_registry


def get_inference_pipeline() -> UnifiedInferencePipeline:
    global _inference_pipeline
    if _inference_pipeline is None:
        with _lock:
            if _inference_pipeline is None:
                registry = get_model_registry()
                _inference_pipeline = UnifiedInferencePipeline(
                    model_registry=registry,
                    project_root=PROJECT_ROOT,
                )
    return _inference_pipeline


def get_clinical_pipeline() -> ClinicalPipeline:
    global _clinical_pipeline
    if _clinical_pipeline is None:
        with _lock:
            if _clinical_pipeline is None:
                _clinical_pipeline = ClinicalPipeline()
    return _clinical_pipeline


def new_prediction_id() -> str:
    return uuid.uuid4().hex


def prediction_dir(prediction_id: str) -> Path:
    return PREDICTIONS_ROOT / prediction_id


def _read_index() -> dict[str, Any]:
    if not INDEX_PATH.is_file():
        return {"predictions": {}}
    with INDEX_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {"predictions": {}}
    predictions = data.get("predictions")
    if not isinstance(predictions, dict):
        data["predictions"] = {}
    return data


def _write_index(data: dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp.replace(INDEX_PATH)


def register_prediction(
    prediction_id: str,
    *,
    artifact_dir: Path,
    model_id: str,
    scan_name: str,
    clinical_report_path: Path | None,
    framework: str,
) -> dict[str, Any]:
    record = {
        "prediction_id": prediction_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "model_id": model_id,
        "scan_name": scan_name,
        "framework": framework,
        "clinical_report_path": str(clinical_report_path) if clinical_report_path else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        data = _read_index()
        data.setdefault("predictions", {})[prediction_id] = record
        meta_path = Path(artifact_dir) / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        _write_index(data)
    return record


def get_prediction_record(prediction_id: str) -> dict[str, Any] | None:
    with _lock:
        data = _read_index()
        record = data.get("predictions", {}).get(prediction_id)
        if isinstance(record, dict):
            return record
    # Fall back to per-run meta.json
    meta_path = prediction_dir(prediction_id) / "meta.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            return loaded
    return None


def save_upload(prediction_id: str, filename: str, content: bytes) -> Path:
    destination = prediction_dir(prediction_id) / "uploads"
    destination.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    path = destination / safe_name
    path.write_bytes(content)
    return path


def publish_input_ct(prediction_id: str, upload_path: Path) -> Path:
    """Copy the uploaded CT to a stable downloadable name ``input_ct.nii.gz``."""
    artifact_root = prediction_dir(prediction_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    published = artifact_root / "input_ct.nii.gz"
    if upload_path.resolve() != published.resolve():
        shutil.copy2(upload_path, published)
    return published


def list_existing_artifacts(artifact_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in EXPOSED_ARTIFACTS:
        path = artifact_dir / name
        if path.is_file():
            found[name] = path
    return found


__all__ = [
    "API_V1_ROOT",
    "CONFIDENCE_KEYS",
    "EXPOSED_ARTIFACTS",
    "INDEX_PATH",
    "PREDICTIONS_ROOT",
    "PROJECT_ROOT",
    "VOLUME_KEYS",
    "get_clinical_pipeline",
    "get_inference_pipeline",
    "get_model_registry",
    "get_prediction_record",
    "get_project_root",
    "list_existing_artifacts",
    "new_prediction_id",
    "prediction_dir",
    "publish_input_ct",
    "register_prediction",
    "save_upload",
]
