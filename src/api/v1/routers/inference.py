"""Synchronous prediction endpoint — orchestrates frozen pipelines only."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.v1.dependencies import (
    get_clinical_pipeline,
    get_inference_pipeline,
    get_model_registry,
    list_existing_artifacts,
    new_prediction_id,
    prediction_dir,
    publish_input_ct,
    register_prediction,
    save_upload,
)
from api.v1.schemas.response import ClinicalResponse, PredictResponse
from clinical.clinical_pipeline import ClinicalPipeline
from deployment.inference.pipeline import UnifiedInferencePipeline
from deployment.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


def _validate_nifti_filename(filename: str) -> None:
    lower = filename.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")


def _json_safe(value):
    """Convert numpy / Path values into JSON-serializable structures."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:  # noqa: BLE001
        pass
    return value


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="CT volume (.nii / .nii.gz)"),
    model_id: str | None = Form(default=None),
    mask: UploadFile | None = File(default=None),
    registry: ModelRegistry = Depends(get_model_registry),
    inference: UnifiedInferencePipeline = Depends(get_inference_pipeline),
    clinical: ClinicalPipeline = Depends(get_clinical_pipeline),
) -> PredictResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded CT filename is required.")
    _validate_nifti_filename(file.filename)

    ct_bytes = await file.read()
    if not ct_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CT file is empty.")

    resolved_id = model_id.strip() if model_id and model_id.strip() else None
    try:
        model = registry.get(resolved_id) if resolved_id else registry.get_default()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    prediction_id = new_prediction_id()
    artifact_dir = prediction_dir(prediction_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    input_path = save_upload(prediction_id, file.filename, ct_bytes)
    publish_input_ct(prediction_id, input_path)
    mask_path = None
    if mask is not None and mask.filename:
        _validate_nifti_filename(mask.filename)
        mask_bytes = await mask.read()
        if mask_bytes:
            mask_path = save_upload(prediction_id, mask.filename, mask_bytes)

    try:
        inference_result = inference.run(
            input_path,
            model_id=model.model_id,
            mask_path=mask_path,
            output_dir=artifact_dir,
        )
        clinical_result = clinical.run(
            inference_result,
            output_dir=artifact_dir,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("API v1 prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    clinical_report = Path(clinical_result.report_path)
    register_prediction(
        prediction_id,
        artifact_dir=artifact_dir,
        model_id=inference_result.model_id,
        scan_name=str((inference_result.metadata or {}).get("scan_name") or input_path.stem),
        clinical_report_path=clinical_report if clinical_report.is_file() else None,
        framework=inference_result.framework,
    )

    existing = list_existing_artifacts(artifact_dir)
    download_urls = {
        name: f"/api/v1/artifacts/{prediction_id}/{name}" for name in existing
    }
    scan_name = str((inference_result.metadata or {}).get("scan_name") or input_path.stem)

    clinical_payload = ClinicalResponse(
        severity_status=clinical_result.severity.status,
        severity_tier=clinical_result.severity.tier,
        hemorrhage_detected=clinical_result.severity.hemorrhage_detected,
        recommendations=list(clinical_result.recommendations),
        limitations=list(clinical_result.limitations),
        disclaimer=clinical_result.summary.disclaimer,
        severity=_json_safe(clinical_result.severity.to_dict()),
        summary=_json_safe(clinical_result.summary.to_dict()),
        clinical_report_path=str(clinical_report) if clinical_report.is_file() else None,
    )

    return PredictResponse(
        prediction_id=prediction_id,
        model_id=inference_result.model_id,
        framework=inference_result.framework,
        category=model.category,
        scan_name=scan_name,
        artifact_dir=str(artifact_dir),
        volumes=_json_safe(inference_result.volumes),
        confidence=_json_safe(inference_result.confidence),
        clinical=clinical_payload,
        artifacts={name: str(path) for name, path in existing.items()},
        download_urls=download_urls,
        report_url=f"/api/v1/report/{prediction_id}",
        artifacts_url=f"/api/v1/artifacts/{prediction_id}",
        metadata={
            "experiment_id": model.experiment_id,
            "checkpoint_path": (inference_result.metadata or {}).get("checkpoint_path"),
            "device": (inference_result.metadata or {}).get("device"),
            "processing_time_sec": (inference_result.metadata or {}).get("processing_time_sec"),
            "capabilities": model.capabilities.as_dict(),
            "cache_hit": False,
        },
    )


__all__ = ["router"]
