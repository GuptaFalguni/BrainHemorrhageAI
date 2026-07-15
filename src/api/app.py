"""FastAPI application for BrainHemorrhageAI production inference."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import HealthResponse, PredictionResponse
from api.service import (
    REQUIRED_API_FILES,
    get_pipeline,
    resolve_checkpoint_path,
    run_prediction,
    save_upload,
)

logger = logging.getLogger(__name__)


def _validate_nifti_filename(filename: str) -> None:
    lower = filename.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Brain Hemorrhage AI",
        description="Production inference API for BHSD brain CT hemorrhage segmentation.",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            checkpoint = resolve_checkpoint_path()
            pipeline = get_pipeline()
            return HealthResponse(
                status="ok",
                model_loaded=True,
                checkpoint=str(checkpoint),
                device=str(pipeline.device),
            )
        except FileNotFoundError as error:
            return HealthResponse(status="degraded", model_loaded=False, checkpoint=str(error))

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        mask: UploadFile | None = File(default=None),
    ) -> PredictionResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded CT filename is required.")
        _validate_nifti_filename(file.filename)

        ct_bytes = await file.read()
        if not ct_bytes:
            raise HTTPException(status_code=400, detail="Uploaded CT file is empty.")

        input_path = save_upload(file.filename, ct_bytes)
        mask_path = None
        if mask is not None and mask.filename:
            _validate_nifti_filename(mask.filename)
            mask_bytes = await mask.read()
            if mask_bytes:
                mask_path = save_upload(mask.filename, mask_bytes)

        try:
            artifacts = run_prediction(input_path=input_path, mask_path=mask_path)
        except FileNotFoundError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        except Exception as error:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error

        for required_name in REQUIRED_API_FILES:
            if required_name not in artifacts.files or not artifacts.files[required_name].exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required artifact: {required_name}",
                )

        download_urls = {
            name: f"/files/{artifacts.job_id}/{name}"
            for name in artifacts.files
        }

        return PredictionResponse(
            scan_name=artifacts.scan_name,
            job_id=artifacts.job_id,
            output_dir=str(artifacts.output_dir),
            prediction_summary=artifacts.prediction_summary,
            confidence=artifacts.confidence,
            volume_report=artifacts.volume_report,
            prediction_report_md=artifacts.prediction_report_md,
            files={name: str(path) for name, path in artifacts.files.items()},
            download_urls=download_urls,
        )

    @application.get("/files/{job_id}/{filename}")
    def download_file(job_id: str, filename: str) -> FileResponse:
        if ".." in job_id or ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid path.")
        if filename not in REQUIRED_API_FILES and filename not in {
            "prediction.nii.gz",
            "prediction_overlay.png",
            "probabilities.npz",
        }:
            raise HTTPException(status_code=404, detail="Unknown file.")

        from api.service import API_REPORTS_DIR

        matches = list(API_REPORTS_DIR.glob(f"*_{job_id}"))
        if not matches:
            raise HTTPException(status_code=404, detail="Job not found.")
        output_dir = matches[0]
        file_path = output_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found.")
        return FileResponse(path=file_path, filename=filename)

    return application


app = create_app()
