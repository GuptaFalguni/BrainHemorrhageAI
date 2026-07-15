"""Pydantic schemas for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str | None = None
    device: str | None = None


class PredictionFileMap(BaseModel):
    prediction_summary_json: str = Field(alias="prediction_summary.json")
    confidence_json: str = Field(alias="confidence.json")
    volume_report_csv: str = Field(alias="volume_report.csv")
    prediction_mask_nifti: str = Field(alias="prediction_mask.nii.gz")
    overlay_png: str = Field(alias="overlay.png")
    prediction_report_md: str = Field(alias="prediction_report.md")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    scan_name: str
    job_id: str
    output_dir: str
    prediction_summary: dict[str, Any]
    confidence: dict[str, Any]
    volume_report: list[dict[str, str]]
    prediction_report_md: str
    files: dict[str, str]
    download_urls: dict[str, str]
