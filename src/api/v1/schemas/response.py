"""Response schemas for API v1."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelCapabilitiesResponse(BaseModel):
    supports_overlay: bool
    supports_volume: bool
    supports_confidence: bool
    supports_multiclass: bool
    supports_severity: bool
    supports_uncertainty: bool


class ModelResponse(BaseModel):
    model_id: str
    display_name: str
    experiment_id: str
    category: str
    framework: str
    version: str
    default: bool
    description: str = ""
    paper: str = ""
    training_dataset: str = ""
    macro_dice: float | None = None
    volume_error_mean_ml: float | None = None
    speed: str = ""
    limitations: str = ""
    recommended_use: str = ""
    model_card: str = ""
    capabilities: ModelCapabilitiesResponse


class ModelsListResponse(BaseModel):
    default_model_id: str
    models: list[ModelResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    available_models: list[str]
    default_model_id: str | None = None
    device: str
    gpu_available: bool
    python_version: str
    pytorch_version: str
    monai_version: str | None = None
    registry_loaded: bool = True


class ClinicalResponse(BaseModel):
    severity_status: str
    severity_tier: str | None = None
    hemorrhage_detected: bool
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str
    severity: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    clinical_report_path: str | None = None


class ArtifactFileInfo(BaseModel):
    name: str
    exists: bool
    path: str | None = None
    download_url: str | None = None


class ArtifactResponse(BaseModel):
    prediction_id: str
    artifact_dir: str
    files: list[ArtifactFileInfo]


class PredictResponse(BaseModel):
    prediction_id: str
    model_id: str
    framework: str
    category: str | None = None
    scan_name: str
    artifact_dir: str
    volumes: dict[str, Any]
    confidence: dict[str, Any]
    clinical: ClinicalResponse
    artifacts: dict[str, str]
    download_urls: dict[str, str]
    report_url: str
    artifacts_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportResponse(BaseModel):
    prediction_id: str
    report_type: str
    path: str
    markdown: str


__all__ = [
    "ArtifactFileInfo",
    "ArtifactResponse",
    "ClinicalResponse",
    "HealthResponse",
    "ModelCapabilitiesResponse",
    "ModelResponse",
    "ModelsListResponse",
    "PredictResponse",
    "ReportResponse",
]
