"""Report and artifact download endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from api.v1.dependencies import EXPOSED_ARTIFACTS, get_prediction_record, list_existing_artifacts
from api.v1.schemas.response import ArtifactFileInfo, ArtifactResponse, ReportResponse

router = APIRouter(tags=["reports"])


def _artifact_dir_for(prediction_id: str) -> Path:
    if ".." in prediction_id or "/" in prediction_id or "\\" in prediction_id:
        raise HTTPException(status_code=400, detail="Invalid prediction_id.")
    record = get_prediction_record(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    artifact_dir = Path(str(record["artifact_dir"]))
    if not artifact_dir.is_dir():
        raise HTTPException(status_code=404, detail="Artifact directory not found.")
    return artifact_dir


@router.get("/report/{prediction_id}", response_model=ReportResponse)
def get_report(prediction_id: str) -> ReportResponse:
    artifact_dir = _artifact_dir_for(prediction_id)
    record = get_prediction_record(prediction_id) or {}
    candidates = []
    if record.get("clinical_report_path"):
        candidates.append(Path(str(record["clinical_report_path"])))
    candidates.extend(
        [
            artifact_dir / "clinical_report.md",
            artifact_dir / "prediction_report.md",
        ]
    )
    report_path = next((path for path in candidates if path.is_file()), None)
    if report_path is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    markdown = report_path.read_text(encoding="utf-8")
    report_type = "clinical" if report_path.name == "clinical_report.md" else "prediction"
    return ReportResponse(
        prediction_id=prediction_id,
        report_type=report_type,
        path=str(report_path),
        markdown=markdown,
    )


@router.get("/report/{prediction_id}/raw", response_class=PlainTextResponse)
def get_report_raw(prediction_id: str) -> PlainTextResponse:
    payload = get_report(prediction_id)
    return PlainTextResponse(content=payload.markdown, media_type="text/markdown")


@router.get("/artifacts/{prediction_id}", response_model=ArtifactResponse)
def list_artifacts(prediction_id: str) -> ArtifactResponse:
    artifact_dir = _artifact_dir_for(prediction_id)
    existing = list_existing_artifacts(artifact_dir)
    files = []
    for name in EXPOSED_ARTIFACTS:
        path = existing.get(name)
        files.append(
            ArtifactFileInfo(
                name=name,
                exists=path is not None,
                path=str(path) if path else None,
                download_url=(
                    f"/api/v1/artifacts/{prediction_id}/{name}" if path is not None else None
                ),
            )
        )
    return ArtifactResponse(
        prediction_id=prediction_id,
        artifact_dir=str(artifact_dir),
        files=files,
    )


@router.get("/artifacts/{prediction_id}/{filename}")
def download_artifact(prediction_id: str, filename: str) -> FileResponse:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if filename not in EXPOSED_ARTIFACTS:
        raise HTTPException(status_code=404, detail="Unknown artifact.")
    artifact_dir = _artifact_dir_for(prediction_id)
    path = artifact_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    return FileResponse(path=path, filename=filename)


__all__ = ["router"]
