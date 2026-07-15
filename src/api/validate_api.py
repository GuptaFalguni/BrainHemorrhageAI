"""Automated validation for the production inference API."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from api.schemas import PredictionResponse
from api.service import REQUIRED_API_FILES, get_pipeline, reset_pipeline_cache
from preprocessing.preprocess_volume import IMAGES_DIR


def _record(checks: list[tuple[str, bool, str]], name: str, passed: bool, detail: str = "") -> None:
    checks.append((name, passed, detail))
    status = "OK" if passed else "FAIL"
    line = f"  {name}: {status}"
    if detail:
        line += f" ({detail})"
    print(line)


def validate_api() -> None:
    """Run API validation checks."""
    checks: list[tuple[str, bool, str]] = []

    app = create_app()
    client = TestClient(app)

    health = client.get("/health")
    _record(checks, "API starts", health.status_code == 200, f"status={health.status_code}")

    reset_pipeline_cache()
    pipeline = get_pipeline()
    _record(checks, "model loads", pipeline is not None, str(pipeline.device))

    image_files = sorted(IMAGES_DIR.glob("*.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No test scans in {IMAGES_DIR}")
    sample_path = image_files[0]

    with sample_path.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"file": (sample_path.name, handle.read(), "application/gzip")},
        )
    _record(checks, "upload works", response.status_code == 200, f"status={response.status_code}")
    _record(checks, "inference runs", response.status_code == 200)

    payload = response.json()
    try:
        PredictionResponse.model_validate(payload)
        schema_valid = True
    except Exception as error:
        schema_valid = False
        schema_detail = str(error)
    else:
        schema_detail = f"job_id={payload.get('job_id')}"
    _record(checks, "JSON schema valid", schema_valid, schema_detail)

    output_dir = Path(payload["output_dir"])
    for artifact in REQUIRED_API_FILES:
        exists = (output_dir / artifact).exists()
        _record(checks, f"artifact {artifact}", exists, str(output_dir / artifact))

    overlay_exists = (output_dir / "overlay.png").exists()
    _record(checks, "frontend assets generated", overlay_exists)

    service_module = importlib.import_module("service")
    module_source = inspect.getsource(service_module)
    run_source = inspect.getsource(service_module.run_prediction)
    uses_inference = "InferencePipeline" in module_source and "pipeline.predict" in run_source
    preprocess_duplicated = any(
        token in module_source
        for token in ("apply_intensity_clip", "apply_normalization", "preprocess_nifti_scan")
    )
    _record(
        checks,
        "no duplicated preprocessing",
        uses_inference and not preprocess_duplicated,
        "service delegates to inference.InferencePipeline",
    )

    summary_path = output_dir / "prediction_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _record(
        checks,
        "prediction_summary fields",
        {"shape", "spacing", "total_volume_ml", "classes_present"}.issubset(summary.keys()),
    )

    pipeline_second = get_pipeline()
    _record(checks, "model cached", pipeline_second is pipeline, "same instance reused")

    failed = [name for name, passed, _ in checks if not passed]
    if failed:
        raise RuntimeError(f"API validation failed: {', '.join(failed)}")

    print(f"\nAPI validation passed — sample output: {output_dir}")


def main() -> None:
    validate_api()


if __name__ == "__main__":
    main()
