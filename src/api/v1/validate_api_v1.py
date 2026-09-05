"""Validate API v1 — orchestration over frozen pipelines only."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from fastapi.testclient import TestClient

from api.v1.app import create_app
from api.v1.dependencies import CONFIDENCE_KEYS, PROJECT_ROOT, VOLUME_KEYS, get_model_registry

V1_ROOT = Path(__file__).resolve().parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _pick_scan() -> Path:
    images = PROJECT_ROOT / "data" / "raw" / "label_192" / "images"
    preferred = images / "ID_0237f3c9_ID_40015688b9.nii.gz"
    if preferred.is_file():
        return preferred
    candidates = sorted(images.glob("*.nii*"))
    if not candidates:
        raise FileNotFoundError(f"No BHSD images under {images}")
    return candidates[0]


def _no_ai_logic_in_api() -> None:
    forbidden_imports = {
        "torch.nn",
        "monai.networks",
        "nnunetv2.inference",
    }
    forbidden_calls = {
        "InferencePipeline",
        "run_nnunet_prediction",
        "prepare_nnunet_model_folder",
        "stack_scan_predictions",
    }
    required_reuse = {
        "UnifiedInferencePipeline": False,
        "ClinicalPipeline": False,
        "load_deployment_registries": False,
        "ModelRegistry": False,
    }

    for path in V1_ROOT.rglob("*.py"):
        if path.name == "validate_api_v1.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None) or ""
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [f"{module}.{alias.name}" if module else alias.name for alias in node.names]
                    names.append(module)
                for name in names:
                    for forbidden in forbidden_imports:
                        _assert(
                            forbidden not in name,
                            f"API must not import '{forbidden}' ({path})",
                        )
            if isinstance(node, ast.Name) and node.id in forbidden_calls:
                raise AssertionError(f"API must not reference '{node.id}' ({path})")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_calls:
                raise AssertionError(f"API must not reference '{node.attr}' ({path})")

        for key in list(required_reuse):
            if key in source:
                required_reuse[key] = True

    for key, seen in required_reuse.items():
        _assert(seen, f"API v1 must reuse '{key}'")


def run_validation() -> int:
    print("API v1 validation")
    _no_ai_logic_in_api()
    print("  no duplicated AI logic: OK")

    registry = get_model_registry()
    default_id = registry.get_default().model_id
    _assert(default_id == "monai_best30h", f"Unexpected default model: {default_id}")
    _assert("nnunet_fold0" in registry.records, "nnunet_fold0 missing from registry")
    _assert("nnunet_ssl_2fold" in registry.records, "nnunet_ssl_2fold missing from registry")
    print(f"  registry loading / default model ({default_id}): OK")

    app = create_app()
    client = TestClient(app)

    health = client.get("/api/v1/health")
    _assert(health.status_code == 200, f"/health failed: {health.status_code} {health.text}")
    health_body = health.json()
    _assert(health_body["status"] == "ok", "health status not ok")
    _assert("available_models" in health_body, "available_models missing")
    _assert(
        "python_version" in health_body and "pytorch_version" in health_body,
        "python/pytorch versions missing from health",
    )
    _assert(default_id in health_body["available_models"], "default model not listed")
    print("  health endpoint: OK")

    models = client.get("/api/v1/models")
    _assert(models.status_code == 200, f"/models failed: {models.status_code}")
    models_body = models.json()
    _assert(models_body["default_model_id"] == default_id, "models default mismatch")
    model_ids = {item["model_id"] for item in models_body["models"]}
    _assert({"monai_best30h", "nnunet_fold0"} <= model_ids, "Expected models missing")
    _assert("nnunet_ssl_2fold" in model_ids, "nnunet_ssl_2fold missing from registry")
    sample = next(item for item in models_body["models"] if item["model_id"] == default_id)
    _assert(
        "capabilities" in sample and "recommended_use" in sample,
        "model metadata incomplete",
    )
    _assert(sample["category"] == "interactive", "default category mismatch")
    print("  models endpoint / model selection metadata: OK")

    scan = _pick_scan()
    with scan.open("rb") as handle:
        predict = client.post(
            "/api/v1/predict",
            files={"file": (scan.name, handle, "application/octet-stream")},
            data={},  # default model
        )
    _assert(predict.status_code == 200, f"/predict failed: {predict.status_code} {predict.text[:500]}")
    body = predict.json()
    prediction_id = body["prediction_id"]
    _assert(body["model_id"] == default_id, "predict did not use default model")
    _assert(
        "volumes" in body and "confidence" in body and "clinical" in body,
        "predict response missing volumes/confidence/clinical",
    )
    _assert(VOLUME_KEYS <= set(body["volumes"].keys()), f"volumes missing keys: {VOLUME_KEYS - set(body['volumes'])}")
    _assert(
        CONFIDENCE_KEYS <= set(body["confidence"].keys()),
        f"confidence missing keys: {CONFIDENCE_KEYS - set(body['confidence'])}",
    )
    _assert("total_volume_ml" in body["clinical"]["summary"], "clinical.summary must use total_volume_ml")
    _assert(
        body["clinical"]["severity_status"] in {"not_configured", "no_hemorrhage", "configured"},
        "unexpected severity status",
    )
    _assert(body["metadata"].get("experiment_id"), "metadata.experiment_id required")
    print(f"  prediction endpoint (default={default_id}): OK  id={prediction_id}")

    # Explicit model selection
    with scan.open("rb") as handle:
        predict_named = client.post(
            "/api/v1/predict",
            files={"file": (scan.name, handle, "application/octet-stream")},
            data={"model_id": "monai_best30h"},
        )
    _assert(predict_named.status_code == 200, f"named predict failed: {predict_named.status_code}")
    _assert(predict_named.json()["model_id"] == "monai_best30h", "model selection failed")
    print("  model selection: OK")

    report = client.get(f"/api/v1/report/{prediction_id}")
    _assert(report.status_code == 200, f"/report failed: {report.status_code} {report.text[:300]}")
    _assert("markdown" in report.json() and report.json()["markdown"], "empty report")
    print("  report endpoint: OK")

    missing_report = client.get("/api/v1/report/does_not_exist_123")
    _assert(missing_report.status_code == 404, "missing report must be 404")

    artifacts = client.get(f"/api/v1/artifacts/{prediction_id}")
    _assert(artifacts.status_code == 200, f"/artifacts failed: {artifacts.status_code}")
    files = artifacts.json()["files"]
    existing = {item["name"]: item for item in files if item["exists"]}
    _assert("input_ct.nii.gz" in existing, "input_ct.nii.gz must be exposed")
    _assert("overlay.png" in existing or "prediction_overlay.png" in existing, "overlay missing")
    _assert("prediction_mask.nii.gz" in existing, "prediction mask missing")
    _assert("volume_report.csv" in existing, "volume CSV missing")
    _assert("clinical_report.md" in existing, "clinical_report.md missing")
    _assert("meta.json" in existing, "meta.json missing")
    _assert(
        "confidence.json" in existing or "prediction_summary.json" in existing,
        "JSON missing",
    )
    print("  artifacts endpoint: OK")

    # Download one existing file
    sample_name = "input_ct.nii.gz"
    download = client.get(f"/api/v1/artifacts/{prediction_id}/{sample_name}")
    _assert(download.status_code == 200, f"artifact download failed: {sample_name}")
    _assert(len(download.content) > 0, "empty artifact download")
    print(f"  artifact download ({sample_name}): OK")

    print("VALIDATION PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate API v1.")
    parser.parse_args()
    try:
        raise SystemExit(run_validation())
    except Exception as exc:  # noqa: BLE001
        print("VALIDATION FAILED")
        print(f"  - {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
