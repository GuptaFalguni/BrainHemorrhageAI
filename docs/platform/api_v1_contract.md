# API v1 Contract (Authoritative) — Backend Freeze F0.9

**Status:** **FROZEN** for frontend implementation (F0.9)  
**Code:** `src/api/v1/`  
**Validation:** `src/api/v1/validate_api_v1.py`  
**Product notice:** Research Software — Not for Clinical Use

This is the **only** authoritative HTTP contract for new clients.  
Do not use legacy `src/api/app.py` routes (`/health`, `/predict`, `/files/{job_id}/...`) for new work.

**Stable Python entrypoints (do not break):**

| Module | Public API |
|--------|------------|
| `deployment.registry` | `load_deployment_registries` |
| `deployment.model_registry` | `ModelRegistry`, `ModelRecord` |
| `deployment.experiment_registry` | `ExperimentRegistry`, `ExperimentRecord` |
| `deployment.inference` | `UnifiedInferencePipeline`, `InferenceFactory`, `UnifiedInferenceResult` |
| `clinical` | `ClinicalPipeline`, `ClinicalPipelineResult`, `SeverityEngine` |
| `api.v1.app` | `app`, `create_app` |

---

## Base

| Item | Value |
|------|-------|
| Prefix | `/api/v1` |
| App entry | `uvicorn api.v1.app:app` |
| Auth | None (local demo) |
| Jobs / websockets | None (sync predict only) |
| Artifact root | `reports/api_v1/predictions/{prediction_id}/` |
| Index | `reports/api_v1/index.json` |

---

## Canonical names (must match everywhere)

| Name | Meaning |
|------|---------|
| `prediction_id` | Hex UUID for one API run |
| `model_id` | Deployable registry id (`monai_best30h`, `nnunet_fold0`, `nnunet_ssl_2fold`) |
| `experiment_id` | Experiment registry id |
| `scan_name` | Directory-safe scan stem |
| `total_volume_ml` | Total hemorrhage volume (mL) |
| `per_class_volume_ml` | Map label → mL |
| `volume_ml` | Per-subtype field inside severity subtype rows / YAML rules |

---

## Routes

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Liveness, versions, device, available models |
| `GET` | `/api/v1/models` | Model registry metadata + capabilities |
| `POST` | `/api/v1/predict` | Sync inference + clinical pipeline |
| `GET` | `/api/v1/report/{prediction_id}` | Markdown report JSON wrapper |
| `GET` | `/api/v1/report/{prediction_id}/raw` | Raw markdown body |
| `GET` | `/api/v1/artifacts/{prediction_id}` | List whitelisted artifacts |
| `GET` | `/api/v1/artifacts/{prediction_id}/{filename}` | Download one artifact |

### HTTP status conventions

| Code | When |
|------|------|
| 200 | Success |
| 400 | Bad upload (empty file, wrong extension) |
| 404 | Unknown `model_id` or unknown `prediction_id` / missing artifact |
| 500 | Inference/clinical failure or missing checkpoint |

---

## Predict

**Request:** `multipart/form-data`

| Field | Required | Notes |
|-------|----------|-------|
| `file` | Yes | `.nii` / `.nii.gz` CT |
| `model_id` | No | Registry id; omit → default (`monai_best30h`) |
| `mask` | No | Optional GT NIfTI |

**Pipeline:** `UnifiedInferencePipeline` → `ClinicalPipeline`

### PredictResponse (required fields)

| Field | Type | Notes |
|-------|------|-------|
| `prediction_id` | string | |
| `model_id` | string | |
| `framework` | string | `monai` \| `nnunet` |
| `category` | string \| null | `interactive` \| `research` |
| `scan_name` | string | |
| `artifact_dir` | string | Server path (debug) |
| `volumes` | object | see below |
| `confidence` | object | see below |
| `clinical` | object | see below |
| `artifacts` | object | name → filesystem path (existing only) |
| `download_urls` | object | name → `/api/v1/artifacts/{id}/{name}` |
| `report_url` | string | `/api/v1/report/{prediction_id}` |
| `artifacts_url` | string | `/api/v1/artifacts/{prediction_id}` |
| `metadata` | object | includes `experiment_id`, `device`, `processing_time_sec`, `capabilities`, `cache_hit` |

### volumes

```text
total_volume_ml: float
per_class_volume_ml: { "<label>": float, ... }   # JSON keys are strings
classes_present: int[]
spacing: [x, y, z]
```

### confidence

```text
study_confidence: float
per_class_confidence: { "<label>": float, ... }
mean_softmax_confidence: float
confidence_histogram: { counts: int[], bin_edges: float[] }
```

### clinical

```text
severity_status: not_configured | no_hemorrhage | configured
severity_tier: string | null
hemorrhage_detected: bool
recommendations: string[]
limitations: string[]
disclaimer: string
severity: object          # SeverityResult.to_dict()
summary: object           # includes total_volume_ml (same naming as volumes)
clinical_report_path: string | null
```

Default model is **interactive** MONAI. Research models `nnunet_fold0` and `nnunet_ssl_2fold` may take **minutes on CPU**.

---

## Models

| model_id | category | framework | default |
|----------|----------|-----------|---------|
| `monai_best30h` | interactive | monai | yes |
| `nnunet_fold0` | research | nnunet | no |
| `nnunet_ssl_2fold` | research | nnunet | no |

Capabilities (both): overlay, volume, confidence, multiclass, severity = true; uncertainty = false.

---

## Artifacts (download whitelist)

| Filename | Source |
|----------|--------|
| `input_ct.nii.gz` | Published copy of uploaded CT (API layer) |
| `overlay.png` | Alias of prediction overlay |
| `prediction_overlay.png` | Axial overlay |
| `prediction_overlay_3d.png` | Ortho/MIP figure |
| `prediction_mask.nii.gz` | Label map alias |
| `prediction.nii.gz` | Label map |
| `volume_report.csv` | Volume table |
| `confidence.json` | Confidence dump |
| `prediction_summary.json` | Summary JSON |
| `prediction_report.md` | Inference markdown |
| `clinical_report.md` | Clinical markdown |
| `probabilities.npz` | Softmax / probs |
| `meta.json` | prediction_id index record |

Raw upload originals may also exist under `uploads/` but are **not** the stable download name — use `input_ct.nii.gz`.

---

## Explicitly out of contract

- Dated paths `.../artifacts/{date}/{scan}/{model}/...`
- Query-style report URLs
- Async jobs / websockets / auth
- Prediction list/delete APIs

---

## Related

- Freeze record: [`backend_freeze_f09.md`](backend_freeze_f09.md)  
- Frontend UI: [`frontend_spa_workspace_design.md`](frontend_spa_workspace_design.md)
