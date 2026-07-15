# Backend Freeze F0.9 — Record

**Date:** 2026-07-15  
**Purpose:** Final backend contract freeze before Next.js implementation  
**HTTP SSoT:** [`api_v1_contract.md`](api_v1_contract.md)

---

## What was frozen

| Surface | Status |
|---------|--------|
| Registries YAML + loaders | Frozen (text sync only in F0.5) |
| Unified inference algorithms | Frozen (untouched in F0.9) |
| Evaluation / preprocessing / training | Frozen (untouched) |
| Clinical pipeline logic | Frozen; naming aligned (`total_volume_ml`) |
| API v1 routes + schemas | Frozen |
| Artifact whitelist | Frozen (includes `input_ct.nii.gz`, `meta.json`) |

## Module responsibilities (verified)

| Package | Responsibility |
|---------|----------------|
| `deployment` | Registries only at package root; inference via `deployment.inference` |
| `deployment.inference` | Framework-agnostic predict → `UnifiedInferenceResult` |
| `clinical` | Post-seg severity/summary/report; no model calls |
| `api.v1` | HTTP orchestration only |
| `evaluation` | Metrics (research); reused by backends, not called from API routers |
| `inference` | MONAI production pipeline wrapped by `MonaiBackend` |

## Naming (canonical)

`prediction_id` · `model_id` · `experiment_id` · `scan_name` · `total_volume_ml` · `per_class_volume_ml` · `volume_ml` (per-subtype)

## Changes in F0.9 (contract only)

1. Publish `input_ct.nii.gz` for durable viewer loads  
2. Expose `meta.json`  
3. Align clinical summary field → `total_volume_ml`  
4. Add `metadata.experiment_id` on predict  
5. Strengthen API validation against volume/confidence/artifact contract  
6. Integration + implementation checklists  

## Explicitly deferred

- Auth, async jobs, prediction list API  
- Severity numeric thresholds  
- Deleting legacy `src/api/app.py` (Streamlit interim)

## Gate

Frontend may use this freeze + [`api_v1_contract.md`](api_v1_contract.md). Product UI: [`frontend_spa_workspace_design.md`](frontend_spa_workspace_design.md).
