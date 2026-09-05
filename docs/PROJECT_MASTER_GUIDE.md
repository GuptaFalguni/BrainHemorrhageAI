# BrainHemorrhageAI — Project Master Guide

**Status:** Authoritative project entry (F0.5 documentation freeze)  
**Product notice:** Research Software — Not for Clinical Use  
**Package:** `brain-hemorrhage-ai` `1.0.0`

This is the **only document you need to start**. Detailed science and engineering live in linked authoritative docs — do not treat this file as a second source of metrics or API routes.

**Documentation map:** [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) · **Dependency graph:** [`platform/documentation_dependency_graph.md`](platform/documentation_dependency_graph.md)

---

## Project overview

BrainHemorrhageAI is a multi-class intracranial hemorrhage (ICH) **CT segmentation** research system on the BHSD `label_192` dataset. It provides:

- Frozen preprocessing / train / evaluate pipelines  
- MONAI 2.5D U-Net **interactive** baseline  
- nnU-Net 3D **research** comparator  
- Unified deployment inference (both models)  
- Clinical post-processing (severity literature-gated)  
- Production **API v1**  
- Interim Streamlit UI; **Next.js frontend designed, not implemented**

---

## Research problem

Segment five hemorrhage subtypes (EDH, SDH, SAH, IPH, IVH) plus background on non-contrast head CT; report volumes and uncalibrated confidence; prepare research comparison suitable for thesis/publication tables.

Protocol: [`research/experiment_protocol.md`](research/experiment_protocol.md)

---

## Dataset

| Item | Value |
|------|-------|
| Dataset | BHSD `label_192` |
| Locked split | 134 / 29 / 29 — `data/metadata/splits.csv` (seed 42) |
| Labels | 0–5 BG, EDH, SDH, SAH, IPH, IVH |

Authority: [`architecture/data_split_design.md`](architecture/data_split_design.md)

---

## Architecture (science freeze)

Frozen design docs under [`architecture/`](architecture/):

| Topic | Doc |
|-------|-----|
| Split | `data_split_design.md` |
| Windowing | `windowing_normalization_design.md` + `research/windowing_literature_review.md` |
| Resampling | `resampling_design.md` (V1 = native) |
| Model | `model_design.md` |
| Training | `training_pipeline_design.md` |
| Evaluation | `evaluation_pipeline.md` |

---

## Preprocessing

- HU clip `[-40, 120]`, normalize `(HU − 40) / 80`  
- Config: `config/preprocessing.yaml`  
- V1: **no** spatial resampling  

---

## MONAI baseline

| Item | Value |
|------|-------|
| Registry id | `monai_best30h` (category **interactive**, **default**) |
| Experiment | `experiment_best30h` |
| Checkpoint | `checkpoints/experiment_best30h/best_model.pt` |
| Locked-test macro Dice | **0.257375** |

Card: [`models/monai_best30h.md`](models/monai_best30h.md)  
Metrics detail: [`training/experiment_best30h_kaggle_v2_metrics.md`](training/experiment_best30h_kaggle_v2_metrics.md)

---

## nnU-Net benchmark

| Item | Value |
|------|-------|
| Registry id | `nnunet_fold0` (category **research**) |
| Checkpoint | `checkpoints/nnunet_dataset501_fold0/checkpoint_best.pth` |
| Locked-test macro Dice | **0.454723** |
| CPU latency (locked test) | ~328 s/case mean |

Card: [`models/nnunet_fold0.md`](models/nnunet_fold0.md)  
Comparison: [`results/comparison.md`](results/comparison.md)  
Recommendation: [`results/final_model_recommendation.md`](results/final_model_recommendation.md)

**Roles (frozen):** nnU-Net = quality / research; MONAI = interactive default / latency.

SSL 2-fold ensemble (this branch): [`ssl/README.md`](ssl/README.md) · card [`models/nnunet_ssl_2fold.md`](models/nnunet_ssl_2fold.md). Locked-test ensemble Dice **0.488552** (IoU 0.343, volume error 18.5 mL). Best single fold remains fold1 epoch 511 **0.4965**.

---

## Training workflow

- MONAI: `src/training/`, experiment YAMLs under `config/experiments/`  
- nnU-Net: conversion `src/nnunet_prep/`, protocol `docs/training/nnunet_phase0.md`  
- GPU protocol: `docs/training/gpu_training_protocol.md`

---

## Evaluation workflow

- Shared metrics: `src/evaluation/`  
- MONAI eval artifacts: `reports/evaluation/experiment_best30h_eval/`  
- nnU-Net adapter: `src/evaluation/evaluate_nnunet.py` → `reports/evaluation/nnunet_dataset501_fold0_eval/`  
- Do not re-run locked test as a tuning loop

---

## Inference workflow

| Layer | Path |
|-------|------|
| Legacy MONAI CLI pipeline | `src/inference/` (still used internally by MONAI backend) |
| Unified deployment | `src/deployment/inference/` — `UnifiedInferencePipeline` |
| Factory | `InferenceFactory(model_id)` → MONAI or nnU-Net backend |

Authority: platform + API contract below.

---

## Clinical workflow

`src/clinical/` consumes `UnifiedInferenceResult` only.

- Severity: `config/clinical/severity_rules.yaml` → globally **`not_configured`** until literature freeze  
- Outputs: summary + `clinical_report.md`  
- No treatment advice; no invented thresholds  

---

## Deployment workflow

- Registries: `config/experiments/registry.yaml`, `config/models/registry.yaml`  
- Loaders: `src/deployment/`  
- Design history: [`platform/phase_c_inference_platform.md`](platform/phase_c_inference_platform.md)  
- **HTTP truth:** [`platform/api_v1_contract.md`](platform/api_v1_contract.md) · freeze [`platform/backend_freeze_f09.md`](platform/backend_freeze_f09.md)  
- Frontend UI: [`platform/frontend_spa_workspace_design.md`](platform/frontend_spa_workspace_design.md) · app in `web/`

Artifacts for API runs: `reports/api_v1/predictions/{prediction_id}/`

---

## API

Canonical contract: **[`platform/api_v1_contract.md`](platform/api_v1_contract.md)**

```text
GET  /api/v1/health
GET  /api/v1/models
POST /api/v1/predict
GET  /api/v1/report/{prediction_id}
GET  /api/v1/artifacts/{prediction_id}[/{filename}]
```

Run: `uvicorn api.v1.app:app --reload` (from environment with `src` on path / editable install).

**Legacy** `src/api/app.py` (`/predict`, `/files/{job_id}`) — migration only; not for new clients.

---

## Frontend

| Item | Status |
|------|--------|
| Production UI | **Delivered** — single-page workspace in `web/` |
| Canonical design | [`platform/frontend_spa_workspace_design.md`](platform/frontend_spa_workspace_design.md) |
| Interim UI | Streamlit `frontend/app.py` (legacy only) |

---

## Folder structure (high level)

```text
BrainHemorrhageAI/
  config/           # preprocessing, experiments, models, clinical, nnU-Net
  data/             # raw BHSD, splits, nnU-Net raw (local)
  checkpoints/      # frozen weights
  docs/             # this guide + authoritative docs
  reports/          # evaluation, inference, api_v1 artifacts
  src/              # python packages (dataset → api)
  frontend/         # interim Streamlit only
  web/              # (planned) production frontend — not created yet
  notebooks/        # exploratory only
```

Folder audit notes: [`release/release_readiness_f05.md`](release/release_readiness_f05.md)

---

## How to reproduce (summary)

1. `python -m venv .venv` → activate → `pip install -e ".[api]"` (add `nnunet` extra for research model).  
2. Ensure BHSD under `data/raw/label_192/` and checkpoints present.  
3. Smoke: `python src/deployment/validate_registry.py`  
4. Inference engine: `python src/deployment/inference/validate_inference_engine.py` (nnU-Net long on CPU).  
5. Clinical: `python src/clinical/validate_clinical_pipeline.py`  
6. API: `python src/api/v1/validate_api_v1.py`  
7. Serve API: `uvicorn api.v1.app:app --host 127.0.0.1 --port 8000`

Locked-test re-scoring is expensive; prefer existing `reports/evaluation/**` unless invalidating a bugfix.

---

## Important commands

| Command | Purpose |
|---------|---------|
| `pip install -e ".[api,nnunet]"` | Full local stack |
| `python src/deployment/validate_registry.py` | Registry freeze check |
| `python src/api/v1/validate_api_v1.py` | API v1 smoke (MONAI predict) |
| `uvicorn api.v1.app:app` | Production API |
| `streamlit run frontend/app.py` | Interim UI (legacy) |

---

## Current achievements

- Locked split + literature-fixed preprocessing  
- MONAI best30h locked-test metrics published  
- nnU-Net fold0 locked-test metrics + fair comparison  
- Dual-model deployment inference + clinical layer  
- API v1 with prediction_id artifacts  
- Frontend design + architecture freeze (no UI code yet)  
- Documentation F0.5 consistency pass  

---

## Current limitations

- Not a medical device; no clinical validation  
- Severity rules `not_configured`  
- Confidence uncalibrated  
- EDH weak on both models  
- nnU-Net slow on CPU  
- Original CT not downloadable via API whitelist (viewer must cache upload)  
- No auth, no async jobs, no production Next.js UI  
- No prediction list API (browser history only for future UI)  

---

## Future roadmap

1. Accept F0.5 → implement frontend F1+ per blueprint  
2. Optional API expose of input CT for durable Cornerstone  
3. Severity literature freeze → activate YAML thresholds  
4. Auth / cloud / async workers (additive)  
5. PDF reports (future)  
6. Publication polish from `publication_comparison_tables.md`  

---

## Publications / citation anchors

- BHSD: Wu et al., arXiv:2308.11298  
- nnU-Net: Isensee et al., Nature Methods 2021  
- Project tables: [`results/publication_comparison_tables.md`](results/publication_comparison_tables.md)

---

## References to detailed docs

Use [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) for the full table. Prefer the **Authoritative** column in [`platform/documentation_dependency_graph.md`](platform/documentation_dependency_graph.md) when two files overlap.
