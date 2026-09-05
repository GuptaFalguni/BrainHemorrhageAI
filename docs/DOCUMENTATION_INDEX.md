# Documentation Index — BrainHemorrhageAI

Single map of project documentation. **Start:** [`PROJECT_MASTER_GUIDE.md`](PROJECT_MASTER_GUIDE.md) · **Graph:** [`platform/documentation_dependency_graph.md`](platform/documentation_dependency_graph.md)

**Convention**

| Status | Meaning |
|--------|---------|
| Approved and Frozen | Science design locked for V1 |
| Active | Current authority |
| Historical | Context only; do not use for serving/status |
| Reference | Supporting literature / protocol |

---

## Recommended reading order

1. `PROJECT_MASTER_GUIDE.md`  
2. `platform/documentation_dependency_graph.md`  
3. `platform/api_v1_contract.md`  
4. `results/comparison.md` + `results/final_model_recommendation.md`  
5. `platform/frontend_spa_workspace_design.md` (**canonical product UI**)  
6. `architecture/` as needed for thesis depth  

---

## Root

| Document | Purpose | Status |
|----------|---------|--------|
| `PROJECT_MASTER_GUIDE.md` | Only entry overview | Active |

## Architecture (`docs/architecture/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `data_split_design.md` | Locked 134/29/29 | Approved and Frozen |
| `windowing_normalization_design.md` | HU policy | Approved and Frozen |
| `resampling_design.md` | V1 = native | Approved and Frozen |
| `training_pipeline_design.md` | 2.5D + MONAI recipe | Approved and Frozen |
| `model_design.md` | V1 U-Net | Approved and Frozen |
| `evaluation_pipeline.md` | Metrics / volumes / confidence | Approved and Frozen |

## Research (`docs/research/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `experiment_protocol.md` | Sequencing | Active / Reference |
| `windowing_literature_review.md` | Citations | Reference |

## Training (`docs/training/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `experiment_best30h_kaggle_v2_metrics.md` | MONAI metrics dump | Active |
| `nnunet_phase0.md` | nnU-Net prep/train notes | Active |
| `gpu_training_protocol.md` | GPU operators | Active |
| `weak_vs_strong_model_plan.md` | Historical plan | Historical |

## Results (`docs/results/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `comparison.md` | Locked-test MONAI vs nnU-Net | **Authoritative comparison** |
| `publication_comparison_tables.md` | Export tables | Active |
| `final_model_recommendation.md` | Roles (interactive vs research) | Active |
| `monai_results.md` / `nnunet_results.md` | Per-model narrative | Active |

## Models (`docs/models/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `monai_best30h.md` | Model A card | Active |
| `nnunet_fold0.md` | Model B card | Active |
| `nnunet_ssl_2fold.md` | SSL 2-fold ensemble card | Active |

## SSL (`docs/ssl/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `README.md` | Kaggle 2-fold outcomes, compute limits | Active |
| `weights_local_status.md` | Whether epoch-511 weights are on disk | Active |

## Platform (`docs/platform/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `api_v1_contract.md` | **Authoritative HTTP API (F0.9 frozen)** | Active |
| `backend_freeze_f09.md` | Backend freeze record | Active |
| `phase_c_inference_platform.md` | Deployment design history | Active |
| `frontend_spa_workspace_design.md` | **Canonical product UI** (single-page) | Active |
| `documentation_dependency_graph.md` | Doc authority graph | Active |

## Release (`docs/release/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `release_candidate_v1.md` | RC summary | Active (F0.5) |
| `release_readiness_f05.md` | F0.5 scores / blockers | Active |

## Archive (`docs/archive/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `README.md` | Archive policy | Active policy |
