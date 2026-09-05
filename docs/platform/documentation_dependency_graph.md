# Documentation dependency graph

**Authority rule:** Lower layers do not redefine upper-layer science. Horizontal siblings must cross-link, not copy.

```text
Research problem & protocol
        │
        ▼
Architecture (frozen V1 science decisions)
        │
        ▼
Training (MONAI / nnU-Net protocols & metrics dumps)
        │
        ▼
Evaluation (locked-test methods & raw reports under reports/)
        │
        ▼
Results & model cards (interpreted evidence)
        │
        ▼
Deployment platform (registries, inference, clinical)
        │
        ▼
API v1 contract (HTTP as implemented)
        │
        ▼
Frontend SPA design (`frontend_spa_workspace_design.md`)
        │
        ▼
In-app / thesis docs (curated subset)
```

---

## Authoritative documents by concern

| Concern | Authoritative document | Role of others |
|---------|------------------------|----------------|
| Project entry | [`docs/PROJECT_MASTER_GUIDE.md`](../PROJECT_MASTER_GUIDE.md) | Links only |
| Doc map | [`docs/DOCUMENTATION_INDEX.md`](../DOCUMENTATION_INDEX.md) | Index |
| Experiment sequencing | `docs/research/experiment_protocol.md` | Reference |
| Windowing citations | `docs/research/windowing_literature_review.md` | Cited by preprocessing |
| Split / preprocess / model / train / eval design | `docs/architecture/*.md` | Frozen design |
| MONAI locked metrics | `docs/training/experiment_best30h_kaggle_v2_metrics.md` + `reports/evaluation/experiment_best30h_eval/` | Cards summarize |
| nnU-Net training notes | `docs/training/nnunet_phase0.md` + nnU-Net reports | Cards summarize |
| Locked-test comparison | `docs/results/comparison.md` | Tables / recommendation cite it |
| Publication tables | `docs/results/publication_comparison_tables.md` | Export view |
| Model recommendation | `docs/results/final_model_recommendation.md` | Roles only |
| Model A card | `docs/models/monai_best30h.md` | Serving notes point to API contract |
| Model B card | `docs/models/nnunet_fold0.md` | Serving notes point to API contract |
| SSL 2-fold card | `docs/models/nnunet_ssl_2fold.md` | Optional research ensemble |
| Deployment architecture | `docs/platform/phase_c_inference_platform.md` | History + platform intent |
| **HTTP API** | **`docs/platform/api_v1_contract.md`** | Only route table that clients follow |
| **Frontend product UI** | **`docs/platform/frontend_spa_workspace_design.md`** | Canonical single-page workspace |
| Clinical severity YAML | `config/clinical/severity_rules.yaml` | Engine reads this |
| Registries | `config/models/registry.yaml`, `config/experiments/registry.yaml` | Code loaders |
| Release posture | `docs/release/release_candidate_v1.md` | Current RC |
| Readiness scores | `docs/release/release_readiness_f05.md` | F0.5 audit |

---

## Viewer curation (future Next.js `/docs`)

Ship only: master guide excerpt pointers, model cards, `comparison.md`, `final_model_recommendation.md`, `api_v1_contract.md`, research disclaimer text.  
Defer full architecture set unless “Advanced / thesis” mode.
