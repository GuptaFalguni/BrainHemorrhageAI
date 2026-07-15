# BrainHemorrhageAI — Project Context

## Vision

BrainHemorrhageAI is a research-grade clinical decision support pipeline for multi-class intracranial hemorrhage segmentation on non-contrast brain CT. The long-term goal is a production web application; the current focus is a reproducible supervised baseline suitable for thesis work and publication.

## Research goals

The system should support:

- Master's thesis and research publication
- Portfolio and interview demonstration
- Reproducible, auditable experiments

Every engineering decision prioritizes reproducibility, maintainability, and clinical transparency.

## Dataset

**BHSD (Brain Hemorrhage Segmentation Dataset)** — Wu et al., arXiv:2308.11298

| Split | Count | Purpose |
|-------|-------|---------|
| Labeled | 192 CT + mask pairs | Supervised training and evaluation |
| Unlabeled | 1980 volumes | Deferred until Version 1 baseline is complete |

**Classes:** 0 = background; 1 = EDH; 2 = SDH; 3 = SAH; 4 = IPH; 5 = IVH

**Locked split:** 134 train / 29 val / 29 test (`data/metadata/splits.csv`, seed 42, multi-label stratification)

## Current project status

| Area | Status |
|------|--------|
| Architecture design | Complete — `docs/architecture/` |
| Preprocessing pipeline | Complete — HU clip/window + z-score |
| Dataset loader (2.5D) | Complete — MONAI DataLoader |
| Train/val/test split | Complete — locked and versioned |
| Baseline model | Complete — MONAI 2D U-Net |
| Training pipeline | Complete — Trainer, checkpointing, logging |
| Evaluation pipeline | Complete — test metrics, volume, confidence |
| Experiment framework | Complete — preflight, orchestration, tracking |
| Sanity experiment | Complete — `experiment_sanity` (2 epochs) |
| MONAI V1 baseline | Complete — `experiment_best30h` (locked test Dice 0.257375) |
| nnU-Net comparator | Trained fold 0 (best EMA ~0.479); locked-test eval pending |
| Inference pipeline | Complete for MONAI — `src/inference/` |
| FastAPI backend | Complete — `src/api/` (MONAI checkpoint) |
| Streamlit frontend | Complete — `frontend/app.py` |
| Research utilities | `src/research/` |
| Docs / model cards | `docs/DOCUMENTATION_INDEX.md`, `docs/results/`, `docs/models/` |
| Release candidate | Version 1.0 — see `docs/release/release_candidate_v1.md` |
| `experiment_001` | Deprecated — replaced by `experiment_best30h` |

## AI architecture (implemented)

```
Brain CT (NIfTI)
    ↓
Intensity preprocessing (clip + z-score)
    ↓
2.5D slice extraction (3 axial channels)
    ↓
MONAI 2D U-Net
    ↓
Slice predictions → 3D volume reconstruction
    ↓
Segmentation metrics + native-space volume (mL) + confidence
    ↓
Experiment reports
```

**Checkpoint policy:** Best model selected by validation macro Dice (hemorrhage classes 1–5 only).

**Test policy:** Test split evaluated once, post-training only.

## Technology stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Deep learning | PyTorch, MONAI |
| Medical I/O | nibabel (SimpleITK reserved for resampling) |
| Config | YAML |
| Backend | FastAPI (`src/api/`) |
| Frontend | Streamlit (`frontend/app.py`); Next.js deferred |

## Folder structure

```
BrainHemorrhageAI/
├── config/
├── data/metadata/          # Versioned split manifests
├── data/raw/               # Local BHSD volumes (gitignored)
├── docs/                   # See DOCUMENTATION_INDEX.md
├── src/dataset/
├── src/preprocessing/
├── src/models/
├── src/training/
├── src/evaluation/
├── src/experiments/
├── src/nnunet_prep/
├── src/research/
├── src/visualization/
├── reports/                # Generated (gitignored)
└── checkpoints/            # Generated (gitignored)
```

## Clinical constraints

The dataset supports segmentation, subtype labels, and volume estimation. It does **not** include patient demographics, GCS, outcomes, or established severity scores. Severity rules must be evidence-based and documented separately — not predicted by the segmentation model alone.

## Coding philosophy

- Minimal, necessary code only
- No duplicate logic; remove dead code
- One responsibility per module
- Config-driven behavior
- Explain decisions before implementing new features

## Next roadmap

1. nnU-Net locked-test evaluation (project metrics) — still incremental Phase B prep
2. Dual-model inference + frontend selector (Phase B)
3. Deferred: resampling, class weights, unlabeled SSL, deeper architecture ablations
4. Production hardening of the API and optional Next.js frontend

## Key references

- Documentation map: `docs/DOCUMENTATION_INDEX.md`
- BHSD: Wu et al., arXiv:2308.11298
- Experiment protocol: `docs/research/experiment_protocol.md`
- Windowing review: `docs/research/windowing_literature_review.md`
- Weak vs strong plan: `docs/training/weak_vs_strong_model_plan.md`
