# Weak vs Strong Model Plan — MONAI Baseline + nnU-Net Comparator

**Status:** Historical plan — **comparison & dual serving complete** (Phases B–C). Keep for thesis timeline context.  
**Do not use for current serving status.** Use instead:

- [`docs/results/comparison.md`](../results/comparison.md)  
- [`docs/results/final_model_recommendation.md`](../results/final_model_recommendation.md)  
- [`docs/platform/api_v1_contract.md`](../platform/api_v1_contract.md)  
- [`docs/PROJECT_MASTER_GUIDE.md`](../PROJECT_MASTER_GUIDE.md)

**Created:** 2026-07-12  
**Goal (original):** Ship a reproducible side-by-side comparison — one weak baseline, one stronger supervised model — then expose both.

**Related documents (updated):**

- `docs/training/experiment_best30h_kaggle_v2_metrics.md` — locked MONAI V1 results
- `docs/training/gpu_training_protocol.md` — GPU operator rules
- `docs/research/experiment_protocol.md` — research sequencing
- `docs/release/release_candidate_v1.md` — current RC (F0.5)

---

## 1. End goal (product + research)

| Deliverable | Definition of done |
|-------------|--------------------|
| **Weak model (Model A)** | MONAI 2.5D U-Net from `experiment_best30h` — already trained |
| **Strong model (Model B)** | nnU-Net (1-fold, locked split), time-capped on Kaggle — intended to beat Model A overall |
| **Fair comparison** | Same `data/metadata/splits.csv`, same test metrics, per-class Dice + volume error |
| **Honest limitations** | Document EDH rarity, 192 labels, 12h runtime, no unlabeled data in this phase |
| **Frontend** | One upload → predictions from **both** models (side-by-side overlays / volumes) |
| **Future production** | Keep FastAPI + Streamlit path; swap/add checkpoints; harden later (auth, monitoring, model card) |

**Success bar:** Model B is **better at prediction overall** than MONAI (macro Dice / visual quality on typical cases). It is OK if EDH or rare subtypes remain weak.

---

## 2. Constraints (non-negotiable)

| Constraint | Implication |
|------------|-------------|
| Kaggle **12h max** per Save Version | No full default nnU-Net marathon in one commit |
| ~**30h** weekly GPU quota | Typically **two** ≤12h runs available (train + optional resume) — treat as scarce |
| Local machine **no NVIDIA GPU** | All heavy training on Kaggle; local = prep, eval orchestration, frontend |
| Labeled set only **192** volumes | Rare classes stay hard; EDH ≈ 16/134 train scans |
| One wasted run is expensive | Phase 0 local prep **before** any long GPU train |

---

## 3. What we already have (locked)

### Model A — Weak / baseline (COMPLETE)

| Item | Value |
|------|-------|
| Experiment | `experiment_best30h` |
| Architecture | MONAI 2D U-Net, 2.5D input (3 channels), 6 classes |
| Checkpoint | `checkpoints/experiment_best30h/best_model.pt` |
| Best val macro Dice | **0.351169** @ epoch 24 |
| Test macro Dice | **0.257375** |
| Notable failure | EDH ≈ **0.01** Dice; false multi-subtype predictions on thin bleeds |
| Demo path | FastAPI + Streamlit with `BRAIN_HEMORRHAGE_CHECKPOINT` |

**Do not retrain Model A.** It is the fixed weak reference for all comparisons.

### Known failure modes (why Model B exists)

1. Extreme class imbalance (EDH tiny voxel fraction + few scans)
2. Shallow 2.5D U-Net capacity / limited 3D context
3. No class-weighted loss in V1 (deferred — see §5)
4. Thick-slice anisotropy (~5 mm) hurts small lesions
5. Val→test gap on small test set (n=29)

---

## 4. Decisions (approved)

| Decision | Choice | Why |
|----------|--------|-----|
| Strong method | **nnU-Net 3D fullres** | Strongest standard supervised CT comparator; matches research protocol V3 |
| Folds | **1-fold only** | 5-fold does not fit quota |
| Split | **Locked BHSD split** (`splits.csv`) | Fair vs MONAI; no random reshuffle |
| Weighted MONAI ablation | **Skip for now** | Would burn ~8–12h; lower ROI than one strong comparator under quota |
| Unlabeled `unlabel_1980` | **Deferred** | After weak-vs-strong story ships |
| Attention U-Net / Swin | **Deferred** | Extra GPU experiments not needed for dual-demo goal |
| Interactive Draft for long train | **Forbidden** | Use Save Version; continuous checkpoint copy to Output |

---

## 5. Explicitly deferred (and how we justify it)

| Deferred item | Justification in writeup |
|---------------|--------------------------|
| MONAI + class weights / hemorrhage sampling | Compute spent on nnU-Net instead; V1 remains unweighted baseline |
| Full nnU-Net 1000-epoch / 5-fold | Hardware + 12h wall clock |
| Semi-supervised learning | Needs stable supervised comparator first |
| “Fix EDH completely” | Data-limited; report as limitation, not failure of engineering |

---

## 6. Phased plan

### Phase 0 — Local prep (0 GPU) ← **START HERE**

**Purpose:** Eliminate setup failure before spending the 12h run.

| Step | Task | Exit criteria |
|------|------|-----------------|
| 0.1 | Install / pin nnU-Net tooling in a documented env recipe for Kaggle | Reproducible install commands |
| 0.2 | Convert BHSD `label_192` → nnU-Net raw dataset layout | All 192 image/label pairs validated |
| 0.3 | Wire **custom 1-fold split** = train/val from `splits.csv`; hold out test | No test leakage into training |
| 0.4 | Label sanity: classes 0–5, shapes aligned, naming convention correct | Checklist signed off |
| 0.5 | Author Kaggle notebook cells: copy data, install, preprocess, train, **copy checkpoints to `/kaggle/working` every N epochs** | Notebook reviewed |
| 0.6 | Time-cap policy: target train window inside 12h; keep `checkpoint_best` + `checkpoint_latest` | Kill-safe design |
| 0.7 | Optional short Draft smoke: plan + tiny train steps only | Confirms imports/paths; **not** a long run |

**Phase 0 is complete when:** we would bet the 12h Save Version produces downloadable weights even if Kaggle stops the job at 11:50.

**Do not start Phase 1 until Phase 0 exit criteria pass.**

---

### Phase 1 — Kaggle Run #1 (≤12h) — strong model train

| Item | Plan |
|------|------|
| Job type | **Save Version** (GPU T4×2 if available) |
| Model | nnU-Net 3D fullres, 1-fold, locked split |
| Time use | ~0.5–2h preprocess + ~10h train (approx.) |
| Artifacts to download | best/latest checkpoints, plans, logs, validation summaries |
| Success | Usable inference weights on disk locally |
| Stretch | Already trends better than MONAI on val |

---

### Phase 2 — Kaggle Run #2 (≤12h) — only if needed

| If Run #1… | Then Run #2… |
|------------|--------------|
| Usable weights but under-trained | **Resume** nnU-Net only |
| Already clearly better than MONAI | **Skip** — save remaining quota |
| Failed setup / no weights | Fix from Phase 0 lessons and rerun (worst case) |

**Do not** invent a third method in Run #2.

---

### Phase 3 — Local evaluate + dual frontend (0 GPU)

| Step | Task |
|------|------|
| 3.1 | Evaluate Model B on **locked test** with same metric suite as Model A |
| 3.2 | Write comparison table: macro + per-class Dice, volume MAE, runtime, device |
| 3.3 | Document limitations (EDH, n=192, 12h cap, skipped weighted MONAI) |
| 3.4 | API: load Model A + Model B (or selectable backend) |
| 3.5 | Streamlit: side-by-side overlays, volumes, confidence, model name labels |
| 3.6 | Demo script: one “SDH-friendly” case + one known EDH failure case (honest demo) |

---

### Phase 4 — Future productionisation (later)

Not blocking the dual-model demo:

1. Model card (intended use, failure modes, metrics)
2. API auth, upload cleanup, rate limits
3. Calibration / threshold policy for clinical display
4. Optional cloud GPU for longer nnU-Net or semi-supervised
5. Optional Next.js UI (Streamlit remains research demo)

---

## 7. Suggested calendar (1 week)

| Day | Focus |
|-----|--------|
| 1–2 | **Phase 0** — conversion, split, notebook, smoke |
| 3 | **Phase 1** — Kaggle Run #1 (≤12h); download artifacts |
| 4 | Quick local check vs MONAI; decide resume or skip |
| 5 | **Phase 2** only if required |
| 6–7 | **Phase 3** — test eval writeup + dual-model frontend |

---

## 8. Risk register

| Risk | Mitigation |
|------|------------|
| 12h ends mid-epoch with empty Output | Continuous copy of checkpoints to `/kaggle/working`; prefer Save Version |
| nnU-Net preprocess burns too much of 12h | Pre-validate conversion locally; streamline notebook; consider prebuilt dataset dataset on Kaggle if needed |
| Model B does not beat MONAI | Use Run #2 resume; if still worse, report honestly and keep MONAI as default demo model |
| EDH still broken | Expected; show as limitation, not a surprise |
| Dual frontend delayed by nnU-Net inference glue | Budget Phase 3 for nnU-Net predict wrapper; keep MONAI path unchanged |
| Starting long train before Phase 0 | **Hard stop** — violates this plan |

---

## 9. Comparison protocol (when both models exist)

Report at minimum:

1. Test macro Dice (classes 1–5)
2. Per-class Dice / IoU / precision / recall
3. Total and per-subtype volume absolute error (mL)
4. Qualitative overlays on fixed case IDs (best SDH case + hard EDH case)
5. Train compute: GPU type, wall clock, epochs completed
6. Statement of limitations and deferred work (§5)

Primary claim language:

> Under identical BHSD splits and metrics, time-capped nnU-Net (1-fold) outperforms the Version 1 MONAI 2.5D U-Net baseline on overall test macro Dice, while rare-class (EDH) performance remains limited by label prevalence and compute.

---

## 10. Immediate next action

**Start Phase 0.1–0.3:** environment recipe + BHSD→nnU-Net conversion + locked-split wiring.

No Kaggle long train until Phase 0 exit criteria in §6 are met.

---

## 11. Checkpoint status board

| Item | Status |
|------|--------|
| Model A trained + local checkpoint | **Done** |
| Model A frontend smoke | **Done** |
| Phase 0 nnU-Net prep | **Done** |
| Phase 1 Kaggle strong train | **Done (time-capped)** — ~epoch 196; artifacts in `checkpoints/nnunet_dataset501_fold0/` |
| Phase 2 resume (optional) | Deferred — best checkpoint already usable |
| Phase 3 dual frontend | **Next** — evaluate on locked test, then side-by-side UI |
| Phase 4 production hardening | Later |

---

*This plan supersedes the idea of a separate long weighted-MONAI Kaggle run under the current quota. Revisit weighted MONAI only if Model B cannot be delivered or a free short GPU window appears after the dual-demo ships.*
