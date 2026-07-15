# Data Split Design — BHSD (Phase 2)

> **Status: Approved and Frozen**  
> Locked split lives in `data/metadata/splits.csv` (seed 42). Kept for thesis reviewers.

**Project:** BrainHemorrhageAI  
**Dataset:** BHSD labeled set (`label_192`) — 192 CT volumes with multi-class segmentation masks  
**Status:** Approved and Frozen (implementation complete; split must not be regenerated without `--force` review)  
**Last updated:** Phase 2 milestone (freeze banner added during V2 Phase A cleanup)

---

## Context

Phase 1 established that BHSD is a **multi-class, multi-label** segmentation dataset:

| Label | Subtype | Scans containing label |
|-------|---------|------------------------|
| 0 | Background | 192 |
| 1 | EDH | 23 (~12%) |
| 2 | SDH | 127 (~66%) |
| 3 | SAH | 104 (~54%) |
| 4 | IPH | 109 (~57%) |
| 5 | IVH | 70 (~36%) |

Individual scans often contain **multiple hemorrhage subtypes simultaneously**. Geometry varies (19 unique volume shapes; non-uniform voxel spacing). Any split strategy must account for **small sample size**, **class imbalance**, and **research reproducibility**.

---

## 1. Why dataset splitting matters

A split defines which scans the model **learns from**, which scans tune **hyperparameters**, and which scans provide an **unbiased final evaluation**.

Without a principled split:

- **Metrics lie** — evaluating on training data inflates Dice scores and volume accuracy.
- **Decisions become irreproducible** — different random splits yield different conclusions; comparisons across experiments are invalid.
- **Rare subtypes are misrepresented** — a random split can place almost all EDH cases in training, making validation/test metrics for EDH meaningless.
- **Research credibility fails** — thesis committees and reviewers expect a locked test set and documented methodology.

For BrainHemorrhageAI, the split directly affects:

- Segmentation baseline (Phase 2 training)
- Per-subtype volume validation (reusing Phase 1 volume logic)
- Future severity thresholds and clinical reporting claims

The split is **architecture**, not a preprocessing detail. It should be fixed before any model training begins.

---

## 2. Commonly used split ratios in medical imaging

Medical imaging datasets are typically **orders of magnitude smaller** than natural-image benchmarks. Common practices:

| Pattern | Typical use | Rationale |
|---------|-------------|-----------|
| **80 / 10 / 10** | Datasets with 500+ cases | Enough train data; val/test each ~10% for stable metrics |
| **70 / 15 / 15** | Datasets with 100–300 cases | Larger val/test fractions improve reliability on rare classes |
| **K-fold cross-validation (K=5 or 10)** | Datasets under ~500 cases | Every sample used for validation once; reduces split variance |
| **Leave-one-out (LOO)** | Very small datasets (<30) | Maximum training data per fold; high variance in metrics |
| **Official benchmark split** | Public challenges (e.g. BraTS, ISLES) | Enables comparison to published baselines |

Published segmentation work on small 3D medical datasets (brain hemorrhage, stroke, tumor) frequently uses **hold-out test sets (10–20%)** combined with **cross-validation on the remainder**, or **5-fold CV with a single locked test fold**.

There is no universal rule — the choice depends on **dataset size**, **class distribution**, and **compute budget**. For ~200 cases, literature and frameworks such as nnU-Net favor **cross-validation or generous validation fractions** over a thin 10% validation set.

---

## 3. Comparison of candidate strategies

### 80 / 10 / 10

| | |
|---|---|
| **Split sizes (N=192)** | Train 154 · Val 19 · Test 19 |
| **Pros** | Maximizes training data; familiar ratio; simple to explain |
| **Cons** | Val and test sets are small (~19 scans each); with multi-label data, **EDH may appear in only 1–2 test scans** (~23 × 10% ≈ 2); high metric variance on rare classes |
| **Segmentation quality impact** | Stable for common classes (SDH); **unreliable per-class metrics for EDH** |

### 70 / 15 / 15

| | |
|---|---|
| **Split sizes (N=192)** | Train 134 · Val 29 · Test 29 |
| **Pros** | Val/test ~50% larger than 80/10/10; ~3–4 EDH scans expected per val/test if stratified; better hyperparameter tuning signal |
| **Cons** | 18 fewer training scans than 80/10/10; still a single split — results depend on one partition |
| **Segmentation quality impact** | Slightly lower training data may reduce peak performance, but **more trustworthy evaluation** — preferable for research |

### 5-fold cross-validation

| | |
|---|---|
| **Split sizes (N=192)** | 5 folds × ~38 scans validation; ~154 train per fold |
| **Pros** | Every scan validated once; **reduces split-luck variance**; standard for small medical datasets; supports mean ± std reporting (expected in publications) |
| **Cons** | 5× training cost; requires held-out **test set** separately or nested CV to avoid test leakage; more complex experiment tracking |
| **Segmentation quality impact** | Does not change model capacity; produces **more reliable performance estimates** |

### Summary table

| Criterion | 80/10/10 | 70/15/15 | 5-fold CV |
|-----------|----------|----------|-----------|
| Training scans | 154 | 134 | ~154 per fold |
| Val/test reliability | Low | Medium | High (aggregated) |
| EDH evaluation stability | Poor | Moderate | Good (if stratified) |
| Compute cost | 1× | 1× | ~5× |
| Reproducibility | Easy | Easy | Medium (needs fold manifest) |
| Thesis/publication fit | Acceptable | Good | **Strong** |

---

## 4. Best strategy for BHSD (192 scans)

**Primary recommendation: 70 / 15 / 15 hold-out split with stratification, plus optional 5-fold CV on the training subset for thesis-level reporting.**

### Why not 80/10/10?

With only **23 EDH scans**, a 10% test set expects **~2 EDH cases**. One mis-segmented scan swings EDH Dice by tens of percentage points. Validation on 19 scans is too noisy for early stopping and learning-rate decisions on rare subtypes.

### Why 70/15/15 as the operational split?

- **29 scans** in val and test each — enough for ~4 EDH cases per split if stratified (~23 × 0.15 ≈ 3.5).
- **134 training scans** — still sufficient for a 2D/2.5D U-Net baseline on CPU.
- Clear roles: **train** (weights), **val** (early stopping, hyperparameters), **test** (single final report — **never used during development**).

### Why also consider 5-fold CV?

192 scans sit in the range where **single-split variance is a known problem** in medical imaging literature. For a thesis or publication, reporting **5-fold mean Dice ± std** on the development set (train + val combined, excluding locked test) demonstrates rigor. This is optional for the first baseline milestone but recommended before final results.

### Why a locked test set at all?

Cross-validation alone without a held-out test set risks **implicit overfitting to the validation distribution** across many experiments. A **locked 29-scan test set** provides one unbiased number for the portfolio app and final report.

---

## 5. Should we stratify by hemorrhage subtype?

**Yes — but not by a single subtype column, because BHSD is multi-label.**

Each scan may contain labels `{2, 3}`, `{1, 5}`, etc. Simple stratification on one class (e.g. SDH present/absent) ignores co-occurrence and can still orphan EDH in one split.

### Recommended stratification approach

**Multi-label stratification** (iterative stratification / combinatorial label grouping):

1. For each scan, record the **set of present hemorrhage labels** (1–5), e.g. `{2, 3}`, `{1, 2, 5}`.
2. Group scans by **label combination** (or hash of binary presence vector across 5 subtypes).
3. Assign groups to train/val/test proportionally so each split preserves:
   - **Minimum EDH count** (≥3 per val and test if mathematically possible)
   - Approximate **prevalence of each subtype**
   - Representation of **multi-label combinations** (not only single-class scans)

### Fallback (simpler)

If combination groups are too sparse (many unique combinations with 1 scan):

- Stratify primarily on **EDH presence** (rarest, n=23).
- Secondary constraint: balance **total hemorrhage volume quartiles** (from `volume_statistics.csv`) to avoid putting all large bleeds in training.

### Why stratification matters here

| Without stratification | With stratification |
|------------------------|---------------------|
| Test set may contain **zero EDH** scans | Each split evaluates all subtypes |
| Val loss unstable for rare classes | Early stopping reflects rare-class performance |
| Volume/severity calibration biased | Subtype volume statistics generalize better |

**Mandatory for this project:** at minimum, **EDH-aware stratification**.

---

## 6. Should we use a fixed random seed?

**Yes.**

A fixed seed (e.g. `42`) ensures:

- The same 134/29/29 partition is reproduced across machines and time.
- Preprocessing, training, and evaluation scripts reference one canonical split.
- Thesis and portfolio documentation match the code.

**Caveat:** A single seed gives **one** split. For sensitivity analysis (optional, pre-publication), re-run stratified splitting with 2–3 additional seeds and report metric variance. This is research best practice but not required for the first baseline.

---

## 7. How to preserve reproducibility

| Artifact | Purpose |
|----------|---------|
| `data/metadata/splits.csv` | Canonical record: `filename`, `split` (`train` / `val` / `test`) |
| `docs/architecture/data_split_design.md` | This document — rationale and constraints |
| `data/metadata/split_config.json` | Seed, ratio, stratification method, creation timestamp, script version |
| Git commit of split files | Split becomes version-controlled project state |
| Test set lock policy | Document: *test scans never used for augmentation tuning, threshold selection, or model selection* |

### Reproducibility rules

1. **Generate the split once**, review EDH/subtype counts per split, then **commit** the manifest.
2. **Never regenerate** the split mid-project unless a data bug is discovered.
3. Log split membership in **every training run** (config or experiment log).
4. Preprocessing caches (`data/processed/`) should include split in metadata or directory structure to avoid accidental cross-split leakage during debugging.
5. Report **per-class Dice** on val and test separately — not only mean Dice.

---

## 8. Risks

### Class imbalance

- **Background dominates** (~99%+ voxels) — splits do not fix this, but stratification prevents **subtype absence** in val/test.
- **EDH (n=23)** — metrics will remain high-variance; report confidence intervals or bootstrap CIs on test.
- **Mitigation:** Stratified split; weighted loss; hemorrhage-aware slice sampling (training design); report per-class metrics.

### Patient / study leakage

BHSD filenames encode identifiers (e.g. `ID_0237f3c9_ID_40015688b9.nii.gz`). If multiple scans belong to the **same patient or admission**, placing them in different splits **inflates performance** (the model sees correlated anatomy).

| Action | Priority |
|--------|----------|
| Consult BHSD paper/documentation for patient-level grouping | **High** |
| If patient ID available, split at **patient level**, not scan level | **Mandatory if duplicates exist** |
| If one scan per patient (likely for BHSD), document assumption explicitly | Required for thesis |

*Note: Phase 1 analysis treated 192 scans as 192 independent cases. Phase 2 must verify this against BHSD documentation before locking the split.*

### Split variance (single partition)

One 70/15/15 split may lucky or unlucky. Mitigation: optional 5-fold CV on train+val; report sensitivity to seed.

### Test set contamination

Using the test set for:

- choosing augmentation strength  
- tuning post-processing  
- selecting best epoch by peeking at test  

…invalidates results. **Test = one evaluation at the end** (or fixed schedule documented upfront).

### Small validation set clinical generalization

Even 29 test scans may not represent all scanners and spacing variants in BHSD. External validation (future data or unlabeled set after semi-supervised phase) remains a long-term goal.

---

## 9. Final recommendation

### Adopted strategy

| Decision | Choice |
|----------|--------|
| **Split ratio** | **70 / 15 / 15** (134 train · 29 val · 29 test) |
| **Stratification** | **Multi-label**, EDH-presence guaranteed in all splits |
| **Random seed** | **Fixed** (`42` unless sensitivity analysis) |
| **Manifest** | `data/metadata/splits.csv` + `split_config.json` |
| **Test set policy** | Locked — no training decisions based on test |
| **Thesis / publication** | Optional **5-fold CV** on 163 development scans (train+val) before final test evaluation |

### Expected subtype distribution (approximate, if stratification succeeds)

| Split | Scans | EDH scans (target) |
|-------|-------|---------------------|
| Train | 134 | ~16 |
| Val | 29 | ~3–4 |
| Test | 29 | ~3–4 |

Exact counts must be verified at split generation time and recorded in `split_config.json`.

### What we explicitly defer

- Using `unlabel_1980` in the split (Phase 3+ semi-supervised)
- Implementation code (next milestone)
- K=10 fold CV (diminishing returns vs 5× compute on 192 scans)

### Success criteria for the split (before training)

- [ ] All 192 scans assigned exactly once  
- [ ] EDH present in train, val, and test (≥2 scans minimum in val and test)  
- [ ] Each subtype 1–5 appears at least once in val and test  
- [ ] Patient-level leakage checked against BHSD documentation  
- [ ] Manifest committed; test filenames documented as locked  

---

## Why 5-Fold Cross Validation is Deferred

Although 5-fold cross validation provides a more robust estimate of model performance on small datasets, it increases computational cost by approximately five times because the model must be trained independently for each fold.

The primary goal of Version 1 is to establish a stable and reproducible end-to-end pipeline:

Dataset  
→ Preprocessing  
→ Training  
→ Evaluation  
→ Inference  
→ Clinical Rule Engine  
→ Web Application

Introducing cross-validation before validating this baseline would significantly slow development and complicate debugging.

Once the baseline model is stable, 5-fold cross validation will be introduced for the final research evaluation and thesis experiments.

Therefore:

**Version 1:**  
Single fixed stratified split (70/15/15)

**Research Version:**  
5-fold cross validation

---

## Dataset Split Freeze Policy

Once the split has been generated and validated, it becomes part of the project.

The split will never be regenerated unless:

- dataset corruption is discovered
- duplicated patients are identified
- annotation errors are corrected

Changing the split invalidates previous experiments and makes model comparisons unreliable.

Every future model (MONAI U-Net, nnU-Net, SwinUNETR, etc.) must use the exact same dataset split.

---

## References and rationale (research practice)

- Small-sample medical segmentation commonly uses **cross-validation or ≥15% validation** rather than thin 10% holds (nnU-Net framework; MICCAI challenge protocols).
- **Stratified splitting** for imbalanced classification is standard in clinical ML reporting (TRIPOD+, CONSORT-AI extensions for transparent evaluation).
- **Held-out test sets** remain best practice even when CV is used for development — prevents optimistic bias from repeated validation peeking.
- Multi-label datasets require **iterative stratification** rather than single-label `train_test_split(stratify=...)`.

---

*Next milestone: implement split generation script that materializes `splits.csv` according to this design.*
