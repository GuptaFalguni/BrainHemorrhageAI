# Experimental Protocol — BrainHemorrhageAI

**Project:** BrainHemorrhageAI  
**Phase:** Research Protocol (Phase 3)  
**Status:** Governing document for all training and evaluation  
**Version:** 1.0  
**Related documents:**

- `docs/architecture/data_split_design.md`
- `docs/architecture/windowing_normalization_design.md`
- `docs/architecture/resampling_design.md`
- `docs/architecture/training_pipeline_design.md`
- `docs/architecture/model_design.md`
- `docs/research/windowing_literature_review.md`

---

## Document purpose

This protocol defines **how** experiments are conducted, **what** is measured, and **when** a result is considered valid or publishable. Every future training run, ablation, and benchmark must conform to this document unless a protocol amendment is explicitly approved and versioned.

---

## 1. Research objectives

### 1.1 Primary objective

Develop and validate an automated pipeline that performs **multi-class intracranial hemorrhage (ICH) segmentation** on non-contrast brain CT, producing:

- Voxel-wise labels for five hemorrhage subtypes (EDH, SDH, SAH, IPH, IVH) plus background  
- **Native-space volume estimates (mL)** per subtype and in total  
- Outputs suitable for a clinical decision-support report (severity, confidence — downstream)

The primary measurable goal for Version 1 is a **reproducible MONAI 2.5D U-Net baseline** with documented segmentation and volume metrics on the **locked test split**.

### 1.2 Secondary objectives

| ID | Objective |
|----|-----------|
| S1 | Quantify per-subtype segmentation performance, especially rare classes (EDH) |
| S2 | Evaluate **clinical plausibility** via native-space volume error against ground truth |
| S3 | Establish an end-to-end path: raw CT → preprocessing → inference → 3D mask → volume → report fields |
| S4 | Build a comparison framework for future models (nnU-Net, semi-supervised) on **identical splits and metrics** |
| S5 | Document compute requirements (CPU development, optional GPU training) for reproducibility |

### 1.3 Research questions

| RQ | Question |
|----|----------|
| **RQ1** | Can a **2.5D U-Net** achieve clinically useful multi-class hemorrhage segmentation on BHSD with limited hardware? |
| **RQ2** | How do segmentation errors propagate to **subtype volume (mL)** and total hemorrhage burden? |
| **RQ3** | Which subtypes are most difficult under fixed preprocessing (EDH, SAH, IVH)? |
| **RQ4** | Does **BHSD literature preprocessing** outperform alternative windowing/normalization on our split? |
| **RQ5** | (Future) Does **semi-supervised learning** with unlabeled BHSD volumes improve rare-subtype performance? |

### 1.4 Hypotheses

| ID | Hypothesis | Testable when |
|----|------------|---------------|
| **H1** | A 2.5D MONAI U-Net baseline will achieve **non-zero, stable mean Dice (classes 1–5)** on the validation set sufficient for iterative improvement | Version 1 training complete |
| **H2** | **BHSD clip `[-40, 120]` HU** with `(HU−40)/80` normalization will match or exceed alternative fixed windows (e.g. `[0, 100]`) on mean hemorrhage Dice | Ablation A1 |
| **H3** | **Native-space volume MAE** will correlate with segmentation Dice at the scan level (lower Dice → higher volume error) | Version 1 evaluation |
| **H4** | EDH (n=23 dataset-wide) will show **highest metric variance** across scans due to class imbalance | Per-class analysis |
| **H5** | (Future) nnU-Net 3D with GPU will outperform 2.5D baseline on mean Dice but at higher compute cost | Research Version |

*Hypotheses are falsifiable using the metrics and splits defined in Sections 5–6.*

---

## 2. Dataset

### 2.1 BHSD labeled dataset (`label_192`)

| Property | Value |
|----------|--------|
| **Source** | Brain Hemorrhage Segmentation Dataset (BHSD) [Wu et al., arXiv:2308.11298] |
| **Location** | `data/raw/label_192/` |
| **Volumes** | 192 paired CT + multi-class masks |
| **Classes** | 0 = background; 1 = EDH; 2 = SDH; 3 = SAH; 4 = IPH; 5 = IVH |
| **Format** | NIfTI (`.nii.gz`); native spacing variable (Phase 1 analysis) |
| **Use in Version 1** | Supervised training, validation, test |

Multi-label scans (multiple subtypes per volume) are expected. Stratification at split time accounts for label co-occurrence.

### 2.2 BHSD unlabeled dataset (`unlabel_1980`) — future work

| Property | Value |
|----------|--------|
| **Volumes** | 1980 CT scans without pixel-level masks |
| **Status** | **Excluded from Version 1** per project scope |
| **Planned use** | Semi-supervised / pseudo-label experiments after supervised baseline (Section 4) |

No unlabeled data may enter training until a **protocol amendment** documents leakage controls and baseline comparison.

### 2.3 Current split strategy

| Split | Ratio | Scans (N=192) | Role |
|-------|-------|---------------|------|
| **Train** | 70% | 134 | Weight learning; normalization stats policy per preprocessing doc |
| **Validation** | 15% | 29 | Hyperparameter tuning, early stopping, checkpoint selection |
| **Test** | 15% | 29 | **Single final evaluation** — locked |

**Method:** Two-step **iterative multi-label stratification** (`MultilabelStratifiedShuffleSplit`, seed **42**) preserving subtype presence in val/test.  
**Manifest:** `data/metadata/splits.csv`  
**Metadata:** `data/metadata/split_config.json`

**5-fold cross-validation** is **deferred** to Research Version (see `data_split_design.md`).

### 2.4 Split freeze policy

Once generated and validated, the split is **frozen**. It will **not** be regenerated unless:

- Dataset corruption is discovered  
- Duplicate patients are identified  
- Annotation errors require corrected masks  

All models (Version 1 and future) **must use the same split**. Test scans **must not** influence model selection, augmentation tuning, or threshold calibration.

---

## 3. Preprocessing protocol

Summary of intensity and spatial decisions. Full rationale in architecture and literature documents.

### 3.1 Approved (Version 1 — locked)

| Step | Decision | Reference |
|------|----------|-----------|
| **Pipeline order** | Clip/window in native HU → normalization → (spatial steps when approved) | `windowing_normalization_design.md` |
| **HU clip** | **`[-40, 120]`** | Wu et al., BHSD [literature review] |
| **Normalization** | **Z-score:** `(HU − 40) / 80` after clip | BHSD §3.0.2; `config/preprocessing.yaml` |
| **Mask handling** | Integer labels `{0,…,5}`; no intensity transform | Architecture docs |
| **Volume reporting** | Always **native spacing** from original CT header; not resampled grid | `resampling_design.md` |
| **Train-only leakage rule** | Normalization constants are **literature-fixed** for V1; empirical train stats may be reported as ablation only | Literature review |

**Configuration file:** `config/preprocessing.yaml`

### 3.2 Under evaluation (not locked for Version 1 reporting)

| Item | Candidates | Selection criteria |
|------|------------|-------------------|
| **Target resampling spacing** | 1 mm isotropic (leading); anisotropic alternatives | Memory, BHSD anisotropy, nnU-Net compatibility |
| **Input spatial resolution** | 512², 256², brain ROI crop | Hemorrhage bbox analysis, CPU/GPU memory, val Dice |
| **Augmentation policy** | Flip, rotation, HU jitter — strengths TBD | Val stability, EDH performance |

Version 1 **baseline results** may be reported **without resampling** if spatial pipeline is not locked before first training; any resampling change requires re-benchmarking and protocol note.

### 3.3 Future experiments (documented, not Version 1)

| Experiment | Description |
|------------|-------------|
| Alternative HU windows | RSNA multi-window `[0,80]`, MONAI `[0,100]`, nnU-Net percentile clip |
| Empirical vs fixed normalization | Train-split μ/σ vs BHSD fixed 40/80 |
| Skull stripping / brain ROI | Pre-clip spatial mask |
| Full isotropic resample vs slice-native 2.5D | Interaction with anisotropic z-spacing |

---

## 4. Model development roadmap

### 4.1 Version 1 — MONAI 2.5D U-Net baseline (current)

| Component | Specification |
|-----------|---------------|
| **Input** | 3 adjacent axial slices → 3 channels |
| **Output** | Center-slice 6-class segmentation |
| **Architecture** | MONAI 2D `UNet` |
| **Loss** | Dice + Cross-Entropy; class weights for imbalance |
| **Selection metric** | Macro mean Dice (classes 1–5) on validation |
| **Inference** | Slice loop → stack 3D → inverse resample to native (when spatial pipeline active) |

**Why first:** Only path that **trains on local CPU**, matches approved pipeline, establishes **fair benchmark** for all future work, and delivers **end-to-end product** (upload → volume → report fields).

### 4.2 Future versions (scheduled order)

| Order | Model / method | Rationale for sequencing |
|-------|------------------|---------------------------|
| **V2** | **Attention U-Net** (2.5D) | Low incremental cost; tests whether attention helps **small EDH/SAH** without new infrastructure |
| **V3** | **nnU-Net** | Strongest **supervised** CT benchmark; requires **GPU + finalized resampling**; compare directly to V1 on **same split** |
| **V4** | **SwinUNETR** (3D patches) | Transformer capacity; needs **GPU + larger compute**; justified only if CNN baselines plateau |
| **V5** | **Semi-supervised** (`unlabel_1980`) | Requires **stable supervised baseline** and pseudo-label QA; 1980 unlabeled volumes add complexity and leakage risk if rushed |

**Deferred within Version 1:** UNet++ (overlap with Attention U-Net + nnU-Net), pure 2D single-slice (ablation only), full 3D U-Net on CPU (impractical).

Each version **must** reference the same `splits.csv`, report all Section 5 metrics, and log Section 9 reproducibility artifacts.

---

## 5. Evaluation metrics

All segmentation metrics computed on **validation/test** with **all slices** (no balanced sampling). Volume metrics in **native space** (Phase 1 `calculate_volume.py` convention).

### 5.1 Segmentation metrics

| Metric | Definition | Primary use |
|--------|------------|-------------|
| **Dice** | `2\|A∩B\| / (\|A\|+\|B\|)` per class | **Primary** model selection (macro mean, classes 1–5) |
| **IoU (Jaccard)** | `\|A∩B\| / \|A∪B\|` | Stricter overlap; small lesions |
| **Precision** | TP / (TP + FP) | False-positive control (skull edge) |
| **Recall** | TP / (TP + FN) | Sensitivity; critical for **EDH/SAH** |

**Reporting:** Per-class (1–5) **and** macro averages. Background (0) reported separately — **not** used for checkpoint selection.

### 5.2 Clinical metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| **Volume error (mL)** | `\|V_pred − V_gt\|` per scan | Total hemorrhage (classes 1–5) and **per subtype** |
| **Relative volume error** | `\|V_pred − V_gt\| / V_gt` | Report when `V_gt > 0` only |
| **Severity agreement** | Agreement with rule-engine tier | **Future** — protocol placeholder: Cohen's κ or weighted κ once severity thresholds are literature-backed and frozen |

Clinical metrics use **inverse-resampled predictions** on the **native grid** with **original voxel spacing**.

### 5.3 Model / system metrics

| Metric | Definition |
|--------|------------|
| **Inference time** | Wall-clock per scan (slice loop + stack + optional resample) |
| **Model size** | Parameter count; checkpoint file size (MB) |
| **Confidence score** | Mean max-softmax over predicted hemorrhage voxels (Version 1 definition per `model_design.md`) |

Report hardware (CPU model / GPU model) with inference time.

---

## 6. Statistical analysis

### 6.1 Principles

- **N = 192** labeled scans → **small sample**; avoid over-claiming from single test splits.  
- **Test set (n=29)** is evaluated **once** per major protocol version.  
- Prefer **descriptive statistics + confidence intervals** over multiple hypothesis tests on the same test data.

### 6.2 Reporting format

For each metric on validation or test:

| Statistic | Application |
|-----------|-------------|
| **Mean** | Average across scans (macro over scans for volume; macro over classes for Dice) |
| **Standard deviation** | Scan-level variability |
| **Median** | Robust central tendency for skewed volume errors |
| **95% CI** | **Bootstrap percentile CI** (scan-level resampling, ≥10,000 iterations) for mean Dice and mean volume error |

Report **per-class** Dice/IoU/precision/recall with mean ± SD across scans where class is present in ground truth.

### 6.3 Comparing experiments

| Comparison type | Allowed method | Justification |
|-----------------|----------------|---------------|
| **Two models, same test scans** | **Paired analysis** on per-scan metric differences; **Wilcoxon signed-rank** if distributional assumptions unclear | Same 29 patients; paired by scan ID |
| **Single model, single test split** | Descriptive only (mean ± SD, CI) | Insufficient power for multiple testing |
| **Cross-validation (Research Version)** | Mean ± SD **across folds** | Justified when 5-fold CV is implemented |

**Not permitted without protocol amendment:** Repeated peeking at test set; claiming significance from validation-only results as final evidence.

### 6.4 Per-class comparison

Rare classes (EDH) must be reported **separately**. Pooling all subtypes into one "hemorrhage" class is permitted as a **secondary** summary only, not primary.

---

## 7. Ablation studies

Each ablation varies **one factor**; all others fixed per Version 1 baseline.

| ID | Ablation | Purpose |
|----|----------|---------|
| **A1** | **BHSD preprocessing** vs `[0,100]` clip + min-max vs nnU-Net-style percentile clip | Test **H2**; justify literature-fixed BHSD window |
| **A2** | **Input resolution** 512² vs 256² vs brain ROI | Lock spatial resolution (`training_pipeline_design.md`) |
| **A3** | **Loss:** DiceCE only vs + class weights vs focal loss | EDH / imbalance sensitivity |
| **A4** | **Architecture:** pure 2D vs 2.5D vs (GPU) 3D patch | Quantify z-context benefit |
| **A5** | **Data augmentation:** none vs flip/rotate vs + HU jitter | Overfitting on n=134 train |
| **A6** | **Resampling:** native vs 1 mm isotropic (when approved) | Anisotropy vs compute |
| **A7** | **Semi-supervised** with `unlabel_1980` (Research Version) | Test **H5**; label efficiency |

Ablation results are reported on **validation** unless pre-registered for test evaluation with protocol approval.

---

## 8. Success criteria

### 8.1 Version 1 success (minimum viable research system)

Version 1 is **successful** if all of the following are met:

| Criterion | Threshold / condition |
|-----------|------------------------|
| **Reproducibility** | Full Section 9 checklist completed for one training run |
| **Training stability** | Converges without numerical failure; val loss bounded |
| **Segmentation** | Non-trivial **macro mean Dice (1–5) > 0** on val; per-class Dice reported for all five subtypes |
| **EDH** | At least **one EDH case** in val/test evaluated with documented Dice (not ignored) |
| **Volume** | Native-space **total hemorrhage volume MAE (mL)** computed on val |
| **End-to-end** | Inference produces 3D mask + per-subtype volumes for a held-out scan |
| **Test evaluation** | Test split evaluated **once** with frozen `best_model.pt` |

Version 1 **does not require** SOTA nnU-Net performance or clinical deployment readiness.

### 8.2 Publishable improvement (Research Version bar)

An improvement is **publishable / thesis-worthy** if it demonstrates **all** of:

1. **Statistically supported gain** on **locked test set** (paired Wilcoxon or bootstrap CI non-overlapping for primary metric) **or** consistent gain across **5-fold CV** with reported variance  
2. **Clinically relevant** reduction in **volume MAE (mL)** or meaningful EDH/SAH recall gain  
3. **Reproducibility** package (Section 9) for reviewers  
4. **Ablation evidence** isolating the claimed contribution (not confounded preprocessing or split changes)  
5. Comparison against **Version 1 baseline** and, when available, **BHSD published benchmarks** [Wu et al.] with differences in split explicitly acknowledged  

---

## 9. Reproducibility checklist

Required for **every** reported experiment:

| Item | Location / requirement |
|------|------------------------|
| **Random seed** | Config: `random_seed=42` (numpy, torch, dataloaders) |
| **Split** | `data/metadata/splits.csv` hash logged |
| **Preprocessing** | `config/preprocessing.yaml` hash; literature review version |
| **Model config** | Architecture depth, channels, `in_channels=3`, `out_channels=6` |
| **Training config** | Batch size, epochs, optimizer, LR, loss weights |
| **Checkpoint** | `checkpoints/best_model.pt` + `best_model_metadata.json` |
| **Software** | Python, PyTorch, MONAI, nibabel versions in `reports/training/<run_id>/environment.txt` |
| **Hardware** | CPU/GPU model, RAM, training wall-clock |
| **Run ID** | `reports/training/<run_id>/` with `config.yaml`, `metrics.csv` |

**Git:** Commit hash recorded at training start. Split and config files must not change mid-run.

---

## 10. Future work (intentionally deferred)

| Area | Rationale for deferral |
|------|------------------------|
| **`unlabel_1980` semi-supervised** | Requires stable supervised baseline and leakage-safe protocol |
| **Transformers (SwinUNETR)** | Data and GPU requirements; CNN baselines first |
| **Self-supervised pretraining** | No protocol for downstream transfer evaluation yet |
| **5-fold cross-validation** | Compute cost; Version 1 prioritizes end-to-end pipeline |
| **Deployment optimization** | ONNX/TorchScript, API latency — after model validity |
| **Explainability (Grad-CAM, attention maps)** | After Attention U-Net / clinical review needs |
| **Uncertainty estimation** | Deep ensembles, MC dropout — Research Version |
| **Severity rule engine calibration** | Literature review for thresholds not complete |
| **External validation** | No second hospital dataset in Version 1 |
| **Resampling finalization** | Pending spatial architecture sign-off |

---

## Protocol amendments

Changes to split, test-set usage, primary metric, or preprocessing **after Version 1 baseline** require:

1. Dated amendment section appended to this document  
2. Version bump (e.g. 1.0 → 1.1)  
3. Explicit statement of impact on prior results  

---

## Approval

| Role | Status | Date |
|------|--------|------|
| Research protocol v1.0 | Pending approval | — |

---

*This document governs all experiments until superseded. Implementation must not deviate without amendment.*
