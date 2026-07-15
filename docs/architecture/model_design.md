# Model Design — BHSD (Phase 2)

> **Status: Approved and Frozen**  
> Version 1 model is MONAI 2D U-Net (2.5D input). Kept for thesis reviewers.

**Project:** BrainHemorrhageAI  
**Dataset:** BHSD — 192 labeled CT volumes, 6-class segmentation (background + 5 hemorrhage subtypes)  
**Status:** Approved and Frozen (Version 1 model: MONAI 2D U-Net)  
**Training mode:** 2.5D input → **MONAI 2D U-Net** (per `training_pipeline_design.md`)  
**Hardware context:** AMD Ryzen 5 7535U · 16 GB RAM · no local CUDA · Kaggle GPU optional  
**Last updated:** Phase 2 — final architecture milestone (freeze banner added during V2 Phase A cleanup)

---

## Executive summary

**Version 1 model:** **MONAI 2D U-Net** — three-channel 2.5D axial input, six-class softmax output, **Dice + Cross-Entropy** loss, multi-metric validation including **native-space volume error (mL)**.

**Deferred:** Attention U-Net, UNet++, nnU-Net, SwinUNETR (Research Version / GPU benchmarks).

This document finalizes model I/O, loss, metrics, confidence, checkpointing, and experiment tracking for implementation.

---

## Why this milestone matters

The model block is the **core AI component** of BrainHemorrhageAI. Architecture choice affects:

- Whether training completes on **local CPU** within days, not weeks  
- Whether **EDH** (23/192 scans) receives usable gradient signal  
- Whether **confidence scores** and **volume mL** in the clinical report are defensible  
- Whether future models (nnU-Net) compare fairly on **identical splits and preprocessing**

A publication-quality system documents **why** the baseline was chosen and **what** will supersede it — not only which library call to use.

---

## 1. Candidate model comparison

### Overview

| Model | Type | Typical input | Params (indicative) |
|-------|------|---------------|---------------------|
| **MONAI 2D U-Net** | Encoder–decoder CNN | 2D / 2.5D slices | ~5–20 M (depth-dependent) |
| **Attention U-Net** | U-Net + attention gates | 2D / 2.5D | ~10–30 M |
| **UNet++** | Nested dense skip U-Net | 2D / 3D | ~20–40 M |
| **nnU-Net** | Self-configuring 3D pipeline | 3D patches / full resample | Auto (often 30M+) |
| **SwinUNETR** | Swin Transformer + U-Net decoder | 3D patches | ~60 M+ |

---

## 2. Multi-criteria comparison (BHSD + project constraints)

Legend: ●●● strong · ●● moderate · ● weak · ✗ poor fit

| Criterion | MONAI 2D U-Net | Attention U-Net | UNet++ | nnU-Net | SwinUNETR |
|-----------|----------------|-----------------|--------|---------|-----------|
| **Accuracy potential** | ●● | ●●● | ●●● | ●●●● | ●●●● |
| **Computational cost (train)** | ●●● low | ●● | ●● | ● high | ● high |
| **Memory (inference)** | ●●● | ●● | ●● | ●● | ● |
| **Ease of implementation** | ●●● | ●● | ●● | ●● (framework) | ● |
| **BHSD 192 scans** | ●●● | ●● (overfit risk) | ●● | ●●● | ● (too data-hungry) |
| **CPU training** | ●●● | ●● | ●● | ✗ | ✗ |
| **Kaggle GPU** | ●●● | ●●● | ●●● | ●●● | ●● |
| **Clinical deployment** | ●●● | ●●● | ●● | ●● (heavy) | ● (latency) |
| **MONAI ecosystem fit** | ●●● | ●●● | ●● | ● (wrapper) | ●●● |
| **Thesis reproducibility** | ●●● | ●● | ●● | ●●● | ●● |

### Accuracy

- **nnU-Net / SwinUNETR** set SOTA on multi-site CT when GPU and data scale permit.  
- **MONAI 2D U-Net** achieves **competitive Dice** on hemorrhage slice tasks within ~2–5% of heavier 3D models in literature when preprocessing is sound.  
- **Attention U-Net / UNet++** may gain ~1–3% Dice on small lesions but increase overfitting on **192 scans**.

### Computational cost

- **2D U-Net:** one forward pass per slice — **O(H×W)** per slice.  
- **3D models:** **O(H×W×D)** or patch sliding — 10–100× slower on CPU.

### Memory usage

- **2.5D U-Net at H×W (TBD):** primary RAM consumer is batch × 3 × H × W.  
- **SwinUNETR 3D:** typically **>16 GB** even for moderate patches.

### Ease of implementation

- **MONAI UNet:** documented, `UNet` + `DiceCELoss` + transforms — minimal custom code.  
- **nnU-Net:** separate install, dataset conversion, plans — high setup cost.  
- **SwinUNETR:** transformer hyperparameters, pretraining expectations — research overhead.

### Suitability for BHSD

- Multi-class **6-way** softmax maps directly to labels 0–5.  
- **Class imbalance** handled via loss weights + slice sampling (pipeline doc).  
- **Anisotropic** data: 2.5D + 2D U-Net is **standard** when z-spacing ≫ in-plane.

### CPU vs Kaggle GPU

| Environment | MONAI 2D U-Net | Others |
|-------------|----------------|--------|
| **Local CPU** | **Primary development** | Attention/UNet++ possible but slower; 3D impractical |
| **Kaggle T4** | **Fast baseline training** | nnU-Net/SwinUNETR for benchmark runs |

### Clinical deployment

- **2D U-Net:** TorchScript/ONNX export straightforward; slice loop at inference matches training.  
- **nnU-Net:** bundle larger; GPU expected in production.  
- **SwinUNETR:** latency and memory limit edge deployment without GPU.

---

## 3. Why MONAI 2D U-Net is selected for Version 1

1. **Approved training pipeline** — 2.5D three-channel input maps natively to `in_channels=3`.  
2. **Hardware reality** — only architecture that **reliably trains on local CPU** for method development.  
3. **Small dataset (192)** — parameter-efficient encoder–decoder reduces overfitting vs transformers.  
4. **Project stack** — PyTorch + MONAI per `PROJECT_CONTEXT.md`; single codebase for train and serve.  
5. **Research baseline** — established reference point; all future models compare against **same split, metrics, volume protocol**.  
6. **Clinical path** — slice inference → 3D stack → native volume (locked policy) without patch stitching.  
7. **Literature alignment** — U-Net remains the **most cited baseline** in medical image segmentation (Ronneberger et al.; MICCAI challenge baselines).

**Configuration principle:** start **shallow** (e.g. 3 encoder levels, base channels 32–64) — increase depth only if val Dice plateaus and overfitting is absent.

---

## 4. Why remaining models are deferred

| Model | Deferred to | Reason |
|-------|-------------|--------|
| **Attention U-Net** | Research Version | Marginal accuracy gain; more parameters; attention maps useful for thesis figures but not v1 blocker |
| **UNet++** | Research Version | Dense skip connections increase memory/training time; overfitting risk on 192 scans |
| **nnU-Net** | Research Version / Kaggle | **Best benchmark** once GPU available; requires finalized resampling + folder structure; not for CPU-first iteration |
| **SwinUNETR** | Research Version | Transformer needs **large data + GPU**; BHSD n=192 insufficient for reliable pretraining benefit in v1 |

**Policy:** Version 1 must produce a **complete end-to-end system** (upload → report). MONAI 2D U-Net is the **minimum viable research-grade baseline**; deferred models are **evaluation upgrades**, not blockers.

---

## 5. Model specification

### Input

| Property | Specification |
|----------|---------------|
| **Modality** | Non-contrast brain CT (preprocessed HU) |
| **Format** | 2.5D stack of **3 adjacent axial slices** |
| **Channels** | **3** (slice i−1, i, i+1; edge replication) |
| **Spatial size** | **H × W** — resolution **under evaluation** (512², 256², or brain ROI) |
| **Intensity** | After clip/window (TBD) + train z-score |
| **Batch** | `(N, 3, H, W)` float32 |

### Output

| Property | Specification |
|----------|---------------|
| **Type** | Per-pixel **multi-class classification** |
| **Spatial** | Same H × W as input (center slice) |
| **Training** | Logits `(N, 6, H, W)` |
| **Inference** | Softmax → argmax → label map `{0,…,5}` |

### Output channels (6 classes)

| Channel / class index | Label | Subtype |
|-------------------------|-------|---------|
| 0 | Background | — |
| 1 | EDH | Epidural |
| 2 | SDH | Subdural |
| 3 | SAH | Subarachnoid |
| 4 | IPH | Intraparenchymal |
| 5 | IVH | Intraventricular |

**Softmax over 6 channels** — mutually exclusive classes per voxel (matches BHSD annotation scheme).

### Loss function

**Dice + Cross-Entropy (DiceCE)** — MONAI `DiceCELoss`

```text
L = λ_dice · DiceLoss(logits, one_hot_mask) + λ_ce · CrossEntropy(logits, class_indices)
```

| Component | Role |
|-----------|------|
| **Dice** | Directly optimizes overlap — critical for **small hemorrhage regions** |
| **Cross-Entropy** | Stable pixel-wise gradients; class boundaries |
| **Class weights** | Higher weight on channels **1 (EDH)** and underrepresented subtypes — computed from **train split** voxel frequencies |

**Default weights:** `λ_dice = 1`, `λ_ce = 1` (tune if CE dominates on background).

**Background:** included in loss — model must learn skull/brain/background vs hemorrhage; optional `include_background=False` in Dice for hemorrhage-only Dice metric, not necessarily for loss ablation in v1.

### Metrics

All metrics computed on **validation/test** with **all slices** (no balanced sampling).

| Metric | Definition | Primary use |
|--------|------------|-------------|
| **Dice score** | `2|A∩B| / (|A|+|B|)` per class | **Primary model selection** (mean over classes 1–5) |
| **IoU (Jaccard)** | `|A∩B| / |A∪B|` | Secondary; stricter on small objects |
| **Precision** | TP / (TP + FP) | False positive control (skull edge) |
| **Recall** | TP / (TP + FN) | Sensitivity — critical for **EDH/SAH** |
| **Volume error (mL)** | `|V_pred − V_gt|` per class and total | **Clinical relevance** — native-space volumes only |

**Reporting convention (thesis / portfolio):**

- Report **mean ± std** over val scans for Dice/IoU (classes 1–5 separately + macro mean).  
- Report **median absolute volume error (mL)** for total hemorrhage and per subtype.  
- Never select model on **background Dice alone**.

**Volume error protocol:**

1. Stack slice predictions → 3D label volume (preprocessed grid).  
2. Inverse resample to **native grid** (nearest).  
3. Apply Phase 1 volume formula with **original spacing**.  
4. Compare to `volume_statistics.csv` ground truth.

---

## 6. Confidence scores

Clinical report requires a **confidence score** (per `PROJECT_CONTEXT.md`). Version 1 definition:

### Per-voxel confidence

After softmax:

```text
confidence_voxel = max_k P(class k | voxel)
```

### Per-class confidence (subtype presence)

For each hemorrhage class **k ∈ {1,…,5}**:

```text
confidence_class_k = mean(confidence_voxel) over voxels where argmax == k
```

If class **k** absent in prediction: `confidence_class_k = 0` (or max P(k) over all voxels as **detection confidence** — document which in implementation).

**Recommended for report:**

| Score | Formula | Interpretation |
|-------|---------|----------------|
| **Segmentation confidence** | Mean max-softmax over **predicted hemorrhage voxels** (union of classes 1–5) | Overall model certainty on bleed regions |
| **Subtype confidence** | Mean max-softmax for voxels assigned to class k | Per-subtype report line |
| **Study-level flag** | Low if mean confidence < threshold τ (τ from val set calibration — **not hardcoded**) | Triggers “manual review recommended” in UI |

### Calibration (Research Version)

- Temperature scaling on validation logits.  
- Expected calibration error (ECE) for thesis.

**Version 1:** raw softmax statistics are **acceptable** with documented limitation; threshold τ derived from validation distribution (e.g. 5th percentile of mean hemorrhage confidence on correctly segmented cases).

### What confidence is not

- Not a **clinical diagnosis probability** — it is model epistemic output only.  
- Not derived from **volume** or rule engine — separate pipeline stage.

---

## 7. Checkpointing and model selection

### Saved artifacts

| File | Content |
|------|---------|
| `checkpoints/best_model.pt` | Weights at **best val macro Dice (classes 1–5)** |
| `checkpoints/last_model.pt` | Weights at final epoch |
| `checkpoints/best_model_metadata.json` | Epoch, metrics, git hash, config paths |

Each checkpoint includes:

- `model_state_dict`  
- `optimizer_state_dict` (for resume)  
- `epoch`, `val_metrics` snapshot  
- Hashes of `splits.csv`, `preprocess_config.json`  

### Model selection rule

**Primary criterion:** highest **mean Dice over hemorrhage classes 1–5** on validation set (macro average).

**Tie-breakers (in order):**

1. Higher **EDH (class 1) Dice** — rarest critical class  
2. Lower **total hemorrhage volume MAE (mL)** on val (native space)  
3. Lower **mean false positive rate** on background-only slices  

**Early stopping:** patience **15–20 epochs** without improvement in primary criterion; max epochs **150–200** (CPU may use fewer epochs with checkpoint resume on Kaggle).

### Test set

**Locked** — evaluate `best_model.pt` **once** after training completes; never used for checkpoint selection.

---

## 8. Experiment tracking

### Version 1 tracking (minimal, reproducible)

| Artifact | Location | Contents |
|----------|----------|----------|
| **Run config** | `reports/training/<run_id>/config.yaml` | All hyperparameters, paths, resolution, batch size |
| **Metrics log** | `reports/training/<run_id>/metrics.csv` | Per-epoch train loss, val Dice/IoU/volume error |
| **Curves** | `reports/training/<run_id>/` | Optional PNG — loss, mean Dice |
| **Environment** | `reports/training/<run_id>/environment.txt` | Python, torch, MONAI versions |
| **Seed** | Config | Fixed `random_seed=42` for numpy/torch |

### Run naming

```text
v1_unet2d_25d_<resolution>_<date>_<run_id>
```

### Research Version extensions

- Weights & Biases / MLflow (optional)  
- Git tag per benchmark model  
- 5-fold CV run ID linked to fold index  

### Reproducibility rules

1. No change to `splits.csv` mid-run.  
2. Document **exact MONAI and PyTorch versions**.  
3. Save **best** checkpoint only after val sweep — not manual overwrite.  
4. Link each run to **preprocess manifest hash**.

---

## 9. Version 1 training objective

### Primary objective

Learn a mapping:

```text
f: (3-slice CT window) → 6-class segmentation of center slice
```

that **maximizes macro Dice across hemorrhage subtypes 1–5** on the BHSD validation split, subject to **acceptable native-space volume error** on validation scans.

### Secondary objectives

| Objective | Target (illustrative — set after baseline run) |
|-----------|-----------------------------------------------|
| Mean Dice (1–5) | Establish baseline; beat > 0 is first milestone |
| EDH Dice | Non-zero reliable detection on val (≥3 EDH cases) |
| Volume MAE (total) | < 20% of mean GT volume on val (calibrate post-baseline) |
| Inference | Full scan < 60 s CPU / < 15 s GPU (resolution-dependent) |

### Non-objectives (Version 1)

- Beating nnU-Net SOTA on BHSD  
- Semi-supervised use of unlabel_1980  
- Severity classification (downstream rule engine)  
- Real-time streaming inference  

### Success definition for Version 1 model milestone

- [ ] Training completes without divergence on CPU or Kaggle  
- [ ] Val mean Dice (1–5) logged and reproducible from checkpoint  
- [ ] Test set evaluated **once** with full metric table  
- [ ] Native-space volume error computed on test  
- [ ] Confidence scores generated for sample clinical report  
- [ ] End-to-end inference: NIfTI → 3D mask → mL per subtype  

---

## 10. Final recommendation table

| Decision | Version 1 choice | Status |
|----------|------------------|--------|
| **Model** | **MONAI 2D U-Net** | **Approved** |
| **Training mode** | **2.5D** (3 axial slices) | **Approved** |
| **Input channels** | **3** | **Approved** |
| **Output channels** | **6** (background + 5 subtypes) | **Approved** |
| **Output activation** | **Softmax (inference) / logits (train)** | **Approved** |
| **Loss** | **DiceCE + class weights** | **Approved** |
| **Primary metric** | **Macro Dice (classes 1–5)** | **Approved** |
| **Secondary metrics** | IoU, precision, recall, volume error (mL) | **Approved** |
| **Confidence** | Mean max-softmax on predicted hemorrhage voxels | **Approved** |
| **Checkpoint selection** | Best val macro Dice (1–5) | **Approved** |
| **Early stopping** | Patience 15–20 on primary metric | **Approved** |
| **Experiment logs** | `reports/training/<run_id>/` | **Approved** |
| **Input resolution** | 512² / 256² / brain ROI | **Under evaluation** |
| **Attention U-Net** | Deferred | Research Version |
| **UNet++** | Deferred | Research Version |
| **nnU-Net** | Deferred | Kaggle / Research benchmark |
| **SwinUNETR** | Deferred | Research Version (GPU) |

---

## Approval checklist (before model implementation)

- [x] MONAI 2D U-Net selected for Version 1  
- [x] I/O, loss, metrics, confidence, checkpoint policy defined  
- [ ] Input resolution locked (`training_pipeline_design.md`)  
- [ ] Preprocessing + resampling finalized  
- [ ] `splits.csv` frozen  
- [ ] First training run config template created  

---

## Related documents

- `docs/architecture/training_pipeline_design.md` — 2.5D pipeline, resolution TBD  
- `docs/architecture/windowing_normalization_design.md` — intensity preprocessing  
- `docs/architecture/resampling_design.md` — spatial preprocessing (pending)  
- `docs/architecture/data_split_design.md` — split + freeze policy  
- Phase 1: `src/research/calculate_volume.py` — volume error ground truth  

---

## References (baseline rationale)

- Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** MICCAI.  
- Isensee, F. et al. (2021). **nnU-Net: a self-adapting framework for deep learning-based biomedical image segmentation.** Nature Methods.  
- MONAI Consortium. **MONAI UNet and DiceCELoss documentation** — clinical segmentation defaults.  
- Hatamizadeh, A. et al. (2022). **Swin UNETR** — transformer baseline for 3D medical imaging (data/GPU requirements).  
- O, O. et al. (2018). **Attention U-Net** — attention gates for medical segmentation.  
- Zhou, Z. et al. (2018). **UNet++** — nested U-Net for dense skip pathways.

---

*Next milestone: resolution analysis + literature HU clip → implement preprocessing, dataset loader, and MONAI 2D U-Net training script.*
