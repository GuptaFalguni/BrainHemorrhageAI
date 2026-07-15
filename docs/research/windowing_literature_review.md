# Windowing & Normalization — Literature Review

**Project:** BrainHemorrhageAI  
**Dataset:** BHSD labeled set (`label_192`) — derived from RSNA ICH cohort  
**Purpose:** Select **Version 1** intensity preprocessing (fixed for reproducibility)  
**Status:** Complete — configuration updated in `config/preprocessing.yaml`  
**Date:** Phase 3

---

## 1. Why this review is necessary

Phase 2 architecture approved the **pipeline order** (clip/window → z-score → resample TBD) but deliberately **left HU values unset** until evidence could be collected. Using arbitrary windows (e.g. `[0, 100]` from informal examples) would:

- Break **comparability** with published BHSD benchmarks  
- Risk **missing subtle SDH/SAH** (wrong window) or **including skull-dominated signal** (too-wide window without clip)  
- Invalidate **train/inference consistency** if values are changed mid-project  

Version 1 requires **one fixed, cited configuration** before training, normalization statistics validation, and resampling experiments proceed.

---

## 2. Papers selected (and why)

| Paper | Why included |
|-------|----------------|
| **Wu et al., BHSD (2023)** [1] | **Primary source** — our exact dataset; reports full preprocessing for 192 labeled volumes |
| **Isensee et al., nnU-Net (2018/2021)** [2] | BHSD authors follow nnU-Net; de facto CT segmentation standard |
| **Chilamkurthy et al. (2018)** [3] | Foundational **RSNA ICH** paper; multi-window preprocessing on same underlying cohort class |
| **Soun et al., Eur Radiol Exp (2023)** [4] | Quantifies **benefit of windowing** for RSNA-style ICH detection |
| **Tang et al., Brain Sci. (2023)** [5] | RSNA-derived data; tabulates **clinical window presets** |
| **Lee et al. / Ge & Broder (radiology)** [6] | Subdural-specific windowing for **thin SDH** on CT |
| **MONAI community (ScaleIntensityRanged)** [7] | Project stack; hemorrhage segmentation practice with HU clip + scale |
| **Innolitics engineering guide (2024)** [8] | Consolidates common **HU clip ranges** by anatomy |

Papers using **multi-class BHSD segmentation** or **RSNA + segmentation** were prioritized over generic natural-image normalization.

---

## 3. Comparison table

| Paper | Dataset / task | Window (WL / WW) or HU clip | Normalization | Why they used it | Pros | Cons |
|-------|------------------|----------------------------|---------------|------------------|------|------|
| **Wu et al., BHSD [1]** | BHSD 192 vol — **multi-class 3D segmentation** | **Clip HU `[-40, 120]`** (truncate); cites nnU-Net | **`(HU − 40) / 80`** after clip | Follow nnU-Net; cover brain parenchyma + acute blood + partial CSF | **Direct benchmark for our data**; single channel; reproducible | Wider than strict brain window; not multi-window |
| **Isensee et al., nnU-Net [2]** | Multi-dataset CT segmentation | **Clip to 0.5–99.5 percentiles** of foreground train intensities | **Global z-score** (train μ, σ) after percentile clip | CT HU is absolute; robust to outliers | Auto-adapts per dataset; SOTA framework | Percentiles differ from fixed clinical window; needs full train pass |
| **Chilamkurthy et al. [3]** | RSNA ICH — **slice classification** | **Brain 40/80**, **Subdural 80/200**, **Bone 600/2800** → 3-channel RGB | 8-bit grayscale per window | Mimic radiologist multi-window review | Strong for **classification**; SDH sensitivity | 3× channels for windows, not 3D context; not segmentation masks |
| **Soun et al. [4]** | RSNA ICH — detection | **Brain 40/80**, **Soft tissue 40/380**, **Subdural 80/200** | JPEG 8-bit / DL standard | Measure windowing impact on detection | Evidence that windowing **helps** ICH DL | Classification-focused; different task |
| **Tang et al. [5]** | RSNA — detection | Table: Brain **40/80**, Subdural **75/215**, Bone **600/2800**, etc. | Linear HU from DICOM rescale + optional window | Align with radiologist practice | Comprehensive window catalog | Detection; no voxel labels |
| **Ge & Broder [6]** | Radiology review | **Subdural window ~30/130**; brain **20/75** | Display / interpretation | Thin SDH vs bone | Clinical SDH visibility | Display-only; not DL pipeline |
| **MONAI / Cerebra example [7]** | Hemorrhage 3D seg (general) | **`ScaleIntensityRanged` `a_min=0`, `a_max=100` → [0,1]** | Min-max after clip | Simple MONAI transform | Easy deployment | **Not BHSD-specific**; no citation on same dataset |
| **Innolitics [8]** | Engineering guide | Brain **0–80 HU** recommended clip | Clip + optional [0,1] scale | Best-practice summary | Practical defaults | General guide, not hemorrhage benchmark |

---

## 4. Synthesis by topic

### Brain CT windowing (radiology)

Standard **brain window** = **WL 40 HU, WW 80 HU** → displayed range **0–80 HU** [3, 4, 5]. Used to accentuate parenchyma and acute hyperdense hemorrhage.

**Subdural windows** (e.g. WL 80, WW 200) improve thin SDH conspicuity [3, 4, 6] but increase skull/soft-tissue signal — common in **classification** pipelines, not BHSD segmentation baseline.

### Hemorrhage segmentation (BHSD)

Wu et al. explicitly preprocess labeled volumes:

> *"Following nnUnet [13], we first **truncated the HU values** of each scan using the range of **[-40,120]**, and then **normalized** the truncated voxel values by **subtracting 40 and dividing by 80**."* [1, §3.0.2]

This is **wider** than brain window `[0, 80]` — includes lower HU (CSF/soft tissue tail at −40) and upper acute blood (to 120).

### nnU-Net recommendations

Default **CT channel**: collect foreground train intensities → clip **0.5–99.5 percentiles** → **dataset z-score** [2]. BHSD authors **adapt** nnU-Net spirit but use **fixed** `[-40,120]` clip and **fixed** `(40, 80)` linear normalize instead of empirical percentiles [1].

### RSNA ICH papers

RSNA cohort emphasizes **multi-window RGB** for **detection** [3, 4]. Our Version 1 training uses **2.5D slice channels** (spatial context), not multi-window channels — consistent with **BHSD segmentation baseline** (single HU channel) [1].

### MONAI examples

Community hemorrhage tutorials often clip **`[0, 100]` HU** and scale to `[0, 1]` [7]. Useful pattern but **superseded for Version 1** by BHSD paper on the same task and data.

### Single window vs multi-window

| Approach | Evidence | Version 1 fit |
|----------|----------|---------------|
| **Single HU clip + normalize** | BHSD benchmarks [1]; nnU-Net CT [2] | **Selected** — matches 2.5D single-modality input |
| **Multi-window (3-channel)** | RSNA classification [3, 4] | Defer — conflicts with 2.5D using adjacent **slices** as channels |

---

## 5. Version 1 recommendation (evidence-only)

### Approved pipeline

```text
Load native HU
→ Clip to [-40, 120] HU
→ Z-score: (HU − 40) / 80
→ [Resampling — pending]
→ Model
```

Implemented via existing `preprocess_volume.py`:

- `clip_min: -40`, `clip_max: 120`
- `normalization_method: zscore`
- `train_mean: 40`, `train_std: 80` (literature-fixed per BHSD [1], equivalent to their `(x−40)/80`)

### Why this is most suitable for BHSD

1. **Same dataset, same task** — Wu et al. [1] define the published benchmark preprocessing for the **192 labeled volumes we use**.  
2. **Multi-class segmentation** — not RSNA classification windows [3].  
3. **Single channel** — compatible with **2.5D** (3 slices × 1 HU channel) per `training_pipeline_design.md`.  
4. **nnU-Net lineage** — authors cite nnU-Net [2]; clip-then-normalize in HU space matches our approved architecture.  
5. **Reproducible** — fixed constants; no ad-hoc values.

### `window_level` / `window_width` in config

Set to **40 / 80** as **documented radiological reference** (brain window) [3, 5] — **not** the active clip when `clip_min`/`clip_max` are set. Active clip remains **`[-40, 120]`** from BHSD [1].

### Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| **BHSD `[-40,120]` vs brain `[0,80]`** | Matches benchmark; includes slightly wider HU | More skull-edge voxels than narrow brain window |
| **Fixed μ=40, σ=80 vs empirical train z-score** | Identical to BHSD paper; instant inference config | May differ slightly from `compute_dataset_statistics.py` on our 134-scan train split |
| **Single vs multi-window** | Simpler; consistent with BHSD 3D baselines | May sacrifice RSNA-detection-style SDH contrast [4] — ablation for Research Version |
| **vs nnU-Net percentile clip** | Clinical interpretability | Not auto-tuned to our train histogram |

### Optional verification

Run `compute_dataset_statistics.py` after locking clip to compare **empirical** train μ/σ vs **literature-fixed** 40/80. Do **not** change Version 1 config without a documented ablation.

---

## 6. Version 1 locked configuration

See `config/preprocessing.yaml` — updated with values and citation comments.

| Parameter | Value | Source |
|-----------|-------|--------|
| `clip_min` | **−40** | Wu et al., BHSD [1] |
| `clip_max` | **120** | Wu et al., BHSD [1] |
| `window_level` | **40** | Brain window reference [3, 5] |
| `window_width` | **80** | Brain window reference [3, 5] |
| `normalization_method` | **zscore** | Architecture + BHSD linear norm as z-score |
| `train_mean` | **40** | Wu et al., BHSD [1] |
| `train_std` | **80** | Wu et al., BHSD [1] |

---

## References

[1] Wu, B., Xie, Y., Zhang, Z., et al. **BHSD: A 3D Multi-Class Brain Hemorrhage Segmentation Dataset.** arXiv:2308.11298, 2023. Implementation details §3.0.2. https://doi.org/10.48550/arxiv.2308.11298  

[2] Isensee, F., Jaeger, P., Kohl, S., Petersen, J., Maier-Hein, K. **nnU-Net: a self-adapting framework for deep learning-based biomedical image segmentation.** *Nature Methods* 18, 203–211 (2021). arXiv:1809.10486. CT normalization: foreground 0.5–99.5 percentile clip + global z-score.  

[3] Chilamkurthy, S., Ghosh, R., Tanamala, S., et al. **Deep learning algorithms for detection of critical findings in head CT scans.** (RSNA cohort). *Annals of Emergency Medicine* / PMC8377493. Brain 40/80, subdural 80/200, bone 600/2800.  

[4] Soun, J.E., et al. **Evaluation of techniques to improve a deep learning algorithm for the automatic detection of intracranial haemorrhage on CT head imaging.** *European Radiology Experimental* 7, 26 (2023). https://doi.org/10.1186/s41747-023-00330-3  

[5] Tang, D., et al. **An Efficient Framework to Detect Intracranial Hemorrhage Using Hybrid Deep Neural Networks.** *Brain Sciences* 13(4), 400 (2023). RSNA window table.  

[6] Ge, C., Broder, J. **Imaging of Intracranial Hemorrhage.** *Emergency Radiology* review, PMC5307932. Subdural vs brain window for thin collections.  

[7] MONAI **`ScaleIntensityRanged`**; hemorrhage segmentation example (Cerebra / MONAI community). Clip HU 0–100 → [0, 1].  

[8] Innolitics. **Medical Imaging AI/ML Engineering Guide** (2024). HU clipping by tissue task; brain ~0–80 HU.  

---

*Awaiting approval before training or further preprocessing implementation changes.*
