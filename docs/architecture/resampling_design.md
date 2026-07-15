# Resampling Design — BHSD (Phase 2)

> **Status: Approved and Frozen**  
> V1 decision: **no resampling** (native spacing retained for volumes). Spacing alternatives remain documented for future work.

**Project:** BrainHemorrhageAI  
**Dataset:** BHSD labeled set (`label_192`) — 192 CT volumes  
**Status:** Approved and Frozen for Version 1 (native spacing; resampling deferred)  
**Depends on:** Phase 1 dataset analysis, `data_split_design.md`, `windowing_normalization_design.md`  
**Last updated:** Phase 2 milestone (freeze banner added during V2 Phase A cleanup)

---

## Executive summary

This document analyzes resampling options for BrainHemorrhageAI and defines **policies that are locked** vs **decisions still pending**. It prepares the project for preprocessing implementation once the full pipeline architecture (including windowing, normalization, and 2D/2.5D/3D training mode) is finalized.

**Locked policy:** Always preserve **original native voxel spacing**; always compute **reported hemorrhage volume** using spacing from the uploaded CT header (Phase 1 methodology).

**Pending decision:** Target spacing for model input (e.g. isotropic 1.0 mm — **leading candidate, not yet approved**).

---

## Current Decision Status

The final resampling strategy has **NOT** been locked yet.

Although **1 × 1 × 1 mm isotropic resampling** is widely used in medical imaging research and frameworks such as nnU-Net, it may not be the optimal choice for this specific project because:

- BHSD has **highly anisotropic spacing** (~0.49 mm in-plane vs ~4.92 mm through-plane on average).
- Our **local development hardware is limited** (16 GB RAM, no NVIDIA CUDA).
- We have **not yet finalized** the complete preprocessing pipeline.
- **Windowing and intensity preprocessing** decisions may influence the optimal resampling strategy.
- We also need to evaluate whether a **2D, 2.5D, or 3D** training pipeline is ultimately selected.

Therefore, **resampling remains a pending architectural decision**.

**Current policy:**

- Continue **preserving the original voxel spacing** information.
- Continue **calculating hemorrhage volume** using the **original CT spacing**.
- Do **not** permanently commit to a target spacing until preprocessing architecture has been finalized.

**Leading candidate (for analysis only):** 1.0 × 1.0 × 1.0 mm isotropic resampling for model input, with native-space volume reporting. Sections below evaluate this and alternatives; approval requires explicit sign-off after windowing/normalization and training-mode decisions.

---

## Why this milestone matters

Resampling is not a preprocessing detail — it is a **foundational contract** between the medical data, the neural network, and the clinical report.

| If we skip this design | Consequence |
|------------------------|-------------|
| Ad-hoc per-script spacing choices | Training and inference disagree; volumes become inconsistent |
| Wrong interpolation on masks | Fractional labels break loss functions and subtype counts |
| Volume computed on resampled grid | Clinically wrong mL values reported to users |
| Inconsistent target spacing | Cannot compare MONAI U-Net vs nnU-Net vs future models fairly |

### Contribution to the final system

```mermaid
flowchart LR
    A[Upload CT] --> B[Store native spacing]
    B --> C[Resample for model]
    C --> D[Segmentation model]
    D --> E[Resample prediction to native grid]
    E --> F[Volume in mL using native spacing]
    F --> G[Clinical rule engine]
    G --> H[Web report]
```

Every downstream component — **training**, **inference API**, **volume validation against Phase 1**, **severity rules**, **downloadable PDF** — depends on a consistent resampling policy **once finalized**.

### Valid approaches (leading candidate under evaluation)

Multiple approaches are defensible in medical imaging:

| Approach | Valid when |
|----------|------------|
| No resampling | 2D slice-only training with per-scan spacing metadata; hard to batch |
| Anisotropic target (e.g. 0.5×0.5×1 mm) | Reducing z-upsampling artifacts from thick slices |
| Isotropic 1 mm | Standard research practice; future 3D compatibility — **leading candidate** |
| nnU-Net median spacing | Automated per-dataset optimization; best with GPU pipeline |

**Leading candidate: isotropic 1.0 mm** — balances research quality, MONAI/nnU-Net compatibility, CPU-feasible 2.5D training after crop, and a clear inverse-mapping path for clinical volume. **Final approval pending** integration with windowing/normalization and training-mode decisions.

---

## 1. What is voxel spacing?

A medical CT volume is a **3D array of voxels** (volume elements). Each voxel represents a small 3D box in **patient space**, measured in millimeters.

### Pixel spacing (in-plane)

**Pixel spacing** (x, y) is the physical distance between adjacent columns and rows in an axial slice — typically **0.3–1.0 mm** for head CT. BHSD in-plane spacing ranges **~0.37–0.68 mm**.

Two scans with the same matrix size (512×512) can cover **different physical field-of-view** if pixel spacing differs.

### Slice thickness (z)

**Slice thickness** is the distance between consecutive axial slice centers along the head-to-foot axis. BHSD z-spacing ranges **~0.5–6.5 mm**, with dataset average **~4.92 mm**.

Thick slices mean **large gaps** between consecutive axial planes — anatomy between slices is not directly sampled.

### Physical voxel size

The **physical volume of one voxel** in mm³ is:

```text
voxel_volume_mm³ = spacing_x × spacing_y × spacing_z
```

Example (BHSD average): `0.49 × 0.49 × 4.92 ≈ 1.18 mm³` per voxel.

This is why Phase 1 `calculate_volume.py` required spacing from the CT header — voxel count alone is not volume.

### Why medical images differ from normal images

| Normal image | Medical CT |
|--------------|------------|
| 2D RGB, unitless 0–255 | 3D scalar **Hounsfield Units (HU)** |
| Pixel = picture element | Voxel = 3D volume element |
| No physical size metadata | **Spacing + origin + orientation** in NIfTI header |
| Resize for display is cosmetic | Resampling **changes physical sampling** of anatomy |
| Same PNG size = same content scale | Same 512×512 can be different mm coverage |

Neural networks treat input tensors as **uniform grids**. Medical resampling converts **physically heterogeneous** scans into a **consistent millimeter grid** — or maps predictions back to native space for clinical output.

---

## 2. Why resampling is required

### Different scanner settings

BHSD aggregates scans from multiple sources. Acquisition parameters (kernel, kVp, reconstruction) vary. Voxel spacing is the **measurable geometric inconsistency** we must normalize.

### Different slice thickness

Slice thickness directly affects z-axis sampling. A hemorrhage visible on one slice may be **partially volume-averaged** on another. Training on mixed z-spacings without harmonization forces the network to learn **spacing artifacts** alongside anatomy.

### Different voxel spacing

Phase 1 confirmed **non-uniform spacing** across 192 scans. Without resampling:

- Cannot batch tensors in PyTorch
- Patch sizes cover **different physical regions** in mm
- Augmentations (rotation, elastic) distort anatomy unequally
- Cross-scan Dice comparison is confounded by geometry

### Why neural networks expect consistent input spacing

Convolutional filters have **fixed receptive fields in voxel space**. A 3×3 kernel covers:

- **1.47 mm** at 0.49 mm spacing
- **3.0 mm** at 1.0 mm spacing

If spacing varies per scan, the same architecture learns **different physical scales** per batch element — unstable optimization and poor generalization.

**Resampling establishes one millimeter grid** so one voxel ≈ one mm everywhere (under our target).

---

## 3. Why BHSD specifically requires resampling

Phase 1 findings (`analyze_dataset.py`, `inspect_dataset.py`):

| Property | Finding |
|----------|---------|
| Scans | **192** labeled volumes |
| In-plane size | Fixed **512 × 512** |
| Depth | **19 unique** values, range **24–60** slices |
| In-plane spacing (x, y) | **~0.37–0.68 mm** |
| Slice spacing (z) | **~0.5–6.5 mm** |
| Average spacing | **~0.49 × 0.49 × 4.92 mm** |
| Image–mask alignment | **100%** shape match (native space) |
| Anisotropy | z spacing **~10×** in-plane spacing |

### Why this is highly anisotropic

An **isotropic** voxel has equal spacing on all axes (e.g. 1×1×1 mm). BHSD average **0.49 × 0.49 × 4.92 mm** means:

- In-plane: sub-millimeter resolution (~0.5 mm)
- Through-plane: **~5 mm** between slice centers

A “cube” of voxels is physically a **thin slab** — 10× wider in-plane than through-plane. Consequences:

1. **2.5D context** — adjacent slices are **5 mm apart**; they are not fine contiguous samples
2. **3D convolutions** — without z-resampling, kernels span large physical z but sparse sampling
3. **Thin structures** — SAH in sulci may span **one or two slices** only at native z
4. **Volume integration** — z voxel size dominates voxel volume (~1.18 mm³ avg vs ~0.12 mm³ if isotropic 0.5 mm)

BHSD cannot be fed to a consistent training pipeline in native form. **Resampling is mandatory** for this project.

---

## 4. Comparison of resampling approaches

### A. No resampling

| | |
|---|---|
| **Description** | Train and infer on native NIfTI grids; handle spacing in loss or per-sample processing |
| **Pros** | No interpolation artifacts; native slice appearance preserved; volumes trivially correct in native space |
| **Cons** | **Cannot batch** different depths (24–60); spacing varies scan-to-scan; 2.5D/3D patches cover inconsistent mm regions; deployment complexity |
| **Memory** | Lower per-volume if not upsampling z; but batch size effectively **1** |
| **Accuracy** | Theoretical best for native geometry; **practical training accuracy lower** due to optimization instability |
| **Clinical implications** | Volumes correct if counting on native grid; **segmentation model likely underperforms** |

**Verdict:** Not viable for Version 1 pipeline.

---

### B. Anisotropic resampling

| | |
|---|---|
| **Description** | Harmonize to target such as **0.5 × 0.5 × 1.0 mm** or **median native spacing (~0.49 × 0.49 × 1.0 mm)** |
| **Pros** | Less aggressive z-upsampling than full isotropic 1 mm; preserves some through-plane sparsity; lower memory than 1×1×1 if z stays coarse |
| **Cons** | Model still sees **anisotropic voxels**; complicates future 3D and literature comparison; two hyperparameters (xy vs z target); nnU-Net/MONAI defaults often assume near-isotropic after resample |
| **Clinical implications** | Volume must map back to native space; z-interpolation partial but still present if z target < native |

**Verdict:** Reasonable alternative; we reject it for Version 1 to maximize **cross-method compatibility** and simpler documentation.

---

### C. Isotropic resampling

| | |
|---|---|
| **Description** | Resample to **1.0 × 1.0 × 1.0 mm** (or 0.5 mm) on all axes |
| **Pros** | **One spacing parameter**; standard in brain segmentation literature; enables 3D patches later; uniform augmentations; fair cross-scan receptive fields |
| **Cons** | **Heavy z-upsampling** (~5 mm → 1 mm = ~5× more slices); synthetic inter-slice content; increased memory and storage |
| **GPU cost** | Higher volume size → more VRAM; on CPU/RAM-limited setup, mitigated by **2.5D slices + 256² crop** after resample |
| **Clinical implications** | Segmentation on resampled grid **must not** be used directly for volume; inverse map to native + original spacing required |

**Verdict:** **Leading candidate** for this project, pending full preprocessing sign-off — with native-space volume policy (Section 10).

---

## 5. Common target spacings in brain CT segmentation

| Target spacing | Typical use | Appropriate when |
|----------------|-------------|------------------|
| **Native (variable)** | Research exploring minimal preprocessing | Single-scan inference; no batch training |
| **~0.5 × 0.5 × 5 mm (preserve native z)** | 2D slice-wise training | Thick slices accepted; z-context via 2.5D only |
| **0.5 × 0.5 × 1.0 mm** | Anisotropic harmonization | Reduce z upsampling vs 1 mm iso; 3D with thick z still awkward |
| **1.0 × 1.0 × 1.0 mm** | **nnU-Net, MONAI, BraTS-style** pipelines | Multi-site datasets; 3D CNNs; standardized benchmarks |
| **0.5 × 0.5 × 0.5 mm** | High-resolution research | GPU-rich; sub-mm accuracy for small lesions |
| **Median dataset spacing** | nnU-Net auto-configuration | Large heterogeneous cohorts; automated pipelines |

### BHSD fit

- **192 scans**, variable spacing, **anisotropic**, CPU training → need **standardized grid** without extreme 0.5 mm isotropic cost
- **1.0 × 1.0 × 1.0 mm** is a strong match for **research quality + future nnU-Net comparison** — **pending final approval**

Approximate post-resample size (average scan):  
`512 × 0.49/1.0 ≈ 251` in-plane → **~251 × 251 × 148** voxels (from 32 slices × 4.92/1.0) — acceptable with slice/crop strategy.

---

## 6. Interpolation methods

### CT images (continuous HU field)

| Method | Behavior | Use for CT? |
|--------|----------|-------------|
| **Nearest neighbor** | No blending; blocky | **No** — stair-steps HU; destroys tissue contrast |
| **Linear (trilinear in 3D)** | Weighted average of neighbors | **Yes — default for CT** |
| **B-spline (order 3+)** | Smooth polynomial interpolation | Optional; sharper but may overshoot HU; slower; marginal gain for segmentation input |

**Decision:** **Trilinear (linear)** for CT resampling — industry standard, MONAI/SimpleITK default, good speed/quality balance.

### Segmentation masks (discrete labels)

| Method | Behavior | Use for masks? |
|--------|----------|----------------|
| **Nearest neighbor** | Preserves exact label integers | **Yes — mandatory** |
| **Linear / B-spline** | Creates **fractional values** (e.g. 1.37, 2.68) | **Never** |

### Why masks must not use linear interpolation

Linear interpolation blends adjacent labels:

```text
EDH (1) next to SDH (2) → interpolated voxel = 1.5
```

Effects:

- **Invalid classes** for softmax / cross-entropy
- **False mixed-label voxels** that do not exist in ground truth
- **Underestimated small regions** (partial-volume between classes)
- **Broken volume counts** (fractional voxels ambiguous)

**Rule:** CT = linear; mask = nearest neighbor. **Always resample CT and mask in separate passes.**

---

## 7. Computational trade-offs

| Factor | Native spacing | Isotropic 1 mm (our choice) |
|--------|----------------|----------------------------|
| **GPU memory** | Low per volume if no upsampling | **~5× more z voxels** → higher RAM; mitigated by 2.5D + 256² crops, not full-volume 3D on laptop |
| **Training speed** | Slow (batch=1, variable size) | Faster batching; **one-time preprocess** amortizes cost |
| **Inference speed** | Per-scan custom logic | Fixed pipeline; resample + infer + inverse map adds **~seconds** per scan — acceptable clinically |
| **Storage** | 192 raw NIfTIs only | **+192 processed pairs** in `data/processed/` (~several GB) — acceptable |
| **Caching** | On-the-fly resample every epoch | **Precompute once** → CacheDataset reads processed files |

### Kaggle limitations

- GPU sessions time-limited; preprocessed 1 mm volumes enable **faster epoch turnover**
- Storage quota favors compressed `.npz` or gzip NIfTI vs raw re-resampling each run

### Local laptop limitations (Ryzen 7535U, 16 GB RAM, no CUDA)

- Full 3D 251×251×148 on CPU is **not** the training path — **2.5D slices at 256²** after resample
- Preprocessing 192 scans offline overnight is acceptable
- Isotropic 1 mm without caching would **dominate** training time — **cache mandatory**

---

## 8. Leading candidate analysis (not yet approved)

### **Isotropic 1.0 × 1.0 × 1.0 mm resampling for model input, with native-space volume reporting**

*This section documents the leading candidate. Target spacing remains a pending decision (see Current Decision Status).*

| Criterion | Why this wins |
|-----------|---------------|
| **Accuracy** | Stable training grid; literature-aligned; enables fair model comparison |
| **Clinical correctness** | Volumes computed on **native grid** after inverse mask mapping — not on 1 mm grid |
| **Memory** | Full volumes large but **never loaded whole** during 2.5D training — slices only |
| **Training time** | One-time preprocess + cache; batchable 256² slices on CPU |
| **Deployment** | Same resample → infer → inverse map in FastAPI inference |
| **Research quality** | Comparable to nnU-Net / MICCAI baselines; defensible in thesis |
| **Hardware** | Feasible without GPU via 2.5D + crop |
| **Dataset size (192)** | Reduces spacing as confound — model learns from 192 consistent grids |
| **Future scalability** | Direct upgrade to 3D patches or nnU-Net without changing target spacing |

**Why not anisotropic 0.5×0.5×1?** Saves z voxels but adds a second spacing policy to document, complicates nnU-Net migration, and still requires native volume mapping.

**Why not 0.5 mm isotropic?** ~8× more voxels than 1 mm — prohibitive on 16 GB RAM / CPU.

---

## 9. How resampling affects downstream components

### Segmentation quality

- **Positive:** Consistent geometry improves learning; augmentations behave predictably
- **Negative:** z-upsampling may **blur thin SAH** or **split EDH** boundaries — monitor per-class Dice on val; compare native vs resampled visualization during QA

### Volume calculation

- **Wrong:** `voxel_count_on_1mm_grid × 1.0 mm³` after training-only resample — **different from clinical native volume**
- **Correct:** Resample **predicted mask** to native grid (nearest) → count voxels → multiply by **original spacing** from stored header (Phase 1 formula)

Validated ground-truth volumes in `volume_statistics.csv` use **native spacing** — predictions must use the **same convention** for evaluation.

### Clinical report generation

Reports display **mL per subtype** and total hemorrhage. Rule engine thresholds (future) must be calibrated in **native-space mL**. Resampling affects **model input only**, not reported numbers.

### Generalization

Harmonized spacing reduces scanner-specific voxel-size bias. Risk: model relies on **interpolated z content** — mitigated by inverse mapping and multi-site spacing variation already in BHSD train set.

### Inference

```text
Upload NIfTI → extract native spacing (persist) → resample CT to target spacing (TBD) → model predict →
resample label map to native (nearest) → volume with native spacing → API response
```

---

## 10. Preserving original voxel spacing (critical)

### Principle

> **The model may train and infer on resampled data. The patient report must always use the original voxel spacing from the uploaded CT.**

Original spacing is **ground truth for physical measurement**. Resampling is a **view transform** for the network.

### Training workflow

1. Load raw CT + mask pair from `data/raw/label_192/`
2. Read **native spacing** `(sx, sy, sz)` from CT NIfTI header → store in manifest (`data/metadata/preprocess_manifest.csv`)
3. Resample CT (trilinear) and mask (nearest) to **target spacing (pending approval; e.g. 1.0 × 1.0 × 1.0 mm)**
4. Save processed pair to `data/processed/` for training
5. **Evaluation volume metrics during development:** compare model predictions mapped to **native space** against `volume_statistics.csv` (native)

Training never needs to compute loss in native space if masks are correctly resampled with nearest neighbor — but **evaluation of mL** always uses native.

### Inference workflow

1. User uploads CT → read and **persist native spacing, origin, direction, affine**
2. Resample CT to **target spacing (TBD)** → run segmentation
3. Obtain predicted label volume on **resampled grid**
4. **Inverse resample** prediction to **native grid** (nearest neighbor) using same physical transform
5. Count voxels per label on **native grid**
6. Compute:

```text
volume_mm³ = voxel_count × native_sx × native_sy × native_sz
volume_mL  = volume_mm³ / 1000
```

7. Return volumes + overlay (overlay may be generated in native or display space — document in inference design)

### Why this preserves clinical correctness

- Matches radiology PACS measurements tied to **acquisition geometry**
- Consistent with Phase 1 ground-truth methodology
- Changing model target spacing in future **does not change** reported mL if inverse mapping is applied
- Regulatory/explainability narrative: *“Volumes measured in original scan geometry”*

### Implementation note (design only)

Store per-scan: `native_spacing`, `native_shape`, `affine`, resample transform parameters — required to invert without resampling loss on labels (nearest only once on inverse).

---

## 11. Risks and mitigations

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Aliasing** | Downsampling in-plane (0.49→1 mm) loses fine detail | Acceptable for hemorrhage (mm-scale); monitor IPH/SAH Dice; optional 0.5 mm target in Research Version if needed |
| **Interpolation artifacts (CT)** | Trilinear blurs HU at bone/soft-tissue edges | Brain window/clipping before resample; hemorrhage is intracranial soft tissue |
| **Topology changes (mask)** | Nearest neighbor on upsample can create scattered voxels | Small-object removal post-processing (optional, inference only); minimum connected component filter |
| **Boundary smoothing** | z-upsampling interpolates **between** thick slices — boundaries artificially smooth | Validate on visual QA subset; inverse map for volume; report limitation in thesis |
| **Tiny hemorrhage disappearance** | EDH/SAH sub-voxel after downsample or over-smooth | Stratified metrics on EDH; prefer **1 mm not 2 mm** target; nearest on mask preserves label if present in any contributing voxel (upsample) — **downsample risk lower for training target 1 mm from ~0.5 mm** (mild in-plane downsample) |
| **Inverse mapping error** | Wrong affine → volume mismatch | Unit test: round-trip native → 1 mm → native on mask; assert voxel count ≈ original (nearest) |
| **Inconsistent CT/mask transform** | Different transforms applied | Single transform from CT geometry; apply identical spatial transform to mask |

---

## 12. Decision summary table

### Locked decisions

| Decision | Status | Choice | Reason |
|----------|--------|--------|--------|
| **Volume calculation spacing** | **Locked** | **Original native spacing from CT header** | Clinical correctness; matches Phase 1 ground truth |
| **Prediction grid for volume** | **Locked** | **Native (after inverse nearest-neighbor resample)** | Ensures mL matches patient geometry |
| **Mask interpolation** | **Locked** | **Nearest neighbor** | Preserves discrete labels 0–5; mandatory for valid segmentation |
| **CT interpolation** | **Locked** | **Trilinear (linear)** | Preserves HU continuity; MONAI/SimpleITK default |
| **Spacing metadata storage** | **Locked** | **`preprocess_manifest.csv` + NIfTI header backup** | Reproducibility; inference inverse transform |
| **Native spacing preservation** | **Locked** | **Always store and use for volume** | Independent of model grid |

### Pending decisions

| Decision | Status | Leading candidate | Reason |
|----------|--------|-------------------|--------|
| **Target spacing** | **Pending** | **1.0 × 1.0 × 1.0 mm isotropic** | Literature standard; batchable; nnU-Net compatible — subject to hardware and pipeline review |
| **Resample timing** | **Pending** | **Once at preprocessing → `data/processed/`** | Depends on target spacing and training mode |
| **Anisotropic alternative** | **Under review** | 0.5 × 0.5 × 1.0 mm | Lower z-upsampling if 1 mm iso proves suboptimal for BHSD |

### Why the leading candidate (1 mm isotropic) is under serious consideration

1. **Single rule** for all models (MONAI U-Net today, nnU-Net tomorrow)  
2. **Clinically defensible volumes** decoupled from network grid (locked policy)  
3. **Phase 1 validated** volume pipeline remains the reporting authority  
4. **Hardware realistic** — preprocess offline; train 2.5D slices  
5. **Research credible** — aligns with nnU-Net/MICCAI conventions  
6. **Maintainable** — CT linear / mask nearest / native volume is easy to explain and audit  

**Not approved until:** windowing/normalization design finalized; 2D/2.5D/3D training mode selected; preprocessing architecture sign-off.

---

## Approval checklist (before resampling implementation)

- [ ] Full preprocessing architecture finalized (windowing, normalization, training mode)  
- [ ] Target spacing explicitly approved (1 mm iso or alternative documented)  
- [ ] CT trilinear + mask nearest confirmed  
- [ ] Native spacing preserved in manifest for all 192 scans  
- [ ] Volume evaluation protocol uses **native-space** predictions  
- [ ] Inverse resample round-trip test defined in test plan  
- [ ] Processed cache directory structure documented in preprocessing implementation spec  

---

## Related documents

- `docs/architecture/data_split_design.md` — train/val/test split (independent of spacing)
- `docs/architecture/windowing_normalization_design.md` — intensity preprocessing (must align before resampling is locked)
- Phase 1: `src/research/analyze_dataset.py` — spacing statistics source
- Phase 1: `src/research/calculate_volume.py` — native-space volume reference implementation

---

*Next milestone: finalize windowing/normalization; then approve target spacing and implement preprocessing.*
