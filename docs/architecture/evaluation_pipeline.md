# Evaluation Pipeline — BHSD (Phase 3)

> **Status: Approved and Frozen**  
> Metrics, checkpoint selection, volume, and confidence policy locked for V1 experiments. Kept for thesis reviewers.

**Project:** BrainHemorrhageAI  
**Dataset:** BHSD — 192 labeled CT volumes, 6-class segmentation  
**Status:** Approved and Frozen (evaluation infrastructure implemented)  
**Scope:** Metrics, logging, checkpointing, confidence, and clinical volume evaluation

---

## 1. Why evaluation is separated from training

Training and evaluation answer different questions. Training optimizes parameters; evaluation measures whether those parameters generalize under fixed rules. When both live in the same script, metrics drift easily: accidental test-set peeking, inconsistent preprocessing between epochs, or checkpoint selection driven by convenience rather than policy.

Separating evaluation into reusable modules gives four concrete benefits:

1. **Reproducibility** — the same metric functions run in validation loops, ablation studies, and final test reporting. Numbers computed in week one match numbers computed after retraining.
2. **Auditability** — checkpoint selection, split usage, and volume formulas are documented and enforced in one place, not scattered across notebooks.
3. **Testability** — synthetic tensors can verify Dice, IoU, and volume math without loading NIfTI files or instantiating a network.
4. **Publication quality** — thesis and portfolio work require explicit protocols (validation vs test, native-space volumes, confidence definitions). A dedicated pipeline makes those protocols executable, not aspirational.

Training scripts should *call* evaluation utilities; they should not reimplement them.

---

## 2. How metrics are computed

### Segmentation metrics (2D slices or stacked volumes)

All segmentation metrics operate on **hard class assignments** (argmax of logits at inference) unless noted otherwise. Predictions and ground truth are integer label maps with values in `{0, 1, 2, 3, 4, 5}`.

For each class `k`, a one-versus-rest binary mask is formed:

- `pred_k = (prediction == k)`
- `target_k = (ground_truth == k)`

From these masks, true positives, false positives, false negatives, and true negatives are counted. Standard formulas apply:

| Metric | Formula (per class) | Notes |
|--------|---------------------|-------|
| **Dice** | `2·TP / (2·TP + FP + FN)` | Primary overlap measure; smooth term avoids division by zero |
| **IoU (Jaccard)** | `TP / (TP + FP + FN)` | Stricter on small regions |
| **Precision** | `TP / (TP + FP)` | False-positive control |
| **Recall / Sensitivity** | `TP / (TP + FN)` | Lesion detection |
| **Specificity** | `TN / (TN + FP)` | Background / negative control |

**Per-class Dice** reports each class separately. **Macro Dice** averages Dice over selected classes (typically 1–5, excluding background). **Micro Dice** aggregates intersections and cardinalities across classes before applying the Dice formula, weighting frequent classes more heavily.

When `ignore_background=True`, class 0 is excluded from macro summaries. Background-only slices still contribute to per-class background metrics if explicitly requested, but model selection uses hemorrhage classes only.

### Clinical volume metrics (3D, native space)

Volume is **not** derived from resampled training grids. After slice predictions are stacked (and optionally inverse-mapped to native geometry), voxel counts are multiplied by **original voxel volume**:

```text
VoxelVolume_mm³ = spacing_x × spacing_y × spacing_z
Volume_mL = (voxel_count × VoxelVolume_mm³) / 1000
```

Per-subtype volumes sum voxels where the label equals that subtype. Absolute error is `|V_pred − V_gt|` in mL; relative error is `|V_pred − V_gt| / max(V_gt, ε)`.

This matches Phase 1 volume analysis and preserves clinical interpretability.

### Confidence (Version 1)

Confidence uses **deterministic softmax** outputs only — no Monte Carlo dropout, ensembles, or temperature scaling in Version 1.

- **Per-voxel confidence:** `max_k P(class k | voxel)`
- **Class confidence:** mean per-voxel confidence over voxels assigned to that class
- **Study confidence:** mean per-voxel confidence over all predicted hemorrhage voxels (classes 1–5)

Histograms of per-voxel confidence support calibration analysis in later research phases.

---

## 3. When metrics are computed

| Phase | Data | Metrics | Purpose |
|-------|------|---------|---------|
| **Training epoch (train split)** | Minibatches | Loss only (not logged as selection criteria) | Optimization signal |
| **Training epoch (val split)** | All val slices/scans | Dice, IoU, precision/recall, volume error, confidence | **Checkpoint selection**, early stopping, curves |
| **After training (test split)** | All test slices/scans | Same as validation | **Final unbiased report** — run once |
| **Failure analysis** | Val/test subsets | Per-scan Dice, volume error, low-confidence regions | Qualitative review, figure generation |

Validation metrics run **every epoch** (or every N epochs on CPU budget). Test metrics run **once** on `best_model.pt` loaded from validation macro Dice.

Training loss is logged for curves but never used to select checkpoints.

---

## 4. Validation vs test policy

| Split | Scans | Role |
|-------|-------|------|
| **Train (134)** | Learning weights | Never used for metric reporting in papers |
| **Validation (29)** | Hyperparameter tuning, checkpoint selection, early stopping | Repeatable during development |
| **Test (29)** | Final locked evaluation | **Single official run** after training completes |

Rules:

1. **Never** select checkpoints using test metrics.
2. **Never** tune preprocessing, thresholds, or architecture based on test results and re-test on the same split without disclosure.
3. Validation can be run frequently; test is treated as **locked** (see `data_split_design.md` freeze policy).
4. All splits use the same metric functions — only the *policy* differs.

---

## 5. Checkpoint selection policy

Checkpoints are chosen **only** by **validation macro Dice** over hemorrhage classes **1–5** (background excluded).

| Artifact | Saved when | Contents |
|----------|------------|----------|
| `best_model.pt` | Val macro Dice improves | Weights, optimizer, epoch, val metrics snapshot, metadata hashes |
| `last_model.pt` | Every epoch (or end of training) | Same structure; no selection implication |

Tie-breakers (documented in `model_design.md`, applied in order if macro Dice ties within floating-point tolerance):

1. Higher EDH (class 1) Dice on validation  
2. Lower total hemorrhage volume MAE (mL) on validation  
3. Lower false-positive rate on background-only slices  

Test metrics are stored in the final experiment report but **do not** influence `best_model.pt`.

---

## 6. Experiment logging

Each training run creates an isolated directory:

```text
reports/training/<run_id>/
├── config.yaml              # Training + preprocessing config snapshot
├── environment.json         # Python, torch, MONAI, seed, git hash
├── metrics.json             # Summary metrics (best val, final test when run)
├── epoch_metrics.jsonl      # One JSON object per epoch
├── plots/
│   ├── loss_curve.png
│   ├── dice_curve.png
│   ├── per_class_dice.png
│   ├── volume_error.png
│   └── confidence_histogram.png
├── confusion_matrix.png
└── split_manifest_hash.txt  # SHA-256 of splits.csv
```

Logging is append-only per epoch. Config and environment are written at run start so partial runs remain interpretable.

The **split hash** ties every experiment to the exact train/val/test manifest. Changing `splits.csv` produces a different hash and warns when comparing runs.

---

## 7. Confidence estimation (Version 1)

Softmax confidence is a **model output summary**, not a clinical probability. It answers: “How peaked was the predicted class distribution?”

Use cases in BrainHemorrhageAI:

- **UI flag:** study-level confidence below a validation-derived threshold triggers “manual review recommended.”
- **Failure mining:** export slices with low mean confidence for error analysis.
- **Thesis reporting:** distribution plots show whether the model is overconfident on false positives.

Version 1 does **not** implement uncertainty quantification (deep ensembles, evidential networks, etc.). Those belong in Research Version with calibration metrics (ECE, temperature scaling).

---

## 8. Clinical volume evaluation

Segmentation quality (Dice) and clinical utility (volume in mL) can diverge: a model can achieve high slice Dice while under-segmenting along the z-axis, producing acceptable overlap but wrong blood volume estimates.

Volume evaluation therefore runs as a **parallel track**:

1. Reconstruct 3D prediction on the preprocessed grid (stack center-slice outputs).
2. Inverse-map to native spacing (when resampling is approved in preprocessing).
3. Count voxels per subtype using **original spacing** from NIfTI headers.
4. Compare to Phase 1 ground-truth volumes (`volume_statistics.csv`).

Report **median absolute error (mL)** per subtype and for total hemorrhage. Large errors on EDH cases are flagged even when macro Dice appears acceptable.

---

## 9. Failure case visualization

Quantitative metrics hide systematic errors. The evaluation pipeline supports qualitative review:

| Failure mode | What to inspect | Suggested output |
|--------------|-----------------|------------------|
| False positive at skull | High precision drop, high confidence on non-brain | Overlay PNG: CT + FP mask |
| Missed EDH | Class 1 recall drop | Axial slice with GT vs pred |
| Volume underestimate | Low Dice but acceptable; large negative volume error | Side-by-side volume table |
| Low confidence bleed | Confidence histogram tail | Highlight low-confidence voxels on overlay |

Plots are saved as **PNG** (non-interactive) under `reports/training/<run_id>/plots/` or a dedicated `reports/failures/` folder during manual review. The training loop is not required to generate these automatically in Version 1, but metric and confidence utilities must expose the numbers needed to select cases.

---

## 10. Module map

| Module | Responsibility |
|--------|----------------|
| `src/evaluation/metrics.py` | Tensor segmentation metrics |
| `src/evaluation/volume_metrics.py` | Native-space mL volumes and errors |
| `src/evaluation/confidence.py` | Softmax confidence summaries |
| `src/evaluation/checkpoint.py` | Best/last checkpoint I/O and resume |
| `src/evaluation/experiment_logger.py` | Run directories, JSON logs, environment |
| `src/evaluation/plots.py` | PNG training and analysis figures |
| `src/evaluation/validate_evaluation.py` | Synthetic self-test of all components |

Training code (future) imports these modules; it does not duplicate their logic.

---

## 11. Design principles

1. **Single source of truth** — one Dice implementation for validation, test, and unit tests.  
2. **Native spacing for volumes** — resampled grids are for training convenience; mL uses original geometry.  
3. **Test set is sacred** — checkpoint logic has no access to test labels during training.  
4. **Deterministic confidence first** — extend to uncertainty methods later without breaking the API.  
5. **Small, typed functions** — each metric is independently testable on synthetic tensors.

This document governs evaluation behavior for all Version 1 experiments and any future model upgrades (nnU-Net, 3D, semi-supervised) that reuse the BHSD split manifest.
