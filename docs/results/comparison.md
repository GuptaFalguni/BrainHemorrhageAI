# Model Comparison — MONAI vs nnU-Net (locked test)

**Sources only:**  
- `reports/evaluation/experiment_best30h_eval/`  
- `reports/evaluation/nnunet_dataset501_fold0_eval/`  
- Training logs / checkpoint filesystem metadata  
- MONAI local API demos (`reports/api/*/prediction_summary.json`) for MONAI inference latency  

**Protocol:** Same 29 locked-test case IDs; shared metric modules (`metrics`, `volume_metrics`, `confidence`).  
**No fabricated numbers. N/A = not measured.**

---

## Validation checklist

| Check | Status |
|-------|--------|
| Identical locked test case IDs (n=29) | **Yes** (stems match; `.nii` vs `.nii.gz` label only) |
| Identical metric definitions | **Yes** (shared modules) |
| Native-space volume evaluation | **Yes** (BHSD GT NIfTI spacing) |
| No retraining | **Yes** |
| No preprocessing recipe change | **Yes** (nnU-Net uses its own plans at predict; volumes scored on native GT grid) |
| No architecture change | **Yes** |

---

## Side-by-side summary

| Field | MONAI best30h | nnU-Net fold0 |
|-------|---------------|---------------|
| Model | MONAI 2D U-Net (2.5D) | nnU-Net `3d_fullres` |
| Parameters | 650,028 | 44,581,860 |
| Checkpoint size | 7.5 MB | 340.4 MB |
| Train GPU | Kaggle T4 ×2 | Kaggle T4 ×2 |
| Train wall clock | ~20 min/epoch; stop @ epoch 34 (~11 h order; see training metrics doc) | ≈11 h 48 min (log timestamps to epoch 196 start) |
| Val headline | Macro Dice 0.351169 @ ep 24 | EMA pseudo Dice 0.4791 after ep 184 |
| **Test macro Dice** | **0.257375** | **0.454723** |
| Test micro Dice | 0.385548 | 0.551281 |
| Test mean IoU (cls 1–5) | 0.163153 | 0.313058 |
| Volume error mean / median (mL) | 19.533 / 4.909 | 14.318 / 3.733 |
| Mean study confidence | 0.8107 | 0.9936 |
| Inference (CPU) | ~19–22 s (`best30h` API demos) | mean **327.64 s** (n=28 timed) |

### Per-class Dice (locked test)

| Class | MONAI | nnU-Net |
|-------|------:|--------:|
| EDH | 0.010969 | 0.206574 |
| SDH | 0.590550 | 0.724985 |
| SAH | 0.306365 | 0.577518 |
| IPH | 0.228673 | 0.437677 |
| IVH | 0.150319 | 0.326859 |

### Per-class IoU (locked test)

| Class | MONAI | nnU-Net |
|-------|------:|--------:|
| EDH | 0.005514 | 0.115184 |
| SDH | 0.418993 | 0.568609 |
| SAH | 0.180892 | 0.405994 |
| IPH | 0.129097 | 0.280145 |
| IVH | 0.081268 | 0.195357 |

See `publication_comparison_tables.md` for Precision / Recall / Sensitivity / Specificity and qualitative rows.

---

## Strengths / weaknesses / clinical suitability

| | MONAI | nnU-Net |
|-|-------|---------|
| Strengths | Fast CPU demo path; small artifact; integrated API/UI | Higher locked-test Dice/IoU; better volume errors; all subtypes improved |
| Weaknesses | EDH collapsed; lower overall Dice; larger volume errors | Slow CPU inference; huge checkpoint; not in API; high raw confidence |
| Clinical suitability (research demo) | Usable for fast demo; **not** adequate subtype quality | **Better segmentation evidence** on locked test; still not clinical-grade (EDH weak, uncalibrated confidence, no prospective validation) |

---

## Publication stance

On the project’s locked-test protocol, **nnU-Net fold 0 outperforms MONAI best30h** on macro Dice, micro Dice, mean IoU, and mean/median volume error, with higher Dice on every hemorrhage class.

Serving and latency still favor MONAI until dual serving exists.
