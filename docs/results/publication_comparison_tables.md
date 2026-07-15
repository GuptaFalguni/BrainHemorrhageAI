# Publication Comparison Tables — MONAI vs nnU-Net

Locked test, n=29. Values from evaluation reports only.

**MONAI source:** `reports/evaluation/experiment_best30h_eval/`  
**nnU-Net source:** `reports/evaluation/nnunet_dataset501_fold0_eval/`

---

## 1. Global segmentation metrics

| Metric | MONAI | nnU-Net |
|--------|------:|--------:|
| Macro Dice (1–5) | 0.257375 | **0.454723** |
| Micro Dice | 0.385548 | **0.551281** |
| Mean IoU (1–5) | 0.163153 | **0.313058** |

---

## 2. Per-class Dice

| Class | MONAI | nnU-Net |
|-------|------:|--------:|
| EDH | 0.010969 | 0.206574 |
| SDH | 0.590550 | 0.724985 |
| SAH | 0.306365 | 0.577518 |
| IPH | 0.228673 | 0.437677 |
| IVH | 0.150319 | 0.326859 |

---

## 3. Per-class IoU

| Class | MONAI | nnU-Net |
|-------|------:|--------:|
| EDH | 0.005514 | 0.115184 |
| SDH | 0.418993 | 0.568609 |
| SAH | 0.180892 | 0.405994 |
| IPH | 0.129097 | 0.280145 |
| IVH | 0.081268 | 0.195357 |

---

## 4. Per-class Precision

| Class | MONAI | nnU-Net |
|-------|------:|--------:|
| EDH | 0.021363 | 0.317284 |
| SDH | 0.624776 | 0.736305 |
| SAH | 0.365544 | 0.764798 |
| IPH | 0.312952 | 0.478719 |
| IVH | 0.388272 | 0.597269 |

---

## 5. Per-class Recall (= Sensitivity)

| Class | MONAI | nnU-Net |
|-------|------:|--------:|
| EDH | 0.007378 | 0.153140 |
| SDH | 0.559879 | 0.714008 |
| SAH | 0.263678 | 0.463917 |
| IPH | 0.180156 | 0.403118 |
| IVH | 0.093201 | 0.224995 |

---

## 6. Per-class Specificity

| Class | MONAI | nnU-Net |
|-------|------:|--------:|
| EDH | 0.999972 | 0.999973 |
| SDH | 0.999447 | 0.999579 |
| SAH | 0.999806 | 0.999940 |
| IPH | 0.999748 | 0.999721 |
| IVH | 0.999807 | 0.999801 |

---

## 7. Volume error (native spacing, mL)

| Metric | MONAI | nnU-Net |
|--------|------:|--------:|
| Mean abs total error | 19.533 | **14.318** |
| Median abs total error | 4.909 | **3.733** |
| Max abs total error | 183.507 | 162.432 |

---

## 8. Confidence (study-level)

| Metric | MONAI | nnU-Net |
|--------|------:|--------:|
| Mean | 0.8107 | 0.9936 |
| Median | 0.8105 | 0.9946 |
| Min | 0.7046 | 0.9765 |

Both uncalibrated softmax confidence.

---

## 9. Inference speed

| Metric | MONAI | nnU-Net |
|--------|-------|---------|
| Device (reported) | CPU (API demos) | CPU (locked-test eval) |
| Latency | ~19.3–21.8 s / case (`best30h` API demos) | mean **327.64 s** / case (n=28 timed) |

GPU inference latency: **N/A** (not measured in this Phase B eval).

---

## 10. Training time

| Metric | MONAI | nnU-Net |
|--------|-------|---------|
| Schedule | Early stop epoch 34 (max 45); best @ 24 | Time-capped ~196 epochs |
| Wall clock | ~20 min/epoch documented → **~11 h order** (`experiment_best30h_kaggle_v2_metrics.md`) | **≈11 h 48 min** (train log: 16:31:10 → 04:19:18) |

---

## 11. GPU

| Metric | MONAI | nnU-Net |
|--------|-------|---------|
| Training | Kaggle T4 ×2 | Kaggle T4 ×2 |
| This locked-test eval | CUDA on Kaggle (historical) | Local CPU |

---

## 12. Checkpoint size

| Metric | MONAI | nnU-Net |
|--------|------:|--------:|
| File size | 7.5 MB | 340.4 MB |

---

## 13. Parameters

| Metric | MONAI | nnU-Net |
|--------|------:|--------:|
| Count | 650,028 | 44,581,860 |

---

## 14. Qualitative summary (evidence-based)

| Dimension | MONAI | nnU-Net |
|-----------|-------|---------|
| Strengths | Small, fast CPU path; demo-ready | Higher locked-test Dice/IoU; better volumes; all subtypes ↑ |
| Weaknesses | Macro Dice 0.257; EDH ~0.01 | Slow CPU; large model; not served; EDH still 0.21 |
| Clinical suitability | Research demo only; subtype quality insufficient | Better research segmentation candidate; **not** clinically validated |

---

## Regenerating tables

```powershell
python -m evaluation.compare_models
```
