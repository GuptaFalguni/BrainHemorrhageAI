# nnU-Net Results — Dataset501 fold 0 (locked test)

**Status:** Locked-test evaluation complete (research Phase B)  
**Sources:** `reports/evaluation/nnunet_dataset501_fold0_eval/`  
**No fabricated numbers.**

---

## Model

| Field | Value |
|-------|-------|
| Framework | nnU-Net v2 |
| Configuration | `3d_fullres` |
| Fold | 0 |
| Architecture | PlainConvUNet (plans) |
| Parameters | **44,581,860** |
| Test-time mirroring | **False** (parity with MONAI: no TTA) |

## Preprocessing

| Field | Value |
|-------|-------|
| At train / predict | nnU-Net `DefaultPreprocessor` + `CTNormalization` per plans |
| Locked split | Same 29 test case IDs as MONAI (`splits.csv` / `imagesTs`) |
| Volume metrics grid | Native BHSD GT spacing (not plans spacing) |

## Checkpoint

| Field | Value |
|-------|-------|
| Path | `checkpoints/nnunet_dataset501_fold0/checkpoint_best.pth` |
| Size | **340.4 MB** |
| Best val EMA pseudo Dice (train log) | **0.4791** (after epoch 184) |
| Epochs completed (train) | **196** (time-capped) |

## Hardware

| Field | Value |
|-------|-------|
| Training | Kaggle GPU T4 ×2 |
| Locked-test inference (this eval) | **CPU** (local; no NVIDIA GPU) |

## Epochs

| Field | Value |
|-------|-------|
| Train wall clock (log) | 2026-07-12 16:31:10 → 2026-07-13 04:19:18 (**≈11 h 48 min**) to start of epoch 196 |
| Eval cases | **29 / 29** locked test |

## Dice (locked test, n=29)

| Metric | Value |
|--------|------:|
| Macro Dice (classes 1–5) | **0.454723** |
| Micro Dice | 0.551281 |

### Per-class (test)

| Class | Dice | IoU | Precision | Recall | Sensitivity | Specificity |
|-------|-----:|----:|----------:|-------:|------------:|------------:|
| BG | 0.9988 | 0.9976 | 0.9982 | 0.9994 | 0.9994 | 0.5622 |
| EDH | 0.2066 | 0.1152 | 0.3173 | 0.1531 | 0.1531 | 1.0000 |
| SDH | 0.7250 | 0.5686 | 0.7363 | 0.7140 | 0.7140 | 0.9996 |
| SAH | 0.5775 | 0.4060 | 0.7648 | 0.4639 | 0.4639 | 0.9999 |
| IPH | 0.4377 | 0.2801 | 0.4787 | 0.4031 | 0.4031 | 0.9997 |
| IVH | 0.3269 | 0.1954 | 0.5973 | 0.2250 | 0.2250 | 0.9998 |

Mean IoU over hemorrhage classes 1–5: **0.313058**.

## Volume error (native spacing, mL)

| Metric | Value |
|--------|------:|
| Mean abs total hemorrhage volume error | **14.3178** |
| Median | **3.7331** |
| Max | **162.4316** |

## Confidence

| Metric | Value |
|--------|------:|
| Mean study confidence | **0.9936** |
| Median | 0.9946 |
| Min | 0.9765 |

Softmax confidence over predicted hemorrhage voxels; **not calibrated**.

## Inference speed

| Context | Value |
|---------|-------|
| Device | CPU |
| Mean seconds / case | **327.64 s** (n=28 timed; 1 reused smoke prediction untimed) |
| Median (timed) | 303.86 s |
| Min / max (timed) | 175.98 / 769.35 s |

## Strengths

- Locked-test macro Dice **0.455** vs MONAI **0.257**.
- Improves every hemorrhage subtype Dice vs MONAI on the same 29 cases.
- Lower mean / median volume error than MONAI (14.3 / 3.7 vs 19.5 / 4.9 mL).
- True 3D context.

## Weaknesses

- EDH still weak (Dice **0.207**).
- Very high study confidence (~0.99) without calibration — risk of overconfidence.
- Slow on CPU (~5.5 min/case).
- Large checkpoint (~340 MB) and ~45M parameters.

## Limitations

- Train was time-capped (~196 epochs), not full default nnU-Net schedule.
- Inference evaluated **without** test-time mirroring.
- Served as research model `nnunet_fold0` via API v1 (not the interactive default). See [`docs/platform/api_v1_contract.md`](../platform/api_v1_contract.md).
- `splits.csv` file hash differs from the MONAI Kaggle eval hash (manifest bytes changed); **case IDs of the 29 test scans match** MONAI eval stems (`.nii` vs `.nii.gz` naming only).

## Artifacts

All present under `reports/evaluation/nnunet_dataset501_fold0_eval/`:

- `metrics.json`, `per_class_metrics.csv`, `volume_metrics.csv`, `confidence_metrics.csv`, `prediction_summary.csv`, `evaluation_report.md`, `failure_analysis.json`
- `prediction_masks/` (29), `overlays/` (29), `predictions/` (npz), `plots/`
