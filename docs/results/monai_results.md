# MONAI Results — experiment_best30h

**Status:** Locked Version 1 baseline (weak Model A)  
**Sources:** `reports/evaluation/experiment_best30h_eval/`, `docs/training/experiment_best30h_kaggle_v2_metrics.md`  
**No fabricated numbers** — values below are copied from those reports.

---

## Model

| Field | Value |
|-------|-------|
| Framework | MONAI |
| Architecture | 2D U-Net (`monai_2d_unet`) |
| Input | 2.5D (3 adjacent axial slices) |
| Output classes | 6 (BG, EDH, SDH, SAH, IPH, IVH) |
| Channels / strides | `[32, 64, 128, 256]` / `[2, 2, 2]` |
| Experiment | `experiment_best30h` |

## Preprocessing

| Field | Value |
|-------|-------|
| HU clip | `[-40, 120]` |
| Normalization | z-score `(HU − 40) / 80` |
| Resampling | None (native spacing) |
| Config | `config/preprocessing.yaml` |

## Checkpoint

| Field | Value |
|-------|-------|
| Path (repo) | `checkpoints/experiment_best30h/best_model.pt` |
| Best epoch | **24** |
| Val macro Dice at best | **0.351169** |
| Val macro IoU at best | **0.236704** |
| Early stop | Epoch **34** (patience 10; max 45 configured) |

## Hardware

| Field | Value |
|-------|-------|
| Training | Kaggle GPU T4 ×2 (`falguni1234/brainhemorrhageai-gpu-train`, Save Version #2) |
| Eval environment (metrics.json) | Linux, torch `2.10.0+cu128`, MONAI `1.6.0`, Python `3.12.13` |
| Local demo inference | CPU (no local NVIDIA GPU) |

## Epochs

| Field | Value |
|-------|-------|
| Configured max | 45 |
| Completed through early stop | 34 |
| Best checkpoint epoch | 24 |

## Dice (locked test, n=29)

| Metric | Value |
|--------|------:|
| Macro Dice (classes 1–5) | **0.257375** |
| Micro Dice | 0.385548 |

### Per-class Dice / IoU (test)

| Class | Dice | IoU |
|-------|-----:|----:|
| BG (0) | 0.998477 | 0.996958 |
| EDH (1) | 0.010969 | 0.005514 |
| SDH (2) | 0.590550 | 0.418993 |
| SAH (3) | 0.306365 | 0.180892 |
| IPH (4) | 0.228673 | 0.129097 |
| IVH (5) | 0.150319 | 0.081268 |

Mean IoU over hemorrhage classes 1–5 (computed from table above): **0.163153**.

## Volume error (native spacing, mL)

| Metric | Value |
|--------|------:|
| Mean abs total hemorrhage volume error | **19.533** mL |
| Median | **4.909** mL |
| Max | **183.507** mL |

## Confidence

| Metric | Value |
|--------|------:|
| Mean study confidence | **0.8107** |
| Median | 0.8105 |
| Min | 0.7046 |

Softmax confidence; **not calibrated**.

## Inference speed

| Context | Value |
|---------|-------|
| Locked-test batch eval throughput | Not reported as a single latency figure in `evaluation_report.md` |
| Local FastAPI demos (`experiment_best30h`, CPU) | **19.278 s** (`ID_693f2d48_…`), **21.756 s** (`ID_0b10cbee_…`) from `reports/api/*/prediction_summary.json` |

## Strengths

- Reproducible locked-split baseline with full evaluation artifacts.
- Strongest subtype among hemorrhages on test: **SDH** (Dice ≈ 0.59).
- Val learning curve documented through early stop; best checkpoint clearly identified.
- Default interactive model on API v1 (`monai_best30h`). See [`docs/platform/api_v1_contract.md`](../platform/api_v1_contract.md).

## Weaknesses

- **EDH nearly broken** on test (Dice ≈ 0.011).
- IVH / IPH remain weak (Dice ≈ 0.15 / 0.23).
- Val→test gap: 0.351 → 0.257 macro Dice.
- Mean volume error inflated by large outliers (max ≈ 183.5 mL).

## Limitations

- Unweighted DiceCE loss (no class weights in V1).
- Native 512×512; no isotropic resampling.
- 2.5D context only; thick-slice anisotropy (~5 mm).
- Small labeled set (192); EDH rare in train.
- Confidence uncalibrated.
