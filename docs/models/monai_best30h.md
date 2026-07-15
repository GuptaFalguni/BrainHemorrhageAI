# Model Card — MONAI `experiment_best30h`

**Status:** Locked Version 1 interactive baseline (Model A) — default API model `monai_best30h`  
**Sources:** project configs, evaluation reports, registries, API v1  
**Serving:** Default for **`POST /api/v1/predict`** (omit `model_id` or pass `monai_best30h`)

Canonical metrics: [`docs/results/comparison.md`](../results/comparison.md)  
API: [`docs/platform/api_v1_contract.md`](../platform/api_v1_contract.md)

---

## Architecture

- **Name:** `monai_2d_unet`
- **Family:** MONAI 2D U-Net
- **Spatial dims:** 2 (axial slices)
- **Input channels:** 3 (2.5D: center ±1 neighbors; edge replication)
- **Output channels:** 6 (BG, EDH, SDH, SAH, IPH, IVH)
- **Encoder channels:** `[32, 64, 128, 256]`
- **Strides:** `[2, 2, 2]`
- **Config:** `config/model.yaml`
- **Deployable id:** `monai_best30h`

## Dataset

- **Dataset:** BHSD `label_192`
- **Split:** Locked 134 / 29 / 29 — `data/metadata/splits.csv` (seed 42)

## Training configuration

- **Experiment:** `experiment_best30h`
- **Training config:** `config/kaggle_training.yaml`
- **Best epoch:** 24 (early stopping)
- **Loss:** DiceCE (unweighted — V1)
- **Seed:** 42

## Preprocessing

- HU clip `[-40, 120]`; normalize `(HU − 40) / 80`
- No spatial resampling (native grid)
- Config: `config/preprocessing.yaml`

## Checkpoint path

```
checkpoints/experiment_best30h/best_model.pt
```

## Input / output (deployment)

| | |
|--|--|
| **Input** | NIfTI CT; MONAI path via `src/inference` wrapped by `MonaiBackend` |
| **Output** | 6-class segmentation; volumes in native mL; softmax confidence |

## Evaluation summary (locked test)

| Metric | Value |
|--------|------:|
| Val macro Dice (best ckpt) | 0.351169 |
| **Test macro Dice** | **0.257375** |
| Test micro Dice | 0.385548 |
| Mean / median volume error (mL) | 19.533 / 4.909 |
| Mean study confidence | 0.8107 |
| Local CPU demo latency | ~19–22 s/case |

Per-class test Dice: EDH 0.011 · SDH 0.591 · SAH 0.306 · IPH 0.229 · IVH 0.150  
Full report: `reports/evaluation/experiment_best30h_eval/`

## Known limitations

- EDH Dice ≈ 0.01 on locked test.
- Uncalibrated softmax confidence.
- 2.5D context; anisotropic CT.
- Weaker segmentation quality than nnU-Net on the same locked test.

## Recommended use

- **Interactive demos and engineering baseline** (default model).
- **Thesis weak baseline** for comparison against nnU-Net.
- **Not for:** Clinical decision support.

## Inference notes

- Prefer API v1 or `UnifiedInferencePipeline` with `monai_best30h`.
- CLI legacy path: `src/inference/predict_scan.py` still valid for MONAI-only debugging.
- Default deployable model is **`monai_best30h`** (not `experiment_sanity`).
