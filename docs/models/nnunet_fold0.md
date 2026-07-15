# Model Card — nnU-Net Dataset501 fold 0

**Status:** Locked research quality model (Model B) — deployed as `nnunet_fold0`  
**Sources:** nnU-Net reports, locked-test eval, registries, API v1  
**Serving:** Available via **`POST /api/v1/predict`** with `model_id=nnunet_fold0` (research category; slow on CPU)

Canonical metrics comparison: [`docs/results/comparison.md`](../results/comparison.md)  
API: [`docs/platform/api_v1_contract.md`](../platform/api_v1_contract.md)

---

## Architecture

- **Framework:** nnU-Net (self-configuring)
- **Configuration:** `3d_fullres`
- **Network class (plans):** `dynamic_network_architectures.architectures.unet.PlainConvUNet`
- **Stages / features:** 7 stages; features `[32, 64, 128, 256, 320, 320, 320]`
- **Conv:** `Conv3d` with InstanceNorm3d / LeakyReLU (per plans in training log)
- **Fold:** 0
- **Deployable id:** `nnunet_fold0` (`config/models/registry.yaml`)

## Dataset

- **Source:** BHSD `label_192` converted via `src/nnunet_prep/`
- **nnU-Net name:** `Dataset501_BHSD`
- **Labels:** 0–5 (BG, EDH, SDH, SAH, IPH, IVH)
- **Split:** 134 train / 29 val from locked CSV; **29 test held out**
- **Locked fold file:** `data/metadata/nnunet_splits_final.json`

## Training configuration

- **Plans:** nnU-Net default plans (`nnUNetPlans`)
- **Batch size (plans):** 2
- **Patch size:** `[28, 256, 256]`
- **Epochs completed:** ~196 (time-capped; not full default schedule)
- **Best EMA pseudo Dice:** 0.4791 (recorded after epoch 184)
- **Compile:** `torch.compile` enabled on Kaggle run

## Preprocessing

- nnU-Net `DefaultPreprocessor` + `CTNormalization`
- Resampling to plans spacing ≈ `[2.70, 0.488, 0.488]`
- Project conversion config: `config/nnunet.yaml`

## Checkpoint path

```
checkpoints/nnunet_dataset501_fold0/checkpoint_best.pth
```

## Input / output (deployment)

| | |
|--|--|
| **Input** | CT NIfTI via `UnifiedInferencePipeline` / API upload |
| **Backend** | `src/deployment/inference/nnunet_backend.py` |
| **Output** | Multi-class mask + probabilities where exported; volumes/confidence via shared evaluation helpers |

## Evaluation summary (locked test)

| Metric | Value |
|--------|------:|
| Best val EMA pseudo Dice | 0.4791 |
| **Locked-test macro Dice** | **0.454723** |
| Locked-test micro Dice | 0.551281 |
| Mean IoU (classes 1–5) | 0.313058 |
| Mean / median volume abs error (mL) | 14.318 / 3.733 |
| Mean study confidence | 0.9936 (uncalibrated) |
| CPU latency (timed cases) | ~328 s/case mean (n=28) |

Full eval: `reports/evaluation/nnunet_dataset501_fold0_eval/`

## Known limitations

- EDH Dice still weak (~0.207 on locked test) though improved vs MONAI.
- One fold; training truncated by Kaggle time cap.
- Very slow on CPU — use only when quality matters.
- Softmax confidence is not calibrated.
- Not for clinical diagnosis.

## Recommended use

- **Research / thesis primary segmentation quality model.**
- **Not** the interactive default (use `monai_best30h` for demos).

## Hardware used

- Kaggle GPU T4 ×2 for training; local CPU acceptable for occasional research inference.
