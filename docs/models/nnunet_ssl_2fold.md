# Model Card — SSL nnU-Net 2-fold

**Status:** Research model on branch `ssl_sep` — deployed as `nnunet_ssl_2fold`  
**Sources:** Kaggle 5-fold locked-test evals, [`docs/ssl/README.md`](../ssl/README.md)  
**Serving:** `POST /api/v1/predict` with `model_id=nnunet_ssl_2fold` (research; slow on CPU)

This is a **softmax ensemble of fold 0 and fold 1** from the new 5-fold nnU-Net campaign. It is **not** the V1 `nnunet_fold0` (0.454723) and **not** a MONAI SSL student.

---

## Architecture

- **Framework:** nnU-Net `3d_fullres` (same family as `nnunet_fold0`)
- **Deployable id:** `nnunet_ssl_2fold`
- **Inference:** `NnUNetPredictor.initialize_from_trained_model_folder(..., use_folds=(0, 1))`
- **Weights:** `checkpoints/nnunet_ssl_2fold/fold_{0,1}/checkpoint_best.pth` (gitignored)

## Dataset

- **Source:** BHSD `label_192` (192 labeled studies)
- **5-fold JSON:** `data/metadata/nnunet_splits_5fold.json` (fingerprint `4714804f9f1cabd9`)
- **Dev set:** 163 = locked train+val; **locked test (29) never included**
- Unlabeled LabelRun (800/1000) is a **separate** pseudo-label campaign — see the preprocessing notebook

## Locked-test scores (single folds, n=29, classes 1–5)

| Run | Macro Dice |
|---|---|
| V1 `nnunet_fold0` | 0.454723 |
| New fold0 ~epoch 510 | 0.4802 |
| New fold1 epoch 511 | **0.4965** (best so far) |
| New fold1 epoch 679 | 0.4418 (do not serve) |

`macro_dice` in the model registry is **0.496525** (best **single** fold). A pooled **2-fold ensemble** Dice has **not** been scored — do not invent one.

## Checkpoint warning

Fold1 Kaggle `checkpoint_best.pth` may be a **later EMA (~epoch 775)**, not epoch 511. Serve fold1 **epoch 511** for the 0.4965 claim. If that file is missing, do not label the UI model 0.4965.

## Limitations

- Only **2 of 5** folds (Kaggle 12 h cap; several sessions per fold; fold1 timeout ~epoch 920)
- Ensemble is ~**2×** V1 nnU-Net latency (minutes on CPU)
- Not in the Docker v1.0.0 weights tarball — bind-mount `checkpoints/nnunet_ssl_2fold`
- Uncalibrated softmax confidence
- Not for clinical diagnosis

## Recommended use

Research comparison when quality matters more than wait time. Default interactive model remains `monai_best30h`. Compare Both still runs MONAI vs V1 `nnunet_fold0` only.
