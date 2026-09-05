# SSL campaign (branch `ssl_sep`)

**Research software — not for clinical use.**

Two separate pieces of work share this branch. Do not mix their metrics.

| Campaign | What it is | What it is not |
|---|---|---|
| **Labeling** | nnU-Net teacher `nnunet_fold0` (locked-test **0.454723**) inferred pseudo-labels on unlabeled CTs. Pool **1980** (562 anybleed / 1418 nobleed). Planned **1000**, finished **800**. | Not the 0.4965 claim |
| **Training** | Supervised nnU-Net 5-fold on the **163** locked train+val labeled cases (test never in CV). Compute limited us to **2 folds**. | Not MONAI SSL distillation (Phase 8 student was never trained) |

Preprocessing notebook: [`notebooks/ssl_preprocessing.ipynb`](../../notebooks/ssl_preprocessing.ipynb)  
UI / API id: `nnunet_ssl_2fold` (softmax ensemble of fold0 + fold1)  
Model card: [`docs/models/nnunet_ssl_2fold.md`](../models/nnunet_ssl_2fold.md)

Locked-test claim metric is unchanged from V1: **macro Dice, classes 1–5, n=29**, split hash `15473094f33156049d47396a2e6af100285831589c74714ef3c683448d39e7b1`.

## Locked-test outcomes (2 folds)

Same metric as V1 nnU-Net **0.454723**.

| Run | Epoch | Locked-test macro Dice |
|---|---|---|
| V1 nnU-Net `nnunet_fold0` (baseline) | ~196 (time-capped) | 0.454723 |
| New 5-fold fold0 | ~510 | **0.4802** |
| New 5-fold fold1 | 511 | **0.4965** (best so far) |
| New 5-fold fold1 | 679 | 0.4418 (below 0.45; fold-val EMA rose, locked test fell) |

Fold1 epoch 511 **beats** the previous 0.45 baseline. Epoch 679 is a known regression: do not serve it as “best.”

The UI **runs** a 2-fold softmax ensemble (`use_folds=(0, 1)`). A pooled **ensemble** locked-test Dice has **not** been scored. Registry / UI text cites the two **single-fold** numbers; do not invent an ensemble Dice.

## Conclusion — why only 2 folds

Limited computational power stopped a planned 5-fold at **fold 0 and fold 1**. Folds 2–4 were not trained.

### Timeframe (Kaggle T4, 12 h session cap)

| Work | Time |
|---|---|
| One Kaggle session | **12 h** hard cap |
| One 5-fold train toward ~1000 epochs | **several 12 h Save Versions** (resume from `checkpoint_latest.pth`) |
| Fold1 pack | **12 h timeout around epoch ~920** |
| Teacher LabelRun | **~52 s/case** on T4 → **800 cases ~11–12 h** (1000 cases ~14.5 h, does not fit one session) |
| Locked-test eval (29 cases, one fold) | **~30–60 min on T4**; ~128 min local CPU for fold0 |
| Serving `nnunet_ssl_2fold` | **about 2×** existing nnU-Net wait (minutes on CPU) |

Fold-val EMA is **not** the claim metric. Always re-score `checkpoint_best.pth` on the locked 29 cases.

## Checkpoint warning

Kaggle dataset `bhai-nnunet-fold1-ckpt` currently names `checkpoint_best.pth` as **later EMA (~epoch 775)**, not epoch 511.

For the UI copy:

- fold0: locked-test ~epoch 510 weights (**0.4802**)
- fold1: **epoch 511** weights (**0.4965**), not the later EMA file

If the epoch-511 file is missing, do **not** label the served model 0.4965.

Local copy status (this machine): [`weights_local_status.md`](weights_local_status.md). Fold1 epoch 511 `.pth` was **not** found; fold1 pack on disk is later EMA (~775).

Layout (gitignored):

```
checkpoints/nnunet_ssl_2fold/
  fold_0/checkpoint_best.pth
  fold_1/checkpoint_best.pth
  nnUNetPlans.json
  dataset.json
```

Docker’s v1.0.0 weights tarball still ships only MONAI + V1 `nnunet_fold0`. Mount the SSL folder locally to enable the third picker.

## What was not done

- Folds 2–4
- Phase 8 MONAI student (Control A vs Treatment B)
- Changing locked splits, HU clip `[-40, 120]`, or `(HU − 40) / 80`
