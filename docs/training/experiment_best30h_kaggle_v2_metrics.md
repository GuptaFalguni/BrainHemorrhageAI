# Experiment best30h — Kaggle Version #2 metrics

**Notebook:** `falguni1234/brainhemorrhageai-gpu-train`  
**Run:** Save Version #2 (GPU T4 ×2)  
**Experiment ID:** `experiment_best30h`  
**Config:** 45 epochs max, early stopping patience 10, `resume: true`  
**Code:** disk volume cache enabled (`fingerprint=7f10bed46aac`)  
**Last updated:** 2026-07-12 (run complete — early stop epoch 34)

## Setup notes

- Precache completed in ~153s (train 134 + val 29 = 163 volumes).
- Deterministic CE warning on first step is expected (`warn_only=True`); not a failure.
- Checkpoints path: `checkpoints/experiment_best30h/{best_model,last_model}.pt`
- Output tab may show `0 B` until the version finishes packaging; logs confirm saves.
- Log viewer may replay early setup lines mid-file; training timeline below is deduplicated.

## Health check (epoch 1–31)

| Check | Status |
|-------|--------|
| Pipeline / CUDA / cache | OK |
| Train loss trend | Falling (1.22 → 0.79) — learning |
| Val Dice trend | Rising overall (0 → **0.351** peak) — learning |
| Checkpointing | OK (`last` every epoch; `best` on improvements) |
| Val noise | Normal (dips at 9, 12, 14, 18, 21, 31) |
| Early stopping | Patience 10; best @ **24**; as of epoch 31 → **7** epochs without new best |
| Run status at capture | Still progressing through epoch 31+ |

**Verdict:** Everything looks healthy. Use **`best_model.pt` (epoch 24, Dice 0.351169)** as the current best labeled checkpoint.

## Epoch metrics (complete through 31)

| Epoch | Train loss | Val loss | Val macro Dice | Val macro IoU | Duration (s) | LR | Best? |
|------:|-----------:|---------:|---------------:|--------------:|-------------:|-----:|:-----:|
| 1 | 1.221105 | 0.890143 | 0.000000 | 0.000000 | 1283.0 | 1.00e-04 | yes |
| 2 | 0.864462 | 0.852928 | 0.070771 | 0.042980 | 1208.8 | 9.99e-05 | yes |
| 3 | 0.846792 | 0.842420 | 0.104694 | 0.070899 | 1209.1 | 9.95e-05 | yes |
| 4 | 0.840169 | 0.837842 | 0.115859 | 0.078582 | 1193.2 | 9.89e-05 | yes |
| 5 | 0.835907 | 0.834348 | 0.139514 | 0.092211 | 1200.4 | 9.81e-05 | yes |
| 6 | 0.831646 | 0.831500 | 0.188206 | 0.121083 | 1208.2 | 9.70e-05 | yes |
| 7 | 0.827406 | 0.827986 | 0.212546 | 0.137892 | 1208.3 | 9.57e-05 | yes |
| 8 | 0.824051 | 0.823297 | 0.256433 | 0.167393 | 1194.7 | 9.41e-05 | yes |
| 9 | 0.820808 | 0.826138 | 0.243167 | 0.156258 | 1209.4 | 9.24e-05 | no |
| 10 | 0.818434 | 0.819989 | 0.283501 | 0.184644 | 1202.8 | 9.05e-05 | yes |
| 11 | 0.815461 | 0.818330 | 0.314555 | 0.207724 | 1203.4 | 8.83e-05 | yes |
| 12 | 0.813679 | 0.819697 | 0.267179 | 0.173163 | 1201.4 | 8.60e-05 | no |
| 13 | 0.811961 | 0.817889 | 0.302587 | 0.200019 | 1193.7 | 8.35e-05 | no |
| 14 | 0.809883 | 0.820675 | 0.273355 | 0.178036 | 1191.9 | 8.08e-05 | no |
| 15 | 0.808173 | 0.816682 | 0.319731 | 0.211740 | 1205.6 | 7.80e-05 | yes |
| 16 | 0.806827 | 0.816145 | 0.322578 | 0.213124 | 1206.1 | 7.50e-05 | yes |
| 17 | 0.805240 | 0.814935 | 0.334824 | 0.222322 | 1191.9 | 7.19e-05 | yes |
| 18 | 0.803728 | 0.816715 | 0.314331 | 0.207909 | 1191.2 | 6.87e-05 | no |
| 19 | 0.802356 | 0.816518 | 0.317474 | 0.210944 | 1204.2 | 6.55e-05 | no |
| 20 | 0.800775 | 0.814670 | 0.333827 | 0.222425 | 1204.6 | 6.21e-05 | no |
| 21 | 0.799522 | 0.816119 | 0.303160 | 0.199953 | 1200.6 | 5.87e-05 | no |
| 22 | 0.798024 | 0.814171 | 0.336913 | 0.226080 | 1199.1 | 5.52e-05 | yes |
| 23 | 0.796755 | 0.815000 | 0.335577 | 0.223255 | 1202.0 | 5.17e-05 | no |
| 24 | 0.795421 | 0.814152 | **0.351169** | **0.236704** | 1209.4 | 4.83e-05 | **yes (best)** |
| 25 | 0.794228 | 0.813925 | 0.344809 | 0.231249 | 1205.6 | 4.48e-05 | no |
| 26 | 0.792913 | 0.814648 | 0.346786 | 0.232333 | 1197.0 | 4.13e-05 | no |
| 27 | 0.791956 | 0.815043 | 0.337869 | 0.225551 | 1199.4 | 3.79e-05 | no |
| 28 | 0.790917 | 0.814770 | 0.344272 | 0.230451 | 1199.3 | 3.45e-05 | no |
| 29 | 0.789864 | 0.814700 | 0.345818 | 0.231348 | 1214.6 | 3.13e-05 | no |
| 30 | 0.789005 | 0.813890 | 0.344278 | 0.229253 | 1201.7 | 2.81e-05 | no |
| 31 | 0.787969 | 0.816071 | 0.333401 | 0.221820 | 1202.1 | 2.50e-05 | no |

## Final outcome (Version #2)

| Item | Value |
|------|--------|
| Stop reason | **Early stopping at epoch 34** (patience 10) |
| Best val macro Dice | **0.351169** (epoch 24) → `best_model.pt` |
| **Test macro Dice** | **0.257375** (locked test split, once) |
| Experiment folder | `reports/experiments/experiment_best30h` |
| Log line | `Baseline experiment complete: experiment_best30h` |

## Snapshot

- **Best checkpoint:** epoch **24**, val macro Dice **0.351169**, val macro IoU **0.236704**
- **Test Dice** is lower than val Dice — normal gap on small test set (29); use test number for reporting
- **Pace:** ~20 min/epoch

## Best-checkpoint log excerpt

```
Epoch 24 — New best checkpoint (macro Dice=0.351169)
Early stopping triggered at epoch 34.
Baseline experiment complete: experiment_best30h
Best val Dice: 0.3511686623096466
Test Dice: 0.2573750615119934
```

## Next actions

1. Wait until Kaggle status is **Complete** (may lag a few minutes after last log while Output is packaged).
2. Download from Output:
   - `best_model.pt` (V1 model)
   - `last_model.pt`
   - `reports/experiments/experiment_best30h/` if present
3. Prefer **`best_model.pt`** for inference / thesis V1 baseline.
4. Optional later: resume / nnU-Net / unlabeled — only after this checkpoint is saved locally.
