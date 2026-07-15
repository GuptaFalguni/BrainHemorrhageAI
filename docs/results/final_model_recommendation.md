# Final Model Recommendation

**Date:** 2026-07-14 (Phase B locked-test comparison)  
**Evidence sources:**  
- `reports/evaluation/experiment_best30h_eval/`  
- `reports/evaluation/nnunet_dataset501_fold0_eval/`  
- `docs/results/comparison.md`, `docs/results/publication_comparison_tables.md`

---

## Recommendation

**Primary production / research segmentation model: nnU-Net Dataset501 fold 0**  
(`checkpoints/nnunet_dataset501_fold0/checkpoint_best.pth`)

**MONAI `experiment_best30h` / `monai_best30h`** remains the **locked weak baseline** and the **default interactive** API model.  
**nnU-Net fold 0 / `nnunet_fold0`** is available on **API v1** as the **research** model (sync; slow on CPU).  
Authoritative HTTP contract: [`docs/platform/api_v1_contract.md`](../platform/api_v1_contract.md).

---

## Why (measured evidence only)

On the **same 29 locked-test case IDs**, using **shared metric modules** and **native-space volume** evaluation:

| Evidence | MONAI | nnU-Net | Favors |
|----------|------:|--------:|--------|
| Test macro Dice | 0.257375 | **0.454723** | nnU-Net |
| Test micro Dice | 0.385548 | **0.551281** | nnU-Net |
| Mean IoU (classes 1–5) | 0.163153 | **0.313058** | nnU-Net |
| Mean volume abs error (mL) | 19.533 | **14.318** | nnU-Net |
| Median volume abs error (mL) | 4.909 | **3.733** | nnU-Net |
| Per-class Dice (all of EDH/SDH/SAH/IPH/IVH) | lower | **higher** | nnU-Net |

No locked-test metric in the above table favors MONAI.

---

## What the evidence does **not** support

| Claim | Verdict |
|-------|---------|
| “Ready for clinical deployment” | **Not supported** — no prospective clinical study; EDH Dice still 0.207; confidence uncalibrated |
| “nnU-Net is the interactive default” | **False** — default remains MONAI; nnU-Net is explicit `model_id=nnunet_fold0` |
| “nnU-Net is faster” | **False on measured CPU latency** — ~328 s/case vs MONAI ~20 s/case demos |
| “EDH is solved” | **False** — improved vs MONAI (0.011 → 0.207) but still weak |

---

## Operational caveats (facts)

1. nnU-Net checkpoint is **340.4 MB** with **~44.6M** parameters vs MONAI **7.5 MB** / **~0.65M**.  
2. Locked-test nnU-Net inference here used **CPU** and **no test-time mirroring**.  
3. Mean study confidence is **0.9936** (nnU-Net) vs **0.8107** (MONAI) — neither is calibrated; high nnU-Net confidence must **not** be treated as clinical certainty.  
4. Dual serving **is** implemented in Phase C (API v1). Production **Next.js** UI is not yet implemented (Streamlit interim only).

---

## Bottom line

For **segmentation quality on the locked BHSD test protocol**, choose **nnU-Net fold 0**.  
For **interactive demo latency**, choose **MONAI** (`monai_best30h`, API default).

Evidence for the quality recommendation is **sufficient** (complete locked-test reports for both models).  
Evidence for clinical production readiness is **insufficient**.
