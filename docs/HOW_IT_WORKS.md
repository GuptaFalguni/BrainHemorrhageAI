# How BrainHemorrhageAI works

**Research software — not for clinical use.** This page is the only product guide. It is written for a non-specialist who wants the whole idea, the results, and how to run the demo.

## The problem in one paragraph

A non-contrast head CT can show bleeding inside the skull (intracranial hemorrhage). Radiologists look for *where* the blood is and *how much* there is. This project is a research tool that takes a CT file, draws those regions, names the subtype it thinks it sees, and estimates volume in milliliters. It is a study aid and a thesis demo. It is **not** a diagnosis and must never be used to treat a patient.

## The five hemorrhage types

The models know six labels. 0 is background (no blood). 1–5 are the subtypes this project reports:

| Code | Name | Plain meaning |
|---|---|---|
| EDH | Epidural | Blood between the skull and the tough outer membrane |
| SDH | Subdural | Blood under that membrane, often a crescent shape |
| SAH | Subarachnoid | Blood in the spaces around the brain surface |
| IPH | Intraparenchymal | Blood inside the brain tissue |
| IVH | Intraventricular | Blood inside the fluid-filled ventricles |

A single scan can have more than one type. BHSD (the public research set this work uses) has **192** labeled CTs. Only **one** of those 192 has all five types at once. That scan is in the demo folder as `all_classes.nii.gz`.

## What happens when you press Analyze

1. You upload a `.nii` or `.nii.gz` CT (or pick one from `data/demo/`).
2. You choose **MONAI**, **nnU-Net**, **SSL nnU-Net**, or **Compare** (any two, or all three).
3. The website sends the file to a local API running in Docker.
4. The chosen model labels every voxel: background or one of the five types.
5. The app shows:
   - an overlay (the drawing on the CT)
   - estimated volume in mL
   - a research “confidence” number (how sure the model looks — **not** a clinical probability)
   - a short research summary and a downloadable report

Nothing is sent to a hospital system. On a typical laptop this is CPU-only.

```
Upload CT  →  pick a model  →  Docker API  →  mask + volume + report
                 ↑
           weights from GitHub Release v1.1.0 (first run only)
```

## The three models (and what we found)

All scores below are on the **same locked 29-case test set** that was never used for training. “Macro Dice” is overlap with the expert drawing (0 = none, 1 = perfect). “IoU” is another overlap score. “Volume error” is how many milliliters the predicted blood volume missed the expert volume; **lower is better**.

| Model | What it is | Dice | IoU | Volume error (mean) | Typical wait on CPU |
|---|---|---|---|---|---|
| **MONAI** | Smaller 2.5D U-Net. Default for a live demo. | 0.257 | 0.163 | 19.5 mL | ~20 seconds |
| **nnU-Net** | 3D full-resolution, one fold (V1 research model). | 0.455 | 0.313 | **14.3 mL** | ~5+ minutes |
| **SSL nnU-Net** | Softmax ensemble of two new 5-fold runs (fold 0 + fold 1). | **0.489** | **0.343** | 18.5 mL | about 2× nnU-Net |

How to read that table:

- **SSL wins overlap** (Dice and IoU). The drawing matches the expert a bit better than V1 nnU-Net.
- **V1 nnU-Net wins volume** (14.3 mL vs 18.5 mL). It is closer on “how many milliliters of blood.”
- **MONAI wins speed.** Use it when you are showing the interface. It is the weakest on overlap.
- Best **single** SSL fold was 0.4965. The app does **not** serve that one fold. It serves the **2-fold ensemble** (0.489).

A model can overlap better and still be worse on milliliters. Extra voxels far from the bleed add volume; missing a thin sheet of blood can also shift the count.

SSL stopped at two of five planned folds because of GPU time limits (Kaggle 12-hour sessions). That is a compute limit, not a claim that two folds are theoretically enough.

## Which model should you click?

- **Talk or first look:** MONAI
- **“Show me the stronger research drawing”:** nnU-Net
- **“Show the latest campaign model”:** SSL nnU-Net
- **Side-by-side:** Compare, then pick any pair or all three (they run one after another)

## Demo files on this branch

The clone includes `data/demo/` so you do not need the full BHSD download.

| Folder | What to say in a presentation |
|---|---|
| `01_all_classes/all_classes.nii.gz` | “One upload, every class the model knows.” |
| `02_four_classes/` | Large scannable cases missing exactly one subtype |
| `03_by_class/01_EDH` … `05_IVH` | “This is a clear EDH / SDH / SAH / IPH / IVH example.” `_1` is the larger case. |

Do **not** upload files under `ground_truth/` or the `.txt` notes. Those are expert masks and captions for you, not inputs.

Source of the images: BHSD `label_192` (Wu et al., [arXiv:2308.11298](https://arxiv.org/abs/2308.11298)). This is a small research subset for software illustration.

## How someone else runs this on their machine

They need Docker Desktop. Then:

```powershell
git clone --branch release https://github.com/GuptaFalguni/BrainHemorrhageAI.git
cd BrainHemorrhageAI
docker compose up --build
```

Open http://localhost:3000 and upload `data\demo\01_all_classes\all_classes.nii.gz`.

The first start downloads **all three models** (~1 GB) from GitHub Release **v1.1.0**. Later starts reuse the `checkpoints/` folder. nnU-Net needs extra Docker shared memory; this compose file already sets `shm_size: 2gb` so that path does not crash.

`git clone` without `--branch release` still gets the older default branch. This product tree is **`release` only**. `main` and `ssl_sep` are left as they are.

## What this project does not do

- It does not replace a radiologist.
- Confidence is uncalibrated research output.
- It does not include the full 192-scan training set.
- Full research notebooks stay on the `ssl_sep` branch. This branch has two short teaching notebooks in `notebooks/`.
- EDH remains the hardest class, especially for MONAI.

## Teaching notebooks

| File | Audience |
|---|---|
| [`notebooks/01_monai_nnunet_preprocessing.ipynb`](../notebooks/01_monai_nnunet_preprocessing.ipynb) | How a CT is clipped, normalized, and fed to MONAI vs nnU-Net |
| [`notebooks/02_ssl_preprocessing.ipynb`](../notebooks/02_ssl_preprocessing.ipynb) | Same window; what the SSL labeling run was; what the 2-fold ensemble is |

Both use `data/demo/`. They do not start the Docker predictor.

## Pointers

- Software license: MIT (`LICENSE`)
- Dataset citation: Wu et al., BHSD, arXiv:2308.11298
- nnU-Net citation: Isensee et al., Nature Methods 2021
