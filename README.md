# BrainHemorrhageAI

Research software that draws hemorrhage regions on a head CT and estimates volume.

**Not for clinical use. Not a medical device. Not a diagnosis.**

This `product_final` branch is the public demo: a website, four trained models, sample scans, and two short preprocessing notebooks.

## What you need

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- About **1 GB** download the first time (MONAI + nnU-Net + SSL)
- Optional: the labeled + unlabeled nnU-Net weights (`checkpoint_best.pth`)
- A modern browser

## Run it

```powershell
git clone --branch product_final https://github.com/GuptaFalguni/BrainHemorrhageAI.git
cd BrainHemorrhageAI
docker compose up --build
```

Then open **http://localhost:3000**.

Upload a sample from this repo:

`data/demo/01_all_classes/all_classes.nii.gz`

Start with **MONAI** (~20 seconds). The three nnU-Net options take several minutes on a CPU.

First start downloads MONAI, V1 nnU-Net, and SSL weights from GitHub Release `v1.1.0` if `checkpoints/` is empty. Later starts reuse that folder.

To enable **Semi-sup. nnU-Net**, place `checkpoint_best.pth` in:

`checkpoints/nnunet_dataset502_semi/`

Plans and `dataset.json` are already in `config/models/nnunet_dataset502_semi/` and are copied next to the checkpoint automatically. That file is not in the v1.1.0 tarball.

## What the demo shows

Locked-test metric: macro Dice, classes 1–5, n=29.

| Model | Role | Dice | IoU | Volume error |
|---|---|---|---|---|
| MONAI | Fast interactive demo | 0.257 | 0.163 | 19.5 mL |
| nnU-Net | V1 research comparator | 0.455 | 0.313 | **14.3 mL** |
| SSL nnU-Net | 2-fold ensemble | **0.489** | **0.343** | 18.5 mL |
| Semi-sup. nnU-Net | 163 labels + 800 teacher masks | 0.402 | — | — |

Dice / IoU: higher is better. Volume error: lower is better.

Semi-sup. nnU-Net is a later labeled + unlabeled experiment. It is **weaker** than V1 and SSL on the locked test. IoU and volume error were not scored for that run. Do not present 0.402 as an upgrade over 0.489.

## Preprocessing notebooks

Open these in Jupyter. They use `data/demo/` only — no GPU, no Docker.

| Notebook | One-line reading |
|---|---|
| [`notebooks/01_label_preprocessing.ipynb`](notebooks/01_label_preprocessing.ipynb) | We clip the CT gray values so bleeding stands out, then feed the same picture to a fast 2D model (MONAI) and a slower 3D model (nnU-Net). |
| [`notebooks/02_ssl_preprocessing.ipynb`](notebooks/02_ssl_preprocessing.ipynb) | Extra unlabeled scans were labeled by a teacher, then two nnU-Net folds were averaged; that ensemble is the strongest drawer on the locked test (Dice 0.489). |

To *run* a model, use Docker above.

## Sample scans

`data/demo/` — 17 research CTs from BHSD, named by hemorrhage type. The all-class example is `data/demo/01_all_classes/all_classes.nii.gz`.

## License

MIT for the software. Sample CTs are a small BHSD research subset (Wu et al., arXiv:2308.11298).
