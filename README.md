# BrainHemorrhageAI

Research software that draws hemorrhage regions on a head CT and estimates volume.

**Not for clinical use. Not a medical device. Not a diagnosis.**

This `product_final` branch is the public demo: a website, four trained models, sample scans, and two short preprocessing notebooks.

## What you need

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- About **1.4 GB** download the first time (MONAI + nnU-Net + nnU-Net Ensemble + SSL)
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

First start downloads all four model weights if `checkpoints/` is empty (GitHub Releases `v1.1.0` and `v1.2.0`). Later starts reuse that folder. No extra files to copy.

## What the demo shows

| Model | What it is | Dice | IoU | Volume error |
|---|---|---|---|---|
| MONAI | Fast interactive demo | 0.257 locked test | 0.163 | 19.5 mL |
| nnU-Net | V1 3D comparator | 0.455 locked test | 0.313 | **14.3 mL** |
| nnU-Net Ensemble | Two nnU-Net folds averaged | **0.489** | **0.343** | 18.5 mL |
| SSL | Semi-supervised learning | **0.471** | — | — |

Dice / IoU: higher is better. Volume error: lower is better.

SSL means semi-supervised learning: one nnU-Net trained on labeled CTs plus teacher masks. IoU and volume error were not scored for that run.

## Preprocessing notebooks

Open these in Jupyter. They use `data/demo/` only — no GPU, no Docker.

| Notebook | One-line reading |
|---|---|
| [`notebooks/01_label_preprocessing.ipynb`](notebooks/01_label_preprocessing.ipynb) | We clip the CT gray values so bleeding stands out, then feed the same picture to a fast 2D model (MONAI) and a slower 3D model (nnU-Net). |
| [`notebooks/02_ssl_preprocessing.ipynb`](notebooks/02_ssl_preprocessing.ipynb) | Extra unlabeled scans were labeled by a teacher; SSL is semi-supervised learning on labeled + unlabeled CTs. |

To *run* a model, use Docker above.

## Sample scans

`data/demo/` — 17 research CTs from BHSD, named by hemorrhage type. The all-class example is `data/demo/01_all_classes/all_classes.nii.gz`.

## License

MIT for the software. Sample CTs are a small BHSD research subset (Wu et al., arXiv:2308.11298).
