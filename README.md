# BrainHemorrhageAI

Research software that draws hemorrhage regions on a head CT and estimates volume.

**Not for clinical use. Not a medical device. Not a diagnosis.**

This `release` branch is the public demo: a website, three trained models, and sample scans. Training notebooks and research notes live on other branches.

## What you need

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- About **1 GB** download the first time (the three models)
- A modern browser

## Run it (4 commands)

```powershell
git clone --branch release https://github.com/GuptaFalguni/BrainHemorrhageAI.git
cd BrainHemorrhageAI
docker compose up --build
```

Then open **http://localhost:3000**.

Upload a sample from this repo, for example:

`data\demo\01_all_classes\all_classes.nii.gz`

Start with **MONAI** (~20 seconds). **nnU-Net** and **SSL nnU-Net** take several minutes on a CPU.

First start downloads model weights from GitHub Release `v1.1.0` if `checkpoints/` is empty. Later starts reuse that folder.

## What the demo shows

| Model | Role | Locked-test Dice | Volume error |
|---|---|---|---|
| MONAI | Fast interactive demo | 0.257 | 19.5 mL |
| nnU-Net | Stronger research model | 0.455 | **14.3 mL** |
| SSL nnU-Net | 2-fold ensemble | **0.489** | 18.5 mL |

Dice / IoU: higher is better. Volume error: lower is better.

The one longer explanation of the idea, the workflow, and these numbers is [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Sample scans

`data/demo/` — 17 research CTs from BHSD, named by hemorrhage type. See `data/demo/README.txt`.

## License

MIT for the software. Sample CTs are a small BHSD research subset (Wu et al., arXiv:2308.11298).
