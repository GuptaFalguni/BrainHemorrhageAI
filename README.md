# BrainHemorrhageAI

Research-grade multi-class intracranial hemorrhage CT segmentation on BHSD (`label_192`).

**Research Software — Not for Clinical Use.**

## Start here

**[`docs/PROJECT_MASTER_GUIDE.md`](docs/PROJECT_MASTER_GUIDE.md)** — single project entry.

| Doc | Role |
|-----|------|
| [`docs/DOCUMENTATION_INDEX.md`](docs/DOCUMENTATION_INDEX.md) | Full doc map |
| [`docs/platform/api_v1_contract.md`](docs/platform/api_v1_contract.md) | HTTP API (authoritative) |
| [`docs/platform/frontend_spa_workspace_design.md`](docs/platform/frontend_spa_workspace_design.md) | Canonical product UI |
| [`docs/results/comparison.md`](docs/results/comparison.md) | MONAI vs nnU-Net locked-test |

## Current status

| Layer | Status |
|-------|--------|
| MONAI `monai_best30h` | Interactive API default |
| nnU-Net `nnunet_fold0` | Research model (slower on CPU) |
| Unified inference + clinical | Implemented |
| API v1 | Implemented |
| Next.js SPA (`web/`) | Delivered — single-page workspace |
| Streamlit | Legacy interim only (`frontend/app.py`) |

Locked-test macro Dice: MONAI **0.257** · nnU-Net **0.455** (same 29 test cases).

## Quick install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[api]"
# optional research model:
pip install -e ".[api,nnunet]"
```

Place model weights under `checkpoints/` (not in git). Frontend:

```powershell
cd web
npm install
npm run dev -- -p 3000
```

## Quick API + UI

```powershell
# Terminal 1 — API
uvicorn api.v1.app:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI (http://localhost:3000)
cd web
npm run dev -- -p 3000
```

## Dataset

BHSD — Wu et al., [arXiv:2308.11298](https://arxiv.org/abs/2308.11298). Place labeled data under `data/raw/label_192/` (not shipped in-repo). Locked split: `data/metadata/splits.csv` (134/29/29).

## License

MIT — see `LICENSE`.
