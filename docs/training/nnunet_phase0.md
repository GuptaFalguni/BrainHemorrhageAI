# nnU-Net Phase 0 — Prep Checklist & Kaggle Recipe

**Status:** Phase 0 implementation  
**Plan:** `docs/training/weak_vs_strong_model_plan.md`  
**Config:** `config/nnunet.yaml`  
**Code:** `src/nnunet_prep/`

---

## Why this phase exists

A 12h Kaggle Save Version is too expensive to debug dataset layout mistakes. Phase 0 builds and validates the nnU-Net raw dataset **locally** (CPU), locks the train/val fold to `splits.csv`, and documents the exact Kaggle cells for Phase 1.

---

## Design choices

| Choice | Decision | Reason |
|--------|----------|--------|
| Dataset ID | `Dataset501_BHSD` | Avoids MSD IDs 001–010 |
| Training cases in nnU-Net | **train + val only (163)** | Test never enters `labelsTr` |
| Test images | Optional `imagesTs` | For later nnU-Net predict; GT stays in BHSD for our metrics |
| Split file | `data/metadata/nnunet_splits_final.json` (1 fold) | Fair vs MONAI; no random 5-fold |
| Local conversion | Symlink preferred (`--symlink`) | Fast; no duplicate ~GB copies |
| nnU-Net install locally | **Not required** for conversion | Only needed on Kaggle for plan/train |

---

## Phase 0 checklist

| ID | Task | Command / artifact | Done? |
|----|------|--------------------|-------|
| 0.1 | Config + docs | `config/nnunet.yaml`, this file | [x] |
| 0.2 | Convert BHSD → nnU-Net raw | see below | [x] |
| 0.3 | Write locked `splits_final.json` | auto in convert; also versioned under `data/metadata/` | [x] |
| 0.4 | Validate (shapes, labels 0–5, no test leak) | `--validate-only` must print `"ok": true` | [x] |
| 0.5 | Kaggle notebook recipe reviewed | § Kaggle cells below | [x] |
| 0.6 | Time-cap / checkpoint copy policy | § Kill-safe training below | [x] |
| 0.7 | Optional Draft smoke | imports + `plan_and_preprocess` only; **no** long train | [ ] |

**Phase 0 exit criteria:** validation `"ok": true` and you would trust a 12h Save Version to emit downloadable checkpoints.

---

## Local commands

From project root (venv active, package installed):

```powershell
cd C:\Users\1016f\OneDrive\Desktop\BrainHemorrhageAI
.\.venv\Scripts\Activate.ps1
pip install -e .

# Convert (symlinks) + validate
python -m nnunet_prep.convert_bhsd_to_nnunet --symlink

# Re-validate later
python -m nnunet_prep.convert_bhsd_to_nnunet --validate-only
```

Expected raw layout:

```text
data/nnunet/nnUNet_raw/Dataset501_BHSD/
  dataset.json
  conversion_manifest.json
  imagesTr/   # 163 × *_0000.nii.gz
  labelsTr/   # 163 × *.nii.gz
  imagesTs/   # 29 × *_0000.nii.gz (no labels)
```

Versioned split (git):

```text
data/metadata/nnunet_splits_final.json
```

---

## Kaggle environment (Phase 1)

### Env vars

```bash
export nnUNet_raw=/kaggle/working/nnUNet_raw
export nnUNet_preprocessed=/kaggle/working/nnUNet_preprocessed
export nnUNet_results=/kaggle/working/nnUNet_results
```

### Install (pin in notebook; adjust if Kaggle CUDA wheel needs it)

```bash
pip install -q nnunetv2
```

Torch is usually preinstalled on Kaggle GPU images. If `nnunetv2` pulls a conflicting torch, install matching CUDA wheels first — verify once in a **short Draft** before Save Version.

### Data inputs (recommended)

1. Existing BHSD Kaggle dataset (same as MONAI run)  
2. Project code dataset including `src/nnunet_prep/` + `data/metadata/splits.csv` + `nnunet_splits_final.json`  
3. On Kaggle: run conversion into `$nnUNet_raw` (copy mode, not symlink)

---

## Kaggle notebook cells (Phase 1 skeleton)

### Cell A — paths + env

```python
import os
from pathlib import Path

CODE = Path("/kaggle/input/YOUR_CODE_DATASET/BrainHemorrhageAI")  # adjust
DATA = Path("/kaggle/input/YOUR_BHSD_DATASET")  # adjust to images / ground truths layout
WORK = Path("/kaggle/working")

os.environ["nnUNet_raw"] = str(WORK / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(WORK / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(WORK / "nnUNet_results")

for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
```

### Cell B — convert on Kaggle

Point `config/nnunet.yaml` image/label paths at the Kaggle BHSD mount (write a small override YAML or edit paths in a copied config). Then:

```python
import sys
sys.path.insert(0, str(CODE / "src"))
from nnunet_prep import load_nnunet_config, convert_bhsd_to_nnunet, validate_nnunet_raw

# Load config and override roots/paths for Kaggle before convert
cfg = load_nnunet_config(CODE / "config" / "nnunet.yaml")
# ... set cfg["images_dir"], cfg["labels_dir"], cfg["nnunet_raw_root"], cfg["splits_csv"] ...
manifest = convert_bhsd_to_nnunet(cfg, project_root=CODE, use_symlink=False)
report = validate_nnunet_raw(cfg, project_root=CODE)
assert report["ok"], report
```

> Practical tip: either copy metadata into `/kaggle/working` and set `project_root` accordingly, or pass absolute paths inside an overridden config dict.

### Cell C — plan + preprocess

```bash
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity
```

### Cell D — inject locked split (CRITICAL)

After preprocess creates `nnUNet_preprocessed/Dataset501_BHSD/`:

```python
import json, shutil
from pathlib import Path
import os

pre = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_BHSD"
src = Path("/kaggle/working/...") / "nnunet_splits_final.json"  # from conversion or code dataset
dst = pre / "splits_final.json"
shutil.copy2(src, dst)
splits = json.loads(dst.read_text())
assert len(splits) == 1
assert len(splits[0]["train"]) == 134
assert len(splits[0]["val"]) == 29
print("Locked split installed:", dst)
```

### Cell E — kill-safe training + checkpoint mirror

```bash
# Fold 0 only. nnU-Net writes under nnUNet_results; mirror into /kaggle/working/exports often.
nnUNetv2_train 501 3d_fullres 0
```

Parallel / periodic mirror (separate cell or background thread):

```python
import shutil, time
from pathlib import Path
import os

src_root = Path(os.environ["nnUNet_results"]) / "Dataset501_BHSD" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
export = Path("/kaggle/working/exports/nnunet_fold0")
export.mkdir(parents=True, exist_ok=True)

# Run this loop in a thread while training, or between resumed runs:
for name in ("checkpoint_best.pth", "checkpoint_latest.pth", "checkpoint_final.pth"):
    # search fold_0 folder
    matches = list(src_root.rglob(name))
    for m in matches:
        shutil.copy2(m, export / m.name)
```

**Policy:** prefer whatever is in `exports/` at job end. If the 12h axe falls, `checkpoint_latest.pth` / `checkpoint_best.pth` must already be under `/kaggle/working`.

### Cell F — optional resume (Phase 2 only)

```bash
nnUNetv2_train 501 3d_fullres 0 --c
```

---

## Kill-safe / time-cap policy

1. Use **Save Version**, not Draft, for the long train.  
2. Mirror checkpoints into `/kaggle/working/exports` periodically.  
3. Do **not** start 5-fold or `3d_lowres`+`3d_fullres` cascade in the same 12h.  
4. If preprocess finishes with &lt; ~9h left, still start train — partial epochs beat zero.  
5. Phase 2 = resume only (`--c`), not a new experiment.

---

## What “better than MONAI” means after Phase 1

Evaluate Model B on the **locked test 29** with the same metrics as `experiment_best30h`. Primary comparison: test macro Dice (classes 1–5). Per-class tables required; EDH may still be weak.

---

## Status board

| Item | Status |
|------|--------|
| Conversion code | Implemented |
| Locked split writer | Implemented |
| Local convert+validate run | **Done** (`ok: true`, 163 Tr + 29 Ts) |
| Kaggle path overrides smoke | Pending short Draft |
| Phase 1 Save Version | Not started |
