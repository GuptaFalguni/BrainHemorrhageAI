# BrainHemorrhageAI — work done so far, what this repo covers, and price

**Project:** BrainHemorrhageAI  
**Branch this document describes:** `product_final`  
**Total agreed price:** **₹18,000** (eighteen thousand Indian rupees)  
**Status of the work below:** complete and delivered in this demo repo  

**Not for clinical use. Not a medical device. Not a diagnosis.**  
This is research software. It draws hemorrhage regions on a head CT and estimates volume. It must not be used for diagnosis, triage, or treatment.

---

## In one page (non-technical)

The job was to take research head CT scans, teach several AI models to **outline bleeding** (hemorrhage) and **estimate how much blood is there**, then wrap that into a **website you can run on a laptop**.

A person can:

1. Open the website in a browser.
2. Upload a CT file (`.nii` / `.nii.gz`).
3. Choose one or more of **four models**.
4. See a colored overlay, a volume estimate, a short research summary, and a comparison table.
5. Download a report.

The models were trained and scored on a **locked test set** (29 scans that the models were not allowed to practice on). Those scores are the fair comparison numbers in this document.

| What you see on the website | Plain-language meaning | Headline score |
|---|---|---|
| **MONAI** | Fast “first look” model. Good for demos. | Dice **0.257**, volume error **19.5 mL** |
| **nnU-Net** | Stronger 3D research model (Version 1). Best volume estimate. | Dice **0.455**, volume error **14.3 mL** |
| **nnU-Net Ensemble** | Two nnU-Net versions averaged. Best overlap with expert outlines. | Dice **0.489**, IoU **0.343**, volume error **18.5 mL** |
| **SSL** | Semi-supervised: extra unlabeled scans were given teacher-made labels, then one nnU-Net was trained on labeled + unlabeled CTs. | Displayed Dice **0.471** (IoU and volume error not scored) |

**Dice** and **IoU** measure “how much does the AI outline match the expert outline?” Higher is better. **Volume error** is “how many milliliters off is the blood-volume guess?” Lower is better.

---

## Words used in this document

| Term | Everyday meaning |
|---|---|
| **CT** | A 3D X-ray of the head, stored as many slices. |
| **Hemorrhage / ICH** | Bleeding inside the head. |
| **Overlay** | Color drawn on the scan where the model thinks blood is. |
| **Volume (mL)** | Estimated amount of bleeding, in milliliters. |
| **Dice** | Overlap score from 0 to 1. 1 = perfect match with the expert drawing. |
| **IoU** | Another overlap score. Same idea as Dice, slightly different formula. Higher is better. |
| **Locked test** | 29 scans held back until the end so scores are honest. Macro Dice is reported for classes 1–5 (the five bleed types, not background). |
| **HU window** | We clip CT gray values to **−40 to 120** Hounsfield units so bleeding stands out. Same window for every model. |
| **Fold** | One trained copy of nnU-Net on a different split of the training scans. |
| **Ensemble** | Average the predictions of two folds instead of trusting one. |
| **SSL (semi-supervised learning)** | Use expert-labeled scans **plus** extra scans that a “teacher” model labeled automatically. |
| **Confidence** | The model’s own “how sure am I?” number. Research only — **not** a medical probability. |

The five bleed types the project looks for:

| Code | Full name | Everyday picture |
|---|---|---|
| **EDH** | Epidural | Blood between skull and dura (outer covering). Hardest type for these models. |
| **SDH** | Subdural | Blood under the dura, along the outer brain surface. |
| **SAH** | Subarachnoid | Blood in the spaces around the brain. |
| **IPH** | Intraparenchymal | Blood inside brain tissue. |
| **IVH** | Intraventricular | Blood inside the fluid spaces (ventricles). |

---

## Dataset

Every model in this project was trained and scored on the public **Brain Hemorrhage Segmentation Dataset (BHSD)** — research head CTs with expert outlines for five bleed types. This is **not** hospital patient data collected for this product.

**In plain language:** BHSD is a published set of 3D head CTs. Some scans have a full expert drawing on every voxel (pixel-level). Many more scans only have a note per slice (slice-level). We used the fully drawn scans to train MONAI, nnU-Net, and the Ensemble. For SSL we also used extra scans that had **no** expert drawing; a teacher model drew those, then SSL trained on both.

### Links

| What | Link |
|---|---|
| Paper (arXiv) | [https://arxiv.org/abs/2308.11298](https://arxiv.org/abs/2308.11298) |
| Official dataset + code | [https://github.com/White65534/BHSD](https://github.com/White65534/BHSD) |
| Download (Hugging Face) | [https://huggingface.co/datasets/WuBiao/BHSD](https://huggingface.co/datasets/WuBiao/BHSD) |
| Citation | Wu et al., *BHSD: A 3D Multi-class Brain Hemorrhage Segmentation Dataset*, MLMI 2023 (MICCAI workshop) |

BHSD is built from reconstructed **RSNA** head CTs. Authors: Wu, Xie, Zhang, Ge, Yaxley, Bahadir, Wu, Liu, To.

### What BHSD contains (public dataset)

| Part | Count | What it is |
|---|---|---|
| Pixel-level labeled CTs | **192** | Full expert outline on the 3D volume (EDH, SDH, SAH, IPH, IVH) |
| Extra CTs with slice-level labels | **~2,000** | Presence/type of bleed per slice; used here as the unlabeled pool for SSL |
| Total 3D scans in BHSD | **~2,192** | Each scan is typically 24–40 slices of 512×512 |
| Bleed types | 5 | Same five codes this website uses |

How **this project** split the 192 labeled scans (not the paper’s 96/96 split):

| Split | Count | Role |
|---|---|---|
| Train + validation | **163** | Models may practice on these |
| Locked test | **29** | Never used for training. All headline Dice / IoU / volume-error numbers |
| Demo subset in this repo | **17** | Small public samples in `data/demo/` so reviewers can click through without downloading BHSD |

The full 192 training CTs are **not** shipped in `product_final`. Only the 17 demo scans are. Training data lived at `data/raw/label_192/` during model development.

### Which model used what data

| Website name | Internal dataset name | Trained on | Tested on | Notes |
|---|---|---|---|---|
| **MONAI** | `BHSD_label_192` | 192 labeled BHSD CTs (train/val from that set) | Locked test **n = 29** | Supervised only. No unlabeled scans. |
| **nnU-Net** | `BHSD_label_192` → nnU-Net **Dataset501** | Same 192 labeled family; fold 0 of Dataset501 | Locked test **n = 29** | Images/labels from `data/raw/label_192/`. Test held out (`nnunet_splits_final.json`). |
| **nnU-Net Ensemble** | `BHSD_label_192` → Dataset501, **folds 0 and 1** | **163** labeled train+val CTs (5-fold splits; test never in folds) | Locked test **n = 29** | Same labeled BHSD as V1. No teacher masks. Two folds averaged. |
| **SSL** | `BHSD_label_163_plus_800_teacher` → nnU-Net **Dataset502** (`BHSD_semi`) | **163** expert-labeled CTs **+ 800** unlabeled CTs with teacher-made outlines | Locked test **n = 29** | Semi-supervised. Unlabeled pool comes from BHSD’s extra (non-pixel-labeled) scans. |

Same CT window for all four: clip HU to **−40 to 120**, then normalize as (HU − 40) / 80, following the BHSD paper (Section 3.0.2).

**Demo files a reviewer can open without BHSD:** `data/demo/` (see `data/demo/MANIFEST.csv`). Example: `data/demo/01_all_classes/all_classes.nii.gz`.

---

## What this repository covers

This `product_final` repo is the **public demo**: website + API + four models + sample scans + two short notebooks.

### Included

| Area | What was built |
|---|---|
| **CT preparation** | Locked brain window (−40 to 120 HU), same recipe as the BHSD paper (Wu et al., arXiv:2308.11298). Normalization uses mean 40 / std 80. |
| **Four runnable models** | MONAI, nnU-Net (V1), nnU-Net Ensemble, SSL. Each can draw an overlay and estimate volume. |
| **Honest scoring** | Locked-test metrics stored in `config/experiments/registry.yaml` and shown in the UI. No invented numbers. |
| **Website** | Upload a scan, pick models, run analysis, compare, read a research summary, download a report, read “learn” explainers. |
| **API** | REST service (`/api/v1/health`, `/models`, `/predict`, reports). Accepts `.nii` / `.nii.gz`. |
| **Docker** | One command: `docker compose up --build`. Site at **http://localhost:3000**, API at **http://localhost:8000**. |
| **Auto-download of weights** | First start pulls all four models if `checkpoints/` is empty: GitHub Release **v1.1.0** (MONAI + nnU-Net + Ensemble) and **v1.2.0** (SSL). Later starts reuse that folder. |
| **Sample CTs** | 17 research scans from BHSD in `data/demo/`, named by hemorrhage type, with matching expert labels. |
| **Teaching notebooks** | Two Jupyter notebooks using **demo data only** (no GPU, no Docker required to *look* at preprocessing). |
| **Research summary text** | Hemorrhage yes/no (model estimate), subtypes, volume, confidence, limitations. |
| **License** | MIT for the software. Sample CTs are a small BHSD research subset. |

### Website sections (what a user actually sees)

1. **Hero** — what the tool is.  
2. **Upload** — choose a NIfTI CT.  
3. **Model select** — MONAI, nnU-Net, nnU-Net Ensemble, SSL.  
4. **Analyze** — run the chosen model(s).  
5. **Progress** — wait time (MONAI ~20 s on CPU; nnU-Net options take several minutes).  
6. **Results** — overlay, volume, confidence.  
7. **Comparison** — side-by-side locked-test table.  
8. **Clinical summary** — research findings text only.  
9. **Downloads** — report / overlay files.  
10. **Learn** — plain-language Dice, IoU, volume, model comparison.  
11. **Disclaimer** — not for clinical use.

### Intentionally not included (and not billed as clinical product work)

- Use as a hospital device, diagnosis, triage, or treatment advice.
- **Severity grades** (mild / moderate / severe). Rules exist in config but status is **`not_configured`** until literature-backed cutoffs are frozen.
- Treatment **recommendations** (list is empty on purpose).
- Calibrated clinical probability (confidence is a softmax statistic only).
- The full BHSD training dataset inside this demo branch (only 17 sample CTs ship here).
- Uncertainty maps.

---

## Work done for each model

All four models share the same CT window and the same five bleed labels. They differ in how they look at the scan and how they were trained.

### 1. MONAI — fast interactive demo

**In plain language:** A quicker, lighter model. It looks at **three neighboring 2D slices** at a time (2.5D), not the whole 3D block. Useful when someone is clicking through the website and should not wait minutes.

**Work completed**

- Built a MONAI 2D U-Net with 2.5D input (3 input channels, 6 output classes: background + 5 bleed types).
- Trained on the labeled BHSD set (**BHSD_label_192**).
- Quota-aware run: up to **45 epochs**, early stopping patience **10**. Best checkpoint at **epoch 24**; training stopped around **epoch 34**.
- Wired into the API as the **default** model (`model_id`: `monai_best30h`).
- Overlay, volume, confidence, and multi-class output all enabled.

**Locked-test numbers (n = 29, classes 1–5)**

| Metric | Value |
|---|---|
| Macro Dice | **0.257375** (shown as 0.257) |
| Micro Dice | 0.385548 |
| Mean volume error | **19.533 mL** (shown as 19.5 mL) |
| Median volume error | 4.909 mL |
| Validation Dice at best checkpoint | 0.351169 |
| Mean study confidence | 0.8107 |
| Typical CPU time | **~19–22 seconds** per scan |

**Limits (kept visible, not hidden):** EDH Dice is about **0.011** on the locked test (almost no skill on epidural bleeds). Confidence is uncalibrated. 2.5D context only.

---

### 2. nnU-Net — Version 1 research comparator

**In plain language:** A slower, stronger 3D model. It sees the **whole volume**. This is the “serious” baseline to beat. It is the **best at volume** (smallest average milliliter error).

**Work completed**

- Converted labeled CTs into nnU-Net Dataset **501** (`Dataset501_BHSD`), 3D full-resolution, **fold 0**.
- Trained **196 epochs** (best validation EMA pseudo-Dice **0.4791** after epoch **184**).
- Held the test split out of training (`nnunet_splits_final.json`).
- Served through the same API (`model_id`: `nnunet_fold0`).
- Docker shared memory raised to **2 GB** so nnU-Net workers do not crash.

**Locked-test numbers (n = 29, classes 1–5)**

| Metric | Value |
|---|---|
| Macro Dice | **0.454723** (shown as 0.455) |
| Micro Dice | 0.551281 |
| Mean volume error | **14.3178 mL** (**best** of the four models) |
| Median volume error | 3.7331 mL |
| Mean study confidence | 0.9936 |
| Typical CPU time | **~327.64 seconds** per scan (~5+ minutes) |

**Limits:** EDH Dice about **0.207**. Slow on CPU. Confidence uncalibrated.

---

### 3. nnU-Net Ensemble — two folds averaged (strongest overlap)

**In plain language:** Instead of one nnU-Net, we keep **two** independently trained copies (fold 0 and fold 1) and **average** their answers. That improved overlap with expert outlines more than any single model we serve.

**Work completed**

- Trained 5-fold splits (`nnunet_splits_5fold.json`, fingerprint `4714804f9f1cabd9`). Locked test never entered those folds. Training used **163** labeled train+val cases.
- Served **folds 0 and 1** as a softmax ensemble (`model_id`: `nnunet_ssl_2fold`).
- Published locked-test scores in the UI (not training scores).
- Website card renamed from the old “SSL nnU-Net” name to **nnU-Net Ensemble** so it is not confused with the later semi-supervised model.

**Locked-test numbers (n = 29, classes 1–5)**

| Metric | Value |
|---|---|
| Ensemble macro Dice | **0.488552** (shown as **0.489**) — **best overlap we report** |
| Micro Dice | 0.544718 |
| Mean IoU (classes 1–5) | **0.342806** (shown as **0.343**) |
| Mean volume error | **18.5048 mL** (shown as 18.5 mL) |
| Median volume error | 4.4590 mL |
| Fold 0 alone (epoch 510) | Dice 0.480201 |
| Fold 1 alone (epoch 511) | Dice **0.496525** (best single fold; we still serve the 2-fold average) |
| Training length | up to **511** epochs on the folds used |
| Typical CPU time | about **2×** V1 nnU-Net |

**Limits:** Mean volume error is **worse** than V1 nnU-Net (18.5 vs 14.3 mL). Only 2 of 5 folds are served. Weights are in the v1.1.0 tarball.

---

### 4. SSL — semi-supervised nnU-Net (labeled + unlabeled)

**In plain language:** Extra CTs had **no expert outline**. A teacher model drew outlines for those. Then one nnU-Net was trained on **163 expert labels + 800 teacher masks**. That is what **SSL** means here: semi-supervised learning. It is a separate experiment from the Ensemble.

**Work completed**

- Built Dataset **502** (`BHSD_semi`): plans and `dataset.json` live in `config/models/nnunet_dataset502_semi/` and are copied next to the checkpoint at Docker start.
- Trained 3D full-resolution **fold 0** for **438** epochs.
- Wired as the fourth model (`model_id`: `nnunet_dataset502_semi`). The website card is named **SSL**.
- Documented clearly: this is **not** an upgrade over the Ensemble’s 0.489 locked-test Dice.
- Notebook `notebooks/02_ssl_preprocessing.ipynb` shows the same HU window and explains SSL vs Ensemble.

**Numbers (keep both; they are different)**

| Metric | Value | What it is |
|---|---|---|
| Displayed Dice **0.471** | 0.471114 | Best-checkpoint **train EMA** (what the UI shows) |
| Locked-test macro Dice | **0.401539** (~0.402) | Honest test on n = 29. **Weaker** than V1 (0.455) and Ensemble (0.489) |
| IoU | not scored | — |
| Volume error | not scored | — |
| Best checkpoint epoch | 438 | — |
| CPU time | similar to V1 nnU-Net (~5+ min) | — |

**Locked-test Dice by bleed type (SSL only)**

| EDH | SDH | SAH | IPH | IVH |
|---|---|---|---|---|
| 0.108 | 0.712 | 0.520 | 0.395 | 0.273 |

**Limits:** Do not present 0.471 as better than Ensemble 0.489. Weights download automatically on first Docker start (GitHub Release **v1.2.0**).

---

## Side-by-side (same table the product uses)

Locked-test metric unless noted. Macro Dice, classes 1–5, n = 29.

| Model | Role | Dice | IoU | Volume error | Typical wait (CPU) |
|---|---|---|---|---|---|
| MONAI | Fast interactive demo | 0.257 locked test | 0.163 | 19.5 mL | ~20 s |
| nnU-Net | V1 3D comparator | 0.455 locked test | 0.313 | **14.3 mL** | ~5+ min |
| nnU-Net Ensemble | Two folds averaged | **0.489** | **0.343** | 18.5 mL | ~2× nnU-Net |
| SSL | Semi-supervised learning | **0.471** displayed | — | — | ~5+ min |

How to read this for a non-technical stakeholder:

- Want a **live demo that returns quickly** → MONAI.  
- Want the **closest milliliter estimate** → nnU-Net.  
- Want the **closest outline to the expert** on the locked test → nnU-Net Ensemble.  
- Want to **show the unlabeled-data experiment** → SSL, with the caveat that locked-test Dice is 0.402 and 0.471 is the training EMA.

---

## Shared engineering (not a fifth model, but part of the delivery)

These pieces sit under all four models and are in this repo:

| Piece | Location / notes |
|---|---|
| Preprocessing | `config/preprocessing.yaml`, `src/preprocessing/` |
| Dataset loading and volume cache | `src/dataset/` |
| MONAI architecture | `src/models/monai_unet.py` |
| Unified inference | `src/deployment/inference/` (MONAI backend + nnU-Net backend + factory) |
| Evaluation (Dice, volume, plots, logging) | `src/evaluation/` |
| Overlay drawing | `src/inference/overlay.py` |
| Research “clinical” text (no advice) | `src/clinical/` — severity **not_configured** |
| Model and experiment registries | `config/models/registry.yaml`, `config/experiments/registry.yaml` |
| FastAPI v1 | `src/api/v1/` |
| Next.js workspace | `web/` |
| Docker | `Dockerfile.api`, `Dockerfile.web`, `docker-compose.yml`, `docker/api-entrypoint.sh` |
| Label preprocessing notebook | `notebooks/01_label_preprocessing.ipynb` |
| SSL preprocessing notebook | `notebooks/02_ssl_preprocessing.ipynb` |
| Demo scans + labels | `data/demo/` (17 CTs; see `data/demo/MANIFEST.csv`) |

**How to run the delivered demo**

```powershell
git clone --branch product_final https://github.com/GuptaFalguni/BrainHemorrhageAI.git
cd BrainHemorrhageAI
docker compose up --build
```

Then open **http://localhost:3000**. Example file: `data/demo/01_all_classes/all_classes.nii.gz`.

---

## Price breakdown — ₹18,000 total, billed in stages

All amounts are **INR**. Total **₹18,000**. Stages follow the actual work packages. GST/TDS, if any, are extra and not included here.

| Stage | What was delivered | Amount | Suggested invoice trigger | Status |
|---|---|---|---|---|
| **1 — Foundation** | Project setup, BHSD label map (EDH/SDH/SAH/IPH/IVH), locked HU window, train/val/test protocol, preprocessing pipeline | **₹3,000** | Kickoff / advance | Done |
| **2 — MONAI model** | 2.5D U-Net training, locked-test scoring, API serving, ~20 s CPU demo path | **₹3,000** | MONAI numbers frozen and runnable | Done |
| **3 — nnU-Net V1** | Dataset501 conversion, 3D full-res fold 0 training, locked-test scoring, serving, Docker memory fix for nnU-Net | **₹4,000** | V1 numbers frozen and runnable | Done |
| **4 — Ensemble + SSL** | 2-fold ensemble (Dice 0.489), Dataset502 semi-supervised model, UI naming so Ensemble and SSL are not confused | **₹5,000** | Both extra models scored and wired | Done |
| **5 — Product demo** | Website, REST API, Docker one-command demo, sample CTs, two notebooks, README, this summary | **₹3,000** | `product_final` branch handed over | Done |
| | **Total** | **₹18,000** | | |

### If you prefer three invoices instead of five

Same total, fewer payments:

| Invoice | Covers stages | Amount | Share |
|---|---|---|---|
| **A — Advance** | Stage 1 | **₹3,000** | 17% |
| **B — Models** | Stages 2 + 3 + 4 | **₹12,000** | 67% |
| **C — Delivery** | Stage 5 | **₹3,000** | 17% |
| | **Total** | **₹18,000** | 100% |

Rounded check: 3,000 + 3,000 + 4,000 + 5,000 + 3,000 = **18,000**.  
Alternate three-invoice check: 3,000 + 12,000 + 3,000 = **18,000**.

### What is not in this ₹18,000

- Hospital deployment, regulatory / CDSCO / FDA work, or turning this into a medical device.  
- Labeling new hospital scans.  
- Training on a new dataset.  
- Filling in severity thresholds or treatment recommendations.  
- Cloud GPU rental (training compute, if paid separately, is outside this fee).  
- Ongoing hosting or maintenance after handover.

---

## Honest caveats (please keep these when presenting the work)

1. Research software only.  
2. Ensemble wins **overlap** (Dice 0.489). V1 nnU-Net wins **volume error** (14.3 mL). There is no single “best at everything” model.  
3. SSL displayed **0.471** is training EMA; locked-test Dice is **0.402**. Do not market SSL as better than Ensemble.  
4. EDH remains weak, especially for MONAI.  
5. Severity is not configured. Confidence is not a diagnosis probability.  
6. Demo CTs are a small public research subset, not a clinical archive.

---

## Sources of the numbers in this document

- Model cards: `config/models/registry.yaml`  
- Experiment metrics: `config/experiments/registry.yaml`  
- Preprocessing window: `config/preprocessing.yaml`  
- Product README: `README.md`  
- Dataset502 SSL config: `config/experiments/nnunet_dataset502_semi.yaml`  
- Ensemble serving config: `config/experiments/nnunet_ssl_2fold.yaml`  

Paper and dataset: Wu et al., BHSD, [arXiv:2308.11298](https://arxiv.org/abs/2308.11298); dataset at [github.com/White65534/BHSD](https://github.com/White65534/BHSD) and [Hugging Face WuBiao/BHSD](https://huggingface.co/datasets/WuBiao/BHSD). nnU-Net: Isensee et al., Nature Methods 2021.
