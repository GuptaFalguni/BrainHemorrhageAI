# BrainHemorrhageAI — GPU Training Protocol (Version 1.0)

**Status:** Governing protocol for reproducible GPU research  
**Repository version:** 1.0.0  
**Effective date:** 2026-07-11  
**Audience:** Operators running long GPU experiments (Kaggle or equivalent)  
**Related documents:**

- `docs/release/release_candidate_v1.md`
- `docs/research/experiment_protocol.md`
- `docs/architecture/training_pipeline_design.md`
- `docs/architecture/model_design.md`
- `config/kaggle_training.yaml`

This document does not change implementation. It defines how Version 1.0 must be trained, evaluated, logged, and compared.

---

## 1. Repository version

| Field | Value |
|-------|-------|
| Project | BrainHemorrhageAI |
| Package | `brain-hemorrhage-ai` |
| Version | **1.0.0** |
| License | MIT |
| Python | ≥ 3.10 |
| Install | `pip install -e .` |
| GPU training config | `config/kaggle_training.yaml` |
| Local training config | `config/training.yaml` (same recipe; CPU/auto device defaults) |

Version 1.0 is **frozen**. GPU runs must use this codebase as-is. Do not mix results from unfrozen or modified trees with Version 1.0 claims.

---

## 2. Architecture freeze statement

The following decisions are **frozen** for all Version 1.0 GPU experiments:

| Component | Frozen choice |
|-----------|---------------|
| Model | MONAI 2D U-Net (`config/model.yaml`) |
| Input | 2.5D axial stack (previous / current / next); edge replication |
| Channels | 3 → 6 (background + EDH, SDH, SAH, IPH, IVH) |
| Encoder | `channels: [32, 64, 128, 256]`, `strides: [2, 2, 2]` |
| Preprocessing | HU clip `[-40, 120]`; normalize `(HU − 40) / 80` |
| Spatial policy | Native resolution; **no** resampling in Version 1.0 |
| Loss | DiceCELoss (`include_background: true`, λ_dice = λ_ce = 1.0) |
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=1e-5`) |
| Scheduler | Cosine |
| Selection metric | Validation **macro Dice** over hemorrhage classes **1–5 only** |
| Test policy | Locked test split evaluated **once**, post-training only |
| Seed | `42` |

Any change to the above requires a **new version** (e.g. 1.1 or 2.0), not a silent edit under the Version 1.0 label.

---

## 3. Dataset version

| Property | Value |
|----------|-------|
| Dataset | BHSD — Brain Hemorrhage Segmentation Dataset (Wu et al., arXiv:2308.11298) |
| Subset | `label_192` only |
| Counts | 192 paired CT + multi-class masks |
| Classes | 0 = background; 1 = EDH; 2 = SDH; 3 = SAH; 4 = IPH; 5 = IVH |
| Format | NIfTI (`.nii.gz`) |
| Path | `data/raw/label_192/images/` and `data/raw/label_192/ground truths/` |
| Unlabeled set | `unlabel_1980` — **excluded** from Version 1.0 training |

Operators must use the same BHSD labeled release placed under the paths above. Do not substitute alternate datasets or remapped labels under Version 1.0.

---

## 4. Split version

| Property | Value |
|----------|-------|
| Manifest | `data/metadata/splits.csv` |
| Config record | `data/metadata/split_config.json` |
| Ratio | 70% / 15% / 15% |
| Counts | **134 train / 29 val / 29 test** |
| Seed | **42** |
| Method | Multi-label iterative stratification (`MultilabelStratifiedShuffleSplit`) |
| Generated | 2026-07-05T08:14:13.202749+00:00 (UTC) |

The split is **locked**. Regenerating or reshuffling splits invalidates Version 1.0 comparability. Always verify `splits.csv` hash / contents before claiming a Version 1.0 result.

---

## 5. Training protocol

### 5.1 Canonical GPU recipe

Use `config/kaggle_training.yaml` for all long GPU runs. Relative to `config/training.yaml`, **only** device and DataLoader I/O settings differ:

| Setting | Value |
|---------|-------|
| `device` | `cuda` |
| `num_workers` | `2` |
| `pin_memory` | `true` |
| `persistent_workers` | `true` |
| `mixed_precision` | `true` |
| `epochs` | `150` |
| `batch_size` | `4` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.00001` |
| `early_stopping_patience` | `20` |
| `gradient_clip` | `1.0` |
| `seed` | `42` |

### 5.2 Required procedure

1. Install the frozen repository (`pip install -e .`).
2. Confirm BHSD paths and locked split.
3. Run smoke validators (see §11).
4. Prefer a short GPU smoke (e.g. `experiment_001`, 5 epochs) before the 150-epoch run.
5. Launch the long run with the Kaggle training config and a unique experiment ID (§8).
6. Do not touch the test split during training or model selection.
7. Resume only from checkpoints produced by the **same** experiment ID and config family.

### 5.3 Sanity vs publishable baseline

| Run | Purpose | Publishable? |
|-----|---------|--------------|
| `experiment_sanity` (2 epochs) | Pipeline validation | **No** |
| Short GPU smoke (e.g. 5 epochs) | GPU health check | **No** (unless explicitly labeled as smoke) |
| Full 150-epoch Version 1.0 baseline | Primary baseline | **Yes**, if protocol followed |

---

## 6. Checkpoint naming

| Artifact | Default name | Location pattern |
|----------|--------------|------------------|
| Best model (selection metric) | `best_model.pt` | `checkpoints/<experiment_id>/` |
| Last epoch / resume state | `last_model.pt` | `checkpoints/<experiment_id>/` |
| Merged training config (when written by experiment runner) | `training_merged.yaml` | Same checkpoint directory |

**Rules:**

- Best checkpoint is selected by validation macro Dice (classes 1–5).
- Never overwrite another experiment’s checkpoint directory.
- Archive `best_model.pt` and `last_model.pt` with the experiment report folder.
- Inference and evaluation for Version 1.0 claims must use that experiment’s `best_model.pt` unless a protocol amendment says otherwise.

---

## 7. Evaluation protocol

1. Evaluate **only** the locked **test** split (`splits.csv` → `test`).
2. Evaluate **once** after training completes (or after early stopping), using `best_model.pt`.
3. Do not tune hyperparameters on the test set.
4. Required outputs (via existing evaluation / experiment pipeline):
   - Global and per-class Dice / IoU (and related segmentation metrics)
   - Micro Dice over hemorrhage classes **1–5** (not including background)
   - Native-space volume metrics (mL) and volume error vs ground truth
   - Confidence summaries (uncalibrated softmax; report as such)
   - Confusion matrix and standard plots under the experiment report tree
5. Record the checkpoint path, config paths, seed, and split identity in the experiment report.

---

## 8. Experiment naming convention

| Field | Convention | Example |
|-------|------------|---------|
| Experiment ID | `experiment_<nnn>` or descriptive frozen IDs | `experiment_001`, `experiment_sanity` |
| Training `run_id` | Same as experiment ID unless documenting a resume suffix | `experiment_001` |
| Evaluation `run_id` | `<experiment_id>_eval` | `experiment_001_eval` |
| Checkpoint dir | `checkpoints/<experiment_id>/` | `checkpoints/experiment_001/` |
| Experiment report dir | `reports/experiments/<experiment_id>/` | `reports/experiments/experiment_001/` |
| Index row | Append to `reports/experiments/index.csv` | One row per completed experiment |

**GPU long-run recommendation:** use a dedicated ID such as `experiment_v1_baseline_150` (or continue the numeric series) and keep that ID stable across resume attempts for the same logical run.

Do not reuse an experiment ID for a different config or code version.

---

## 9. Allowed modifications

The following are allowed **without** bumping the architecture version, provided they are logged:

| Allowed | Notes |
|---------|-------|
| Hardware / platform | Kaggle GPU, local CUDA, cloud GPU |
| Using `config/kaggle_training.yaml` vs `config/training.yaml` | I/O and `device` only; training recipe must match |
| Experiment ID, paths, `run_id` | Logging and organization |
| `resume: true` for the **same** experiment | Continue interrupted runs |
| Epoch count overrides for **smoke** runs | Must be labeled non-publishable |
| Environment package pins for install reproducibility | Record in environment snapshot |
| Documentation clarifying this protocol | Docs only |

Blocking bugfixes that restore intended frozen behavior (without changing methodology) may be applied only if documented as a Version 1.0.x patch with a clear changelog. Prefer not to change code during an active long run.

---

## 10. Forbidden modifications

Do **not** do any of the following under the Version 1.0 label:

- Change model architecture, channels, strides, or loss formulation
- Change preprocessing (clip bounds, normalization, 2.5D rules)
- Regenerate or alter `splits.csv` / stratification
- Train or select models using the test split
- Enable class weights, hemorrhage-aware sampling, or unlabeled data
- Add spatial resampling or change volume geometry policy
- Change selection metric away from validation macro Dice (classes 1–5)
- Quietly alter learning rate, optimizer, scheduler, or batch size for “better” results while still calling it Version 1.0 baseline
- Compare against Version 1.0 using a different split or metric definition
- Delete or rewrite prior experiment artifacts to hide failed runs

Deferred research (nnU-Net, resampling, semi-supervised, etc.) belongs in a **new versioned protocol**, not Version 1.0.

---

## 11. Kaggle workflow

1. **Upload** the frozen Version 1.0 repository (or clone the frozen tag/commit).
2. **Mount / place** BHSD under `data/raw/label_192/` with images and ground truths.
3. **Create environment** and install:
   ```bash
   pip install -e .
   ```
   Ensure CUDA-enabled PyTorch/MONAI appropriate for the Kaggle image.
4. **Validate** before long training:
   ```bash
   python src/dataset/validate_loader.py
   python src/evaluation/validate_evaluation.py
   python src/training/validate_training.py
   python src/evaluation/validate_evaluation_pipeline.py
   python src/experiments/validate_baseline_experiment.py
   ```
5. **GPU smoke** (recommended): short experiment with `training_config: config/kaggle_training.yaml`.
6. **Long run:** 150-epoch baseline using `config/kaggle_training.yaml` and a unique experiment ID (§8).
7. **Evaluate** locked test split once with `best_model.pt`.
8. **Archive** checkpoints, reports, configs, and environment snapshot (§12).
9. **Download** artifacts off Kaggle before the session expires.

Entry points (existing; do not invent new ones):

- `python src/training/train.py --training-config config/kaggle_training.yaml`
- `python src/experiments/run_baseline_experiment.py` (point experiment YAML `training_config` at `config/kaggle_training.yaml`)

---

## 12. Result logging protocol

For every GPU experiment, retain:

| Artifact | Purpose |
|----------|---------|
| Experiment report (`experiment.md` / tracker outputs) | Human-readable summary |
| `experiment_config.yaml` + merged training config | Exact settings |
| `environment` / dependency snapshot | Reproducibility |
| Training curves (loss, Dice, LR) | Training dynamics |
| `validation_metrics.csv` (or equivalent) | Epoch-wise selection evidence |
| Evaluation `metrics.json` + evaluation report | Test metrics |
| Plots (confusion matrix, per-class Dice, volume error, etc.) | Inspection |
| `best_model.pt` / `last_model.pt` | Weights |
| Row in `reports/experiments/index.csv` | Cross-experiment index |

**Logging rules:**

- Record seed, device, AMP, epoch reached, early-stopping status, and best validation macro Dice.
- Label smoke runs clearly so they are not mistaken for the publishable baseline.
- Never edit metrics files after the fact; if a correction is needed, add a dated note rather than silently rewriting numbers.

---

## 13. How future experiments will compare fairly with Version 1

Fair comparison to Version 1.0 requires **all** of the following:

1. **Same split:** `data/metadata/splits.csv` (134 / 29 / 29, seed 42).
2. **Same test policy:** one post-training evaluation on the locked test set.
3. **Same primary metrics:**
   - Validation / selection: macro Dice, classes 1–5
   - Test reporting: per-class and macro Dice (1–5), IoU, native-space volume errors (mL)
4. **Same preprocessing definition** unless the new work is an explicit preprocessing ablation (then Version 1.0 remains the reference arm).
5. **Same class taxonomy** (labels 0–5 as BHSD).
6. **Disclose** any change in architecture, loss, sampling, or compute; do not claim “Version 1.0 baseline” for a different recipe.
7. Prefer **paired** comparison on the identical test scans; future statistical tests (bootstrap CIs, paired tests) should use this locked test set.

Version 1.0 is the **reference baseline**. Later systems (nnU-Net, transformers, semi-supervised) should cite Version 1.0 metrics produced under this protocol as the comparison point.

---

## 14. Publication protocol

Before including numbers in a thesis, paper, or public claim:

1. Confirm the run used frozen Version **1.0.0** code and locked split.
2. Confirm the publishable run is the full baseline recipe (not sanity / smoke), or clearly label limited-epoch results as preliminary.
3. Report:
   - Dataset (BHSD `label_192`) and split counts
   - Model (MONAI 2D U-Net, 2.5D input)
   - Preprocessing (clip / normalize)
   - Training hyperparameters (epochs, LR, optimizer, loss, batch size, seed)
   - Selection metric and test protocol
   - Per-class and macro metrics; volume metrics in mL
4. State limitations: no resampling, uncalibrated confidence, no class weights, supervised `label_192` only.
5. Cite BHSD: Wu et al., arXiv:2308.11298.
6. Keep artifacts (§12) available for audit for the duration of the thesis / review process.

Do not publish sanity-experiment Dice as the Version 1.0 baseline result.

---

## 15. Professor presentation protocol

When presenting Version 1.0 progress to a supervisor or committee:

1. **Lead with freeze:** architecture, split, and metrics are locked; GPU work is execution, not redesign.
2. **Separate** pipeline proof (`experiment_sanity`) from the GPU baseline (in progress or complete).
3. Show:
   - Problem and BHSD setup (192 scans, 6 classes)
   - Locked split and why it enables fair comparison
   - 2.5D U-Net diagram and frozen training recipe
   - Selection metric (val macro Dice 1–5) and one-shot test evaluation
   - Current GPU status (smoke / full run / metrics)
   - Next milestone and explicitly deferred work (resampling, unlabeled data, stronger models)
4. Bring one experiment folder (plots + `metrics.json` + checkpoint path) as evidence.
5. If asked for changes mid-presentation: record them as **future version** proposals; do not alter Version 1.0 mid-run unless a blocking bug is confirmed.

---

## Document control

| Item | Value |
|------|-------|
| Document | `docs/training/gpu_training_protocol.md` |
| Applies to | BrainHemorrhageAI Version 1.0 GPU research |
| Supersedes | Ad-hoc Kaggle notes; does not replace `docs/research/experiment_protocol.md` |
| Amendment rule | Any methodology change requires a new versioned protocol section or a new project version |

**End of protocol.**
