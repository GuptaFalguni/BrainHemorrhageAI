# Frontend SPA Workspace Design — Single-Page Medical AI Console

**Status:** Canonical product UI — implemented in `web/`  
**Date:** 2026-07-15  
**Product notice:** Research Software — Not for Clinical Use  
**Backend (frozen):** [`api_v1_contract.md`](api_v1_contract.md)  
**Supersedes (UI IA):** Multi-page dashboard / sidebar IA (removed)  
**Keeps:** Design tokens (teal primary), research disclaimer, API-only orchestration, capability honesty  

---

## 1. Decision

| Rejected | Adopted |
|----------|---------|
| Dashboard with Analyze / Runs / Models / Compare / Docs pages | **One continuous scroll page** |
| Sidebar navigation as primary IA | **Minimal chrome** (brand + status + theme) |
| Separate compare route | **Inline “Compare Both”** mode on the same page |
| Information density of admin panels | **Apple-like calm** — large cards, one job per section |

**Principle:** A non-technical user should never wonder “which page?” The mental model is:

> Upload → Choose model(s) → Analyze → See results → Download

Everything appears **in place** as the run progresses.

---

## 2. Design language

### Mood

Premium clinical instrument, closer to a modern radiology product demo than a SaaS admin console.

| Attribute | Choice |
|-----------|--------|
| Theme | **Dark default** (near-black canvas) |
| Surfaces | Large frosted cards — soft glass (`backdrop-blur`, translucent fill, hairline border) |
| Typography | Display for hero brand only; clean UI sans for body; mono for numbers |
| Color | Cool neutrals + single teal accent; semantic green/amber/red sparingly |
| Motion | Soft fade/slide; section reveal; progress pulse — no glow spam |
| Density | Generous padding; one primary CTA per section |
| Clutter | No sidebar nav; no breadcrumb maze; docs/github as footer links only |

### Why dark + glass

CT overlays and medical demos read better on dark stages. Glass cards separate “control deck” from the void without heavy chrome. Matches thesis-demo polish without purple-AI aesthetics.

### Accessibility (non-negotiable)

- Tooltips + expandable “What is this?” cards for every metric  
- Keyboard focus rings on all controls  
- Compare layout stacks on mobile (not side-by-side forever)  
- Disclaimer always visible (sticky thin bar or first/last section)  

---

## 3. UX flow (single page)

```text
[Idle]
  Hero + Disclaimer
  Upload empty
  Model radios idle
  Analyze disabled until file + model chosen

[Ready]
  File chip shown
  Model selected (MONAI | nnU-Net | Compare Both)
  Analyze enabled

[Running]
  Page soft-locks to Progress section (auto-scroll)
  Progress live (elapsed + copy; no fake %)
  Results sections hidden/skeleton

[Complete — Single model]
  Auto-scroll to Results
  Overlay viewer (1)
  Clinical summary
  Volumes
  Downloads
  Beginner metric explainers

[Complete — Compare Both]
  Sequential or parallel predicts (product: sequential via API v1 sync)
  Side-by-side overlays
  Side-by-side metrics
  Explicit comparison strip (who better on what — from returned numbers only)
  Dual downloads

[Error]
  Inline error card under Progress
  Keep upload + model selection intact
  Retry CTA
```

### Compare Both (API reality)

API v1 is **sync, one model_id per POST**. UX still shows one “Analyze” action; under the hood:

1. `POST` with `monai_best30h`  
2. Then `POST` with `nnunet_fold0` (long wait — honest progress: “Model 2 of 2”)  
3. Bind both `prediction_id`s into side-by-side panels  

No fake simultaneous GPU fantasy in the UI.

---

## 4. Page structure (top → bottom)

Sections are **anchors on one page**, revealed progressively.

| # | Section ID | Content |
|---|------------|---------|
| 0 | `chrome` | Brand mark, API status pill, theme toggle (optional), no nav links |
| 1 | `hero` | Brand-forward title, one sentence, research posture |
| 2 | `upload` | Large dropzone for `.nii` / `.nii.gz` |
| 3 | `models` | Three big selectable cards: MONAI / nnU-Net / Compare Both |
| 4 | `analyze` | Single primary Analyze button |
| 5 | `progress` | Live elapsed, stage text, cancel (client abort only) |
| 6 | `results` | Prediction outcome cards (1 or 2 columns) |
| 7 | `overlays` | Overlay viewers side-by-side (or stacked) |
| 8 | `clinical` | Plain-language clinical summary from API |
| 9 | `volumes` | Total + per-class volumes with explainers |
| 10 | `comparison` | Only if Compare Both — delta cards |
| 11 | `downloads` | Report / mask / overlay / CSV buttons |
| 12 | `learn` | Expandable beginner cards for every metric |
| 13 | `disclaimer` | Full research disclaimer (also mirrored in chrome) |

---

## 5. ASCII wireframes

### 5.1 Desktop — idle

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  BH  BrainHemorrhageAI          ● API OK · CPU          ◐              │  chrome
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│              BrainHemorrhageAI                                           │  hero
│         Research CT hemorrhage segmentation                              │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│  RESEARCH SOFTWARE — Not for clinical diagnosis or treatment             │  disclaimer strip
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │  upload
│   │          Drop CT scan here  (.nii / .nii.gz)                     │   │
│   │                    [ Browse ]                                    │   │
│   │                                                                  │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │  models
│   │   MONAI      │  │   nnU-Net    │  │ Compare Both │                   │
│   │ Interactive  │  │  Research   │  │ Side-by-side │                   │
│   │ ~20s · Dice… │  │ minutes·…   │  │ Longer wait  │                   │
│   └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                          │
│                     [  Analyze scan  ]                                   │  analyze
│                                                                          │
│   (progress / results / overlays / … collapsed or dimmed until run)      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Desktop — progress

```text
│   Analyzing with MONAI…                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  ●●●●○○○○  Elapsed 00:18                                         │   │
│   │  Running interactive model. Usually ~20 seconds on CPU.          │   │
│   └──────────────────────────────────────────────────────────────────┘   │
```

### 5.3 Desktop — compare complete (results + overlays)

```text
│  Results                                                                 │
│  ┌─────────────────────────┐   ┌─────────────────────────┐               │
│  │ MONAI                   │   │ nnU-Net                 │               │
│  │ Hemorrhage: Yes         │   │ Hemorrhage: Yes         │               │
│  │ Volume 14.1 mL  ⓘ       │   │ Volume 12.0 mL  ⓘ       │               │
│  │ Confidence 0.81 ⓘ       │   │ Confidence 0.99 ⓘ       │               │
│  └─────────────────────────┘   └─────────────────────────┘               │
│                                                                          │
│  Overlays                                                                │
│  ┌─────────────────────────┐   ┌─────────────────────────┐               │
│  │   [ axial overlay ]     │   │   [ axial overlay ]     │               │
│  │   slice · opacity       │   │   slice · opacity       │               │
│  └─────────────────────────┘   └─────────────────────────┘               │
│                                                                          │
│  Comparison                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ Total volume Δ   ·   Subtypes agree/differ   ·   Latency          │    │
│  │ (numbers from this run only — not locked-test Dice)               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  [ Download MONAI report ]  [ Download nnU-Net report ]  [ CSV… ]        │
```

### 5.4 Mobile

```text
┌─────────────────────┐
│ Brand · API · theme │
│ Hero                │
│ Disclaimer          │
│ Upload card         │
│ Model cards stack   │
│ Analyze             │
│ Progress            │
│ Result card 1       │
│ Overlay 1           │
│ Result card 2       │
│ Overlay 2           │
│ Clinical            │
│ Volumes             │
│ Comparison          │
│ Downloads           │
│ Learn more          │
└─────────────────────┘
```

Side-by-side becomes **vertical stack** with sticky “MONAI / nnU-Net” segment control to jump between columns.

---

## 6. Component hierarchy

```text
WorkspacePage (single route `/`)
├── WorkspaceChrome
│   ├── BrandMark
│   ├── ApiStatusPill          ← GET /health
│   └── ThemeToggle
├── HeroSection
├── DisclaimerBanner           (sticky optional)
├── UploadSection
│   └── ScanDropzone
├── ModelSelectSection
│   ├── ModelChoiceCard (MONAI)
│   ├── ModelChoiceCard (nnU-Net)
│   └── ModelChoiceCard (Compare Both)
├── AnalyzeSection
│   └── AnalyzeButton
├── ProgressSection
│   └── SyncProgressCard
├── ResultsSection
│   └── PredictionResultCard[]   (1 or 2)
├── OverlaySection
│   └── OverlayViewerCard[]      (static PNG F2; Cornerstone later)
├── ClinicalSummarySection
├── VolumeAnalysisSection
│   ├── VolumeTotalCard
│   ├── VolumeBySubtypeTable
│   └── MetricExplainer (shared)
├── ComparisonSection            (Compare Both only)
│   └── ComparisonDeltaGrid
├── DownloadsSection
│   └── ArtifactDownloadGroup[]
├── LearnSection
│   └── ExplainerAccordionItem[]  (Dice context, confidence, volume, severity…)
└── FooterDisclaimer
```

### Key component contracts (design, not code)

| Component | Props (conceptual) | State owner |
|-----------|--------------------|-------------|
| `ScanDropzone` | file, onFile, error | local draft |
| `ModelChoiceCard` | model metadata, selected, mode | Zustand workspace |
| `AnalyzeButton` | disabled, loading, onClick | mutation |
| `SyncProgressCard` | mode, step 1\|2, elapsed, model label | mutation + timer |
| `PredictionResultCard` | PredictResponse subset + explainers | Query/cache by prediction_id |
| `OverlayViewerCard` | overlay URL, label | artifacts list |
| `MetricExplainer` | termId, short, long | static content |
| `ComparisonDeltaGrid` | left/right volumes & clinical | derived |

---

## 7. Revised frontend architecture

### 7.1 Routing

| Route | Role |
|-------|------|
| `/` | **Only product surface** — full workspace |
| `/about` (optional) | Thin legal mirror — or footer modal instead |
| Remove as primary IA | `/analyze`, `/runs`, `/models`, `/compare`, `/experiments`, `/docs` |

F1 multi-page shell becomes a **temporary scaffold**: implementation phase collapses AppShell to chrome-only and puts the workspace on `/`.

### 7.2 Folder structure (revised target)

```text
web/
  app/
    page.tsx                 # WorkspacePage only
    layout.tsx               # minimal chrome + providers
    globals.css
  components/
    chrome/
    workspace/               # all section components
    explainer/
    ui/                      # primitives
  features/
    workspace/               # orchestration hooks
    predict/                 # mutations (single + compare sequence)
  stores/
    workspace-store.ts       # file handle meta, mode, UI section focus
    session-store.ts
  lib/api/                   # health, models, predict, report, artifacts
  content/
    explainers.ts            # beginner metric copy
```

### 7.3 State

| Concern | Tool |
|---------|------|
| health / models | TanStack Query |
| predict (1 or 2) | TanStack Mutation(s); store both results in workspace store |
| selected mode | Zustand `workspace-store` (`monai` \| `nnunet` \| `compare`) |
| upload draft | React state (File) |
| section visibility | derived from run phase |

### 7.4 API usage (unchanged backend)

| UI need | API |
|---------|-----|
| Status pill | `GET /health` |
| Model cards | `GET /models` |
| Analyze | `POST /predict` (×1 or ×2) |
| Overlays | `GET /artifacts/{id}/overlay.png` (+ `input_ct` later for Cornerstone) |
| Report download | `GET /report/{id}` |
| Volumes/clinical | From PredictResponse (no recompute) |

### 7.5 Progressive enhancement

| Phase | Overlay fidelity |
|-------|------------------|
| Next implementation | Static `overlay.png` (fast thesis demo) |
| Later | Cornerstone with `input_ct.nii.gz` + mask |

Do not block SPA redesign on Cornerstone.

### 7.6 Beginner education system

Every metric shows:

1. **Value** (large)  
2. **ⓘ tooltip** — one sentence  
3. **Expand** — 2–4 sentences, “what it is / what it is not”  

Explainers (content IDs):

- `total_volume_ml`  
- `per_class_volume`  
- `study_confidence` (stress: **not** calibrated clinical probability)  
- `severity_status` (`not_configured` is normal)  
- `macro_dice` (registry context on model cards — **not** this-scan accuracy)  
- `interactive_vs_research`  
- `compare_delta_volume`  

Copy must never invent clinical advice.

---

## 8. Interaction details

### Analyze button

- Disabled until file present AND mode selected  
- Compare Both: confirm sheet — “This runs two models. nnU-Net may take several minutes on CPU.”  

### Progress

- Single model: one elapsed timer  
- Compare: “Step 1/2 MONAI…” → “Step 2/2 nnU-Net…”  
- No percentage unless backend later provides real progress (it does not today)  

### Results reveal

- Framer Motion: fade-up sections as data arrives  
- Prefer auto-scroll to `progress`, then to `results` on success  

### Downloads

- Per model group: clinical report `.md`, overlay `.png`, mask `.nii.gz`, volume CSV  
- Compare: two groups, visually paired  

---

## 9. Mapping from rejected dashboard

| Old page | New location |
|----------|--------------|
| Home / Analyze | Upload + Analyze on `/` |
| Models | Model choice cards (from `/models` API) |
| Compare | Compare Both mode + comparison section |
| Runs | Optional “Recent on this device” **collapsed** footer strip (local only) — not a page |
| Docs / Experiments | Footer links or Learn accordion — not separate apps |
| About | Footer / disclaimer |

---

## 10. Implementation phases (revised — after this design is approved)

| Phase | Deliverable |
|-------|-------------|
| **S1** | Collapse to single `/` workspace chrome + hero/upload/models/analyze UI (no predict yet) |
| **S2** | Predict + progress + single-model results/clinical/volumes/downloads + explainers |
| **S3** | Compare Both sequence + side-by-side overlays/metrics |
| **S4** | Motion polish, mobile stacking, a11y pass |
| **S5** | Optional Cornerstone (later) |

**Do not start S1 until this document is explicitly accepted.**

---

## 11. Success criteria

- User completes a demo without leaving `/`  
- A non-engineer understands confidence and volume via explainers  
- Compare Both is obvious and honest about wait time  
- Research disclaimer is impossible to miss  
- UI feels calm, dark, glass, and presentation-ready for a thesis  

---

## 12. Non-goals (still)

- Multi-page product IA  
- Async job queues / websockets  
- PDF generation  
- Invented severity tiers  
- Clinical claims  

---

*End of SPA Workspace Design — design only; no code in this phase.*
