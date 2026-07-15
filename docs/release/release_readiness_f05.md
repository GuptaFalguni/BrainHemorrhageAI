# Release Readiness Report — F0.5 Documentation Freeze

**Date:** 2026-07-15  
**Scope:** Repository consistency after Phase C.4 (API) and F0 (frontend architecture docs); **before** Next.js implementation.

Entry: [`docs/PROJECT_MASTER_GUIDE.md`](../PROJECT_MASTER_GUIDE.md)

---

## Scores (/10)

| Dimension | Score | Notes |
|-----------|------:|-------|
| Repository structure | 8.5 | Clear `src/` packages; Streamlit vs planned `web/` needs discipline |
| Documentation | 8.5 | F0.5 pass; master guide + API contract SSoT |
| Architecture | 9.0 | Science + deployment designs complete |
| Backend | 8.5 | API v1 solid; legacy API still present |
| Inference | 9.0 | Unified engine validates both backends |
| Deployment | 9.0 | Registries + clinical + artifacts |
| Frontend readiness | 6.5 | Blueprint frozen; **zero** production UI code; CT API gap mitigated in design |
| Research reproducibility | 8.5 | Locked reports + validators; heavy eval not one-click |
| Publication readiness | 8.0 | Tables exist; model cards aligned in F0.5 |
| Maintainability | 8.0 | Some dual APIs / dual UIs until cutover |
| **Overall** | **8.4** | Ready for frontend **implementation gate**, not production clinical release |

---

## Folder organization audit

| Folder | Verdict |
|--------|---------|
| `config/` | Correct — single place for YAML truth |
| `docs/` | Correct — master guide + indexed domains |
| `reports/` | Correct — evaluation vs `api_v1` vs legacy `api/` |
| `src/` | Correct package layout |
| `frontend/` | **Interim Streamlit only** — do not place Next.js here |
| `web/` | **Planned** production frontend root (create at F1) |
| `notebooks/` | Exploratory OK; not authority |
| `checkpoints/` | Frozen weights — do not relocate |

**Improvements suggested (no moves in F0.5):** keep legacy `reports/api/` until Streamlit deprecation; add `web/` only when F1 starts.

---

## Code architecture audit (no rewrite)

| Topic | Finding | Action in F0.5 |
|-------|---------|----------------|
| Duplicate registry logic | Single loaders in `deployment/` | None |
| Duplicate inference | MONAI `src/inference` wrapped, not copied, by deployment | None |
| Duplicate clinical markdown | Clinical report ≠ prediction report (intentional) | None |
| Dual API apps | `api/app.py` legacy + `api/v1` | Document only; do not delete yet (Streamlit may depend) |
| `deployment/interfaces.py` stub vs `inference/base.py` | Thin historical stub | Leave frozen; clients use inference base |
| DTO duplication | Pydantic in API only; Python dataclasses upstream | Acceptable |
| YAML loading | PyYAML in registries/clinical | Fine |

**No code refactors performed** — none were both safe and necessary without touching frozen layers or breaking Streamlit.

---

## Validation expectations (operators)

Re-run after doc freeze (docs do not change runtime):

| Check | Command |
|-------|---------|
| Registry | `python src/deployment/validate_registry.py` |
| Clinical | `python src/clinical/validate_clinical_pipeline.py` |
| API v1 | `python src/api/v1/validate_api_v1.py` |

---

## Blockers before frontend implementation

1. F0.5 accepted; UI canonical is [`frontend_spa_workspace_design.md`](../platform/frontend_spa_workspace_design.md) (`web/`).  
2. Package root is **`web/`** (not Streamlit `frontend/`).  
3. CT preview: client-side NIfTI mid-slice + API overlay PNGs.

Non-blockers: severity `not_configured`, legacy API presence, Streamlit interim.
