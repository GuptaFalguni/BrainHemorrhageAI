# BrainHemorrhageAI — Release Candidate v1.0

**Version:** 1.0.0  
**Updated:** 2026-07-15 (F0.5 documentation freeze)  
**Supersedes:** 2026-07-11 pre–Phase C RC text  

**Master entry:** [`docs/PROJECT_MASTER_GUIDE.md`](../PROJECT_MASTER_GUIDE.md)  
**Readiness scores:** [`docs/release/release_readiness_f05.md`](release_readiness_f05.md)

This is an engineering release summary — not a research paper and not a clinical clearance.

---

## Project version

| Field | Value |
|-------|-------|
| Package | `brain-hemorrhage-ai` |
| Version | `1.0.0` |
| Python | ≥ 3.10 |
| License | MIT |
| Dataset | BHSD `label_192` |
| Locked split | 134 / 29 / 29 |

---

## Current capabilities (as implemented)

| Area | Status |
|------|--------|
| Preprocessing / training / evaluation | Frozen and validated |
| MONAI `monai_best30h` | Locked baseline; **default** interactive model |
| nnU-Net `nnunet_fold0` | Locked comparator; **research** model via API |
| Unified inference | `src/deployment/inference/` |
| Clinical pipeline | `src/clinical/` (severity `not_configured`) |
| **API v1** | `src/api/v1/` — see [`api_v1_contract.md`](../platform/api_v1_contract.md) |
| Legacy API | `src/api/app.py` — compatibility only |
| Interim UI | Streamlit `frontend/app.py` |
| Production frontend | **Delivered** — [`frontend_spa_workspace_design.md`](../platform/frontend_spa_workspace_design.md) · `web/` |

---

## Known limitations

- Research software; not for clinical use  
- Severity literature-gated (`not_configured`)  
- Uncalibrated confidence; weak EDH  
- nnU-Net slow on CPU  
- No auth / async jobs  
- Production Next.js app not implemented  
- Original CT not in API artifact whitelist  

---

## Deferred

- Next.js implementation (await F0.5 acceptance → F1+)  
- Auth, cloud hardening, async workers  
- PDF reports; severity numeric activation  
- Optional API exposure of input CT for durable viewers  

---

## Architecture freeze (science)

Unchanged: MONAI 2.5D U-Net design, BHSD preprocessing, locked split, DiceCE selection metric, locked-test policy.  
See `docs/architecture/`.

---

## Recommended next milestone

1. Accept F0.5 documentation freeze + frontend blueprint.  
2. Implement production frontend in `web/` per blueprint (not inside Streamlit tree).  
3. Keep Streamlit until Next.js cutover.  
4. Optionally expose input CT on API for Cornerstone durability.

---

## Checklist

| Check | Result |
|-------|--------|
| Science architecture frozen | Yes |
| Dual-model locked-test comparison | Yes |
| Deployment + clinical + API v1 | Yes |
| Documentation consistent with code (F0.5) | Yes |
| Production frontend implemented | **No** |
| Clinical deployment readiness | **No** |
