"""Literature-gated severity classification from YAML rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from deployment.inference.base import UnifiedInferenceResult
from evaluation.volume_metrics import SUBTYPE_LABELS, format_subtype_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "clinical" / "severity_rules.yaml"

RESEARCH_DISCLAIMER = (
    "Research Software. Not for Clinical Use. Not for diagnosis, triage, or "
    "treatment decisions."
)


@dataclass
class SubtypeSeverity:
    """Per-subtype severity assessment."""

    subtype: str
    label_id: int
    volume_ml: float
    status: str
    tier: str | None = None
    matched_rule: str | None = None
    citation: str | None = None
    reason: str = ""


@dataclass
class SeverityResult:
    """Output of ``SeverityEngine`` for one inference result."""

    status: str
    tier: str | None
    hemorrhage_detected: bool
    total_volume_ml: float
    subtype_results: list[SubtypeSeverity]
    factors: list[str]
    citations: list[str]
    disclaimer: str
    recommendations: list[str]
    limitations: list[str]
    rules_version: int | None = None
    rules_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tier": self.tier,
            "hemorrhage_detected": self.hemorrhage_detected,
            "total_volume_ml": self.total_volume_ml,
            "subtype_results": [
                {
                    "subtype": item.subtype,
                    "label_id": item.label_id,
                    "volume_ml": item.volume_ml,
                    "status": item.status,
                    "tier": item.tier,
                    "matched_rule": item.matched_rule,
                    "citation": item.citation,
                    "reason": item.reason,
                }
                for item in self.subtype_results
            ],
            "factors": list(self.factors),
            "citations": list(self.citations),
            "disclaimer": self.disclaimer,
            "recommendations": list(self.recommendations),
            "limitations": list(self.limitations),
            "rules_version": self.rules_version,
            "rules_path": self.rules_path,
            "metadata": dict(self.metadata),
        }


def _as_float_map(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        out[int(key)] = float(value)
    return out


def _compare(op: str, left: float, right: float) -> bool:
    if op in {"ge", ">="}:
        return left >= right
    if op in {"gt", ">"}:
        return left > right
    if op in {"le", "<="}:
        return left <= right
    if op in {"lt", "<"}:
        return left < right
    if op in {"eq", "=="}:
        return left == right
    raise ValueError(f"Unsupported severity rule operator: {op!r}")


class SeverityEngine:
    """Apply ``severity_rules.yaml`` only — never invents thresholds."""

    def __init__(self, rules_path: Path | str | None = None) -> None:
        self._rules_path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
        self._config: dict[str, Any] | None = None

    @property
    def rules_path(self) -> Path:
        return self._rules_path

    def load_rules(self) -> dict[str, Any]:
        if self._config is not None:
            return self._config
        if not self._rules_path.is_file():
            raise FileNotFoundError(f"Severity rules not found: {self._rules_path}")
        with self._rules_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Severity rules must be a mapping: {self._rules_path}")
        self._config = loaded
        return loaded

    def reload(self) -> dict[str, Any]:
        self._config = None
        return self.load_rules()

    def evaluate(self, inference: UnifiedInferenceResult) -> SeverityResult:
        config = self.load_rules()
        volumes = inference.volumes or {}
        total_volume = float(volumes.get("total_volume_ml", 0.0))
        per_class = _as_float_map(volumes.get("per_class_volume_ml", {}))
        classes_present = [int(v) for v in volumes.get("classes_present", [])]
        if not classes_present:
            classes_present = [label for label, volume in per_class.items() if volume > 0]

        disclaimer = str(config.get("disclaimer") or RESEARCH_DISCLAIMER).strip()
        recommendations = [str(item) for item in (config.get("recommendations") or [])]
        limitations = [str(item) for item in (config.get("limitations") or [])]
        citations_global: list[str] = []
        global_status = str(config.get("status", "not_configured")).strip().lower()
        subtypes_cfg = config.get("subtypes") or {}
        if not isinstance(subtypes_cfg, dict):
            subtypes_cfg = {}

        hemorrhage_detected = total_volume > 0.0 or bool(classes_present)
        if not hemorrhage_detected:
            return SeverityResult(
                status="no_hemorrhage",
                tier=None,
                hemorrhage_detected=False,
                total_volume_ml=total_volume,
                subtype_results=[],
                factors=["No hemorrhage voxels detected by the segmentation model."],
                citations=[],
                disclaimer=disclaimer,
                recommendations=recommendations,
                limitations=limitations,
                rules_version=config.get("version"),
                rules_path=str(self._rules_path),
                metadata={"global_rules_status": global_status},
            )

        # Literature gate: unset YAML must never invent tiers.
        if global_status != "configured":
            subtype_results = [
                SubtypeSeverity(
                    subtype=format_subtype_name(label_id),
                    label_id=label_id,
                    volume_ml=float(per_class.get(label_id, 0.0)),
                    status="not_configured",
                    reason=(
                        "Global severity rules status is "
                        f"'{global_status}'; literature thresholds are not frozen."
                    ),
                )
                for label_id in sorted(set(classes_present) | {k for k, v in per_class.items() if v > 0})
                if label_id in SUBTYPE_LABELS
            ]
            return SeverityResult(
                status="not_configured",
                tier=None,
                hemorrhage_detected=True,
                total_volume_ml=total_volume,
                subtype_results=subtype_results,
                factors=[
                    "Severity classification is not_configured pending literature freeze.",
                    f"Literature gate: {config.get('literature_gate', 'unset')}",
                ],
                citations=[],
                disclaimer=disclaimer,
                recommendations=recommendations,
                limitations=limitations,
                rules_version=config.get("version"),
                rules_path=str(self._rules_path),
                metadata={"global_rules_status": global_status},
            )

        subtype_results: list[SubtypeSeverity] = []
        factors: list[str] = []
        any_missing = False
        tiers_found: list[str] = []

        labels_to_score = sorted(
            {
                label
                for label in set(classes_present) | {k for k, v in per_class.items() if v > 0}
                if label in SUBTYPE_LABELS
            }
        )

        for label_id in labels_to_score:
            name = format_subtype_name(label_id)
            volume_ml = float(per_class.get(label_id, 0.0))
            entry = subtypes_cfg.get(name) or subtypes_cfg.get(str(label_id)) or {}
            if not isinstance(entry, dict):
                entry = {}
            entry_status = str(entry.get("status", "not_configured")).strip().lower()
            rules = entry.get("rules") or []
            entry_citations = [str(c) for c in (entry.get("citations") or [])]

            if entry_status != "configured" or not rules:
                any_missing = True
                subtype_results.append(
                    SubtypeSeverity(
                        subtype=name,
                        label_id=label_id,
                        volume_ml=volume_ml,
                        status="not_configured",
                        reason="No approved literature rule configured for this subtype.",
                    )
                )
                factors.append(f"{name}: not_configured (missing approved rule).")
                continue

            matched: SubtypeSeverity | None = None
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                field_name = str(rule.get("field", "volume_ml"))
                if field_name != "volume_ml":
                    # Only volume_ml is supported until literature freeze expands the schema.
                    continue
                op = str(rule.get("op", "ge"))
                threshold = float(rule["value"])
                if _compare(op, volume_ml, threshold):
                    tier = str(rule.get("tier") or rule.get("name") or "matched")
                    citation = rule.get("citation")
                    if citation:
                        citation = str(citation)
                        citations_global.append(citation)
                    elif entry_citations:
                        citation = entry_citations[0]
                        citations_global.extend(entry_citations)
                    matched = SubtypeSeverity(
                        subtype=name,
                        label_id=label_id,
                        volume_ml=volume_ml,
                        status="configured",
                        tier=tier,
                        matched_rule=str(rule.get("name") or tier),
                        citation=citation,
                        reason=f"Matched YAML rule ({field_name} {op} {threshold}).",
                    )
                    tiers_found.append(tier)
                    factors.append(f"{name}: tier={tier} (volume={volume_ml:.4f} mL).")
                    break

            if matched is None:
                # Configured subtype with rules that did not match — still configured,
                # but no tier assigned from YAML.
                subtype_results.append(
                    SubtypeSeverity(
                        subtype=name,
                        label_id=label_id,
                        volume_ml=volume_ml,
                        status="configured",
                        tier=None,
                        reason="Subtype rules loaded; no YAML rule matched this volume.",
                    )
                )
                factors.append(f"{name}: configured rules present; no matching tier.")
            else:
                subtype_results.append(matched)

        if any_missing:
            overall_status = "not_configured"
            overall_tier = None
            factors.insert(
                0,
                "One or more detected subtypes lack approved literature rules.",
            )
        else:
            overall_status = "configured"
            overall_tier = tiers_found[0] if len(set(tiers_found)) == 1 else (
                "mixed" if tiers_found else None
            )

        # Deduplicate citations while preserving order.
        unique_citations: list[str] = []
        for cite in citations_global:
            if cite not in unique_citations:
                unique_citations.append(cite)

        return SeverityResult(
            status=overall_status,
            tier=overall_tier,
            hemorrhage_detected=True,
            total_volume_ml=total_volume,
            subtype_results=subtype_results,
            factors=factors,
            citations=unique_citations,
            disclaimer=disclaimer,
            recommendations=recommendations,
            limitations=limitations,
            rules_version=config.get("version"),
            rules_path=str(self._rules_path),
            metadata={"global_rules_status": global_status},
        )
