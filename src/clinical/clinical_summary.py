"""Human-readable clinical summary (research software only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clinical.severity_engine import RESEARCH_DISCLAIMER, SeverityResult
from deployment.inference.base import UnifiedInferenceResult
from evaluation.volume_metrics import format_subtype_name


@dataclass
class ClinicalSummary:
    """Structured findings text derived from inference + severity."""

    hemorrhage_detected: bool
    subtypes: list[str]
    total_volume_ml: float
    study_confidence: float | None
    severity_status: str
    severity_tier: str | None
    recommendations_available: bool
    limitations: list[str]
    disclaimer: str
    narrative: str
    findings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hemorrhage_detected": self.hemorrhage_detected,
            "subtypes": list(self.subtypes),
            "total_volume_ml": self.total_volume_ml,
            "study_confidence": self.study_confidence,
            "severity_status": self.severity_status,
            "severity_tier": self.severity_tier,
            "recommendations_available": self.recommendations_available,
            "limitations": list(self.limitations),
            "disclaimer": self.disclaimer,
            "narrative": self.narrative,
            "findings": dict(self.findings),
        }


def build_clinical_summary(
    inference: UnifiedInferenceResult,
    severity: SeverityResult,
) -> ClinicalSummary:
    """Build a research-only findings summary. Does not give medical advice."""
    volumes = inference.volumes or {}
    confidence = inference.confidence or {}
    total_volume = float(volumes.get("total_volume_ml", severity.total_volume_ml))
    classes_present = [int(v) for v in volumes.get("classes_present", [])]
    subtypes = [format_subtype_name(label) for label in classes_present]
    if not subtypes and severity.subtype_results:
        subtypes = [item.subtype for item in severity.subtype_results]

    study_conf_raw = confidence.get("study_confidence")
    study_confidence = float(study_conf_raw) if study_conf_raw is not None else None
    recommendations_available = bool(severity.recommendations)
    limitations = list(severity.limitations) or [
        "Research software only; not a medical device.",
        "Not for clinical diagnosis, triage, or treatment decisions.",
    ]
    disclaimer = severity.disclaimer or RESEARCH_DISCLAIMER

    lines = [
        "Research Software — Not for Clinical Use",
        "",
        "Hemorrhage detected: "
        + ("Yes (model estimate)" if severity.hemorrhage_detected else "No (model estimate)"),
        f"Subtype(s): {', '.join(subtypes) if subtypes else 'None'}",
        f"Estimated total volume: {total_volume:.4f} mL",
    ]
    if study_confidence is not None:
        lines.append(
            f"Model study confidence: {study_confidence:.4f} "
            "(softmax statistic; not a clinical probability)"
        )
    lines.append(f"Severity status: {severity.status}")
    if severity.tier:
        lines.append(f"Severity tier: {severity.tier}")
    else:
        lines.append("Severity tier: unavailable")
    lines.append(
        "Recommendations available: "
        + ("Yes (configuration-provided statements only)" if recommendations_available else "No")
    )
    lines.append("Limitations:")
    for item in limitations:
        lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            disclaimer,
            "This summary does not provide diagnosis or treatment advice.",
        ]
    )

    return ClinicalSummary(
        hemorrhage_detected=severity.hemorrhage_detected,
        subtypes=subtypes,
        total_volume_ml=total_volume,
        study_confidence=study_confidence,
        severity_status=severity.status,
        severity_tier=severity.tier,
        recommendations_available=recommendations_available,
        limitations=limitations,
        disclaimer=disclaimer,
        narrative="\n".join(lines),
        findings={
            "model_id": inference.model_id,
            "framework": inference.framework,
            "per_class_volume_ml": volumes.get("per_class_volume_ml", {}),
            "per_class_confidence": confidence.get("per_class_confidence", {}),
            "severity_factors": list(severity.factors),
        },
    )
