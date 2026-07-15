"""Clinical pipeline — post-segmentation orchestration only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clinical.clinical_report import ClinicalReport, write_clinical_report
from clinical.clinical_summary import ClinicalSummary, build_clinical_summary
from clinical.severity_engine import SeverityEngine, SeverityResult
from deployment.inference.base import UnifiedInferenceResult


@dataclass
class ClinicalPipelineResult:
    """Unified clinical outputs for one inference result."""

    severity: SeverityResult
    summary: ClinicalSummary
    report_path: Path
    recommendations: list[str]
    limitations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    report: ClinicalReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.to_dict(),
            "summary": self.summary.to_dict(),
            "report_path": str(self.report_path),
            "recommendations": list(self.recommendations),
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }


class ClinicalPipeline:
    """``UnifiedInferenceResult`` → severity → summary → clinical report."""

    def __init__(
        self,
        *,
        severity_engine: SeverityEngine | None = None,
        rules_path: Path | str | None = None,
    ) -> None:
        self._severity = severity_engine or SeverityEngine(rules_path=rules_path)

    @property
    def severity_engine(self) -> SeverityEngine:
        return self._severity

    def run(
        self,
        inference: UnifiedInferenceResult,
        *,
        output_dir: Path | str | None = None,
        report_path: Path | str | None = None,
    ) -> ClinicalPipelineResult:
        if not isinstance(inference, UnifiedInferenceResult):
            raise TypeError(
                "ClinicalPipeline expects UnifiedInferenceResult; it never calls a model."
            )

        severity = self._severity.evaluate(inference)
        summary = build_clinical_summary(inference, severity)

        if report_path is not None:
            out_report = Path(report_path)
        elif output_dir is not None:
            out_report = Path(output_dir) / "clinical_report.md"
        else:
            out_report = None

        report = write_clinical_report(
            inference,
            severity,
            summary,
            output_path=out_report,
        )

        return ClinicalPipelineResult(
            severity=severity,
            summary=summary,
            report_path=report.path,
            recommendations=list(severity.recommendations),
            limitations=list(summary.limitations),
            metadata={
                "model_id": inference.model_id,
                "framework": inference.framework,
                "scan_name": (inference.metadata or {}).get("scan_name"),
                "severity_status": severity.status,
                "rules_path": severity.rules_path,
                "disclaimer": summary.disclaimer,
            },
            report=report,
        )
