"""Clinical post-processing layer — severity, summary, and report (Phase C.3).

Consumes ``UnifiedInferenceResult`` only. Never calls a model.
"""

from clinical.clinical_pipeline import ClinicalPipeline, ClinicalPipelineResult
from clinical.clinical_report import ClinicalReport, write_clinical_report
from clinical.clinical_summary import ClinicalSummary, build_clinical_summary
from clinical.severity_engine import SeverityEngine, SeverityResult

__all__ = [
    "ClinicalPipeline",
    "ClinicalPipelineResult",
    "ClinicalReport",
    "ClinicalSummary",
    "SeverityEngine",
    "SeverityResult",
    "build_clinical_summary",
    "write_clinical_report",
]
