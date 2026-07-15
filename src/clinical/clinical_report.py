"""Markdown clinical report generation from inference + severity + summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical.clinical_summary import ClinicalSummary
from clinical.severity_engine import SeverityResult
from deployment.inference.base import UnifiedInferenceResult
from evaluation.volume_metrics import format_subtype_name


@dataclass
class ClinicalReport:
    """Paths and content for the written clinical markdown report."""

    path: Path
    markdown: str


def _volume_rows(volumes: dict[str, Any]) -> list[str]:
    per_class = volumes.get("per_class_volume_ml") or {}
    rows: list[str] = []
    if isinstance(per_class, dict):
        for key in sorted(per_class.keys(), key=lambda value: int(value)):
            label = format_subtype_name(int(key))
            rows.append(f"| {label} | {float(per_class[key]):.4f} |")
    total = float(volumes.get("total_volume_ml", 0.0))
    rows.append(f"| **Total** | **{total:.4f}** |")
    return rows


def build_clinical_markdown(
    inference: UnifiedInferenceResult,
    severity: SeverityResult,
    summary: ClinicalSummary,
) -> str:
    """Compose the clinical markdown document (research software only)."""
    meta = inference.metadata or {}
    volumes = inference.volumes or {}
    confidence = inference.confidence or {}
    scan_name = str(meta.get("scan_name") or "unknown_scan")
    spacing = volumes.get("spacing") or meta.get("spacing")
    shape = meta.get("shape")

    lines: list[str] = [
        "# Clinical Report",
        "",
        "> **Research Software — Not for Clinical Use**",
        ">",
        f"> {summary.disclaimer}",
        "",
        "## Scan Information",
        f"- Scan name: `{scan_name}`",
        f"- Shape: `{shape}`",
        f"- Spacing (mm): `{spacing}`",
        f"- Processing time (sec): `{meta.get('processing_time_sec')}`",
        "",
        "## Model",
        f"- Model ID: `{inference.model_id}`",
        f"- Framework: `{inference.framework}`",
        f"- Checkpoint: `{meta.get('checkpoint_path')}`",
        f"- Device: `{meta.get('device')}`",
        "",
        "## Findings Summary",
        "",
        "```",
        summary.narrative,
        "```",
        "",
        "## Volumes",
        f"- Estimated total hemorrhage volume: **{float(volumes.get('total_volume_ml', 0.0)):.4f} mL**",
        "",
        "| Subtype | Volume (mL) |",
        "|---------|------------:|",
    ]
    lines.extend(_volume_rows(volumes))

    study_conf = confidence.get("study_confidence")
    mean_conf = confidence.get("mean_softmax_confidence")
    lines.extend(
        [
            "",
            "## Confidence",
            "",
            "Model softmax confidence only — not a calibrated clinical probability.",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Study confidence | {study_conf} |",
            f"| Mean softmax confidence | {mean_conf} |",
        ]
    )
    per_class_conf = confidence.get("per_class_confidence") or {}
    if isinstance(per_class_conf, dict):
        for key in sorted(per_class_conf.keys(), key=lambda value: int(value)):
            lines.append(
                f"| {format_subtype_name(int(key))} confidence | {float(per_class_conf[key]):.6f} |"
            )

    lines.extend(
        [
            "",
            "## Subtype Table",
            "",
            "| Subtype | Label | Volume (mL) | Severity status | Tier |",
            "|---------|------:|------------:|-----------------|------|",
        ]
    )
    if severity.subtype_results:
        for item in severity.subtype_results:
            lines.append(
                f"| {item.subtype} | {item.label_id} | {item.volume_ml:.4f} | "
                f"{item.status} | {item.tier or '—'} |"
            )
    else:
        lines.append("| — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Severity",
            f"- Overall status: **{severity.status}**",
            f"- Overall tier: `{severity.tier}`",
            f"- Rules file: `{severity.rules_path}`",
            f"- Rules version: `{severity.rules_version}`",
            "",
            "### Severity factors",
        ]
    )
    if severity.factors:
        for factor in severity.factors:
            lines.append(f"- {factor}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "### Severity table",
            "",
            "| Subtype | Status | Tier | Matched rule | Reason |",
            "|---------|--------|------|--------------|--------|",
        ]
    )
    if severity.subtype_results:
        for item in severity.subtype_results:
            lines.append(
                f"| {item.subtype} | {item.status} | {item.tier or '—'} | "
                f"{item.matched_rule or '—'} | {item.reason} |"
            )
    else:
        lines.append("| — | — | — | — | — |")

    lines.extend(["", "## Recommendations", ""])
    if severity.recommendations:
        for item in severity.recommendations:
            lines.append(f"- {item}")
    else:
        lines.append(
            "- None. This research software does not provide treatment or diagnostic recommendations."
        )

    lines.extend(["", "## Limitations", ""])
    for item in summary.limitations:
        lines.append(f"- {item}")

    lines.extend(["", "## Citations", ""])
    if severity.citations:
        for cite in severity.citations:
            lines.append(f"- {cite}")
    else:
        lines.append(
            "- No severity citations available (rules status is not literature-frozen)."
        )

    lines.extend(
        [
            "",
            "## Inference Artifacts",
            f"- Prediction report: `{inference.report_path}`",
        ]
    )
    for key, path in sorted(inference.artifacts.items()):
        lines.append(f"- `{key}`: `{path}`")

    lines.extend(
        [
            "",
            "---",
            "",
            "*Generated by BrainHemorrhageAI clinical pipeline. "
            "Research Software — Not for Clinical Use.*",
            "",
        ]
    )
    return "\n".join(lines)


def write_clinical_report(
    inference: UnifiedInferenceResult,
    severity: SeverityResult,
    summary: ClinicalSummary,
    *,
    output_path: Path | None = None,
) -> ClinicalReport:
    """Write ``clinical_report.md`` next to inference artifacts when possible."""
    if output_path is None:
        meta = inference.metadata or {}
        if meta.get("output_dir"):
            output_path = Path(str(meta["output_dir"])) / "clinical_report.md"
        elif inference.report_path:
            output_path = Path(inference.report_path).parent / "clinical_report.md"
        else:
            output_path = Path("clinical_report.md")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_clinical_markdown(inference, severity, summary)
    output_path.write_text(markdown, encoding="utf-8")
    return ClinicalReport(path=output_path, markdown=markdown)
