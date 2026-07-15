"""Validate the clinical pipeline (post-segmentation only)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import yaml

from clinical.clinical_pipeline import ClinicalPipeline
from clinical.severity_engine import SeverityEngine
from deployment.inference.base import UnifiedInferenceResult
from evaluation.volume_metrics import format_subtype_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = PROJECT_ROOT / "config" / "clinical" / "severity_rules.yaml"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _make_inference(
    *,
    total_volume_ml: float,
    per_class: dict[int, float],
    classes_present: list[int],
    output_dir: Path,
) -> UnifiedInferenceResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_report = output_dir / "prediction_report.md"
    pred_report.write_text("# prediction placeholder\n", encoding="utf-8")
    return UnifiedInferenceResult(
        model_id="monai_best30h",
        framework="monai",
        prediction_mask=np.zeros((8, 8, 4), dtype=np.int64),
        overlay_paths={},
        volumes={
            "total_volume_ml": total_volume_ml,
            "per_class_volume_ml": per_class,
            "classes_present": classes_present,
            "spacing": (1.0, 1.0, 1.0),
        },
        confidence={
            "study_confidence": 0.81,
            "per_class_confidence": {k: 0.7 for k in classes_present},
            "mean_softmax_confidence": 0.75,
            "confidence_histogram": {"counts": [1], "bin_edges": [0.0, 1.0]},
        },
        artifacts={"prediction_report": pred_report},
        report_path=pred_report,
        metadata={
            "scan_name": "clinical_validation_scan",
            "output_dir": str(output_dir),
            "checkpoint_path": "checkpoints/dummy.pth",
            "device": "cpu",
            "processing_time_sec": 1.0,
            "shape": [8, 8, 4],
            "spacing": [1.0, 1.0, 1.0],
        },
    )


def run_validation(*, project_root: Path | None = None) -> int:
    root = project_root or PROJECT_ROOT
    rules_path = root / "config" / "clinical" / "severity_rules.yaml"
    print("Clinical pipeline validation")
    print(f"  project_root: {root}")
    print(f"  rules_path: {rules_path}")

    _assert(rules_path.is_file(), f"Missing severity rules YAML: {rules_path}")
    with rules_path.open(encoding="utf-8") as handle:
        production_rules = yaml.safe_load(handle)
    _assert(isinstance(production_rules, dict), "severity_rules.yaml must be a mapping")
    _assert(
        str(production_rules.get("status")).lower() == "not_configured",
        "Production severity rules must remain status=not_configured until literature freeze",
    )
    for name, entry in (production_rules.get("subtypes") or {}).items():
        _assert(
            str(entry.get("status")).lower() == "not_configured",
            f"Production subtype '{name}' must be not_configured",
        )
        _assert(not (entry.get("rules") or []), f"Production subtype '{name}' must have empty rules")
    print("  YAML rules load + production not_configured gate: OK")

    engine = SeverityEngine(rules_path=rules_path)
    loaded = engine.load_rules()
    _assert(loaded.get("version") == production_rules.get("version"), "Rules version mismatch")
    print("  severity engine loads YAML: OK")

    import clinical.clinical_pipeline as pipe_mod
    import clinical.clinical_report as report_mod
    import clinical.clinical_summary as summary_mod
    import clinical.severity_engine as sev_mod

    for mod in (sev_mod, summary_mod, report_mod, pipe_mod):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        _assert("InferenceFactory" not in text, f"{mod.__file__} must not call InferenceFactory")
        _assert("InferencePipeline" not in text, f"{mod.__file__} must not call InferencePipeline")
        _assert("run_nnunet_prediction" not in text, f"{mod.__file__} must not call nnU-Net")

    pipe_text = Path(pipe_mod.__file__).read_text(encoding="utf-8")
    _assert("UnifiedInferenceResult" in pipe_text, "Pipeline must consume UnifiedInferenceResult")
    _assert("SeverityEngine" in pipe_text, "Pipeline must use SeverityEngine")
    _assert(
        "format_subtype_name" in Path(sev_mod.__file__).read_text(encoding="utf-8"),
        "Severity engine must reuse evaluation subtype labels",
    )
    _assert(
        "format_subtype_name" in Path(summary_mod.__file__).read_text(encoding="utf-8"),
        "Summary must reuse evaluation subtype labels",
    )
    _assert(
        "format_subtype_name" in Path(report_mod.__file__).read_text(encoding="utf-8"),
        "Report must reuse evaluation subtype labels",
    )
    print("  no duplicated model/inference logic; reuses evaluation labels: OK")

    out_root = root / "reports" / "clinical" / "_validation"
    out_root.mkdir(parents=True, exist_ok=True)

    inference = _make_inference(
        total_volume_ml=14.0,
        per_class={1: 5.0, 4: 9.0},
        classes_present=[1, 4],
        output_dir=out_root / "default_rules",
    )
    pipeline = ClinicalPipeline(rules_path=rules_path)
    result = pipeline.run(inference)
    _assert(result.severity.status == "not_configured", "Expected not_configured severity")
    _assert(result.severity.tier is None, "Tier must be None when not_configured")
    _assert(result.severity.hemorrhage_detected is True, "Hemorrhage should be detected from volumes")
    _assert(Path(result.report_path).is_file(), f"Missing clinical report: {result.report_path}")
    _assert("Research Software" in result.summary.narrative, "Summary missing research banner")
    _assert(
        "Not for Clinical Use" in result.summary.disclaimer
        or "Not for Clinical" in result.summary.disclaimer,
        "Disclaimer must state not for clinical use",
    )
    _assert(result.recommendations == [], "Production recommendations must be empty")
    _assert(len(result.limitations) > 0, "Limitations must be present")
    report_text = Path(result.report_path).read_text(encoding="utf-8")
    _assert("Severity" in report_text and "Limitations" in report_text, "Report sections missing")
    _assert("EDH" in report_text and "IPH" in report_text, "Subtype labels missing from report")
    print("  severity engine executes (not_configured): OK")
    print("  summary generated: OK")
    print("  report generated: OK")
    print(f"  report_path: {result.report_path}")

    # Missing-rule handling with global configured but empty subtype rules.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing_rules = {
            "version": 99,
            "status": "configured",
            "disclaimer": "Research Software. Not for Clinical Use.",
            "recommendations": [],
            "limitations": ["Validation fixture"],
            "subtypes": {
                "EDH": {"label_id": 1, "status": "not_configured", "rules": [], "citations": []},
                "IPH": {"label_id": 4, "status": "not_configured", "rules": [], "citations": []},
            },
        }
        fixture = tmp_path / "missing_subtype_rules.yaml"
        fixture.write_text(yaml.safe_dump(missing_rules), encoding="utf-8")
        missing_inference = _make_inference(
            total_volume_ml=3.0,
            per_class={1: 3.0},
            classes_present=[1],
            output_dir=out_root / "missing_rules",
        )
        missing_result = ClinicalPipeline(rules_path=fixture).run(missing_inference)
        _assert(
            missing_result.severity.status == "not_configured",
            "Missing subtype rules must yield not_configured",
        )
        _assert(
            missing_result.severity.subtype_results[0].status == "not_configured",
            "Subtype status must be not_configured when rules are absent",
        )
        print("  missing rules handled: OK")

        # Configured smoke fixture — arbitrary numeric threshold for engine branching only.
        configured = {
            "version": 99,
            "status": "configured",
            "disclaimer": "Research Software. Not for Clinical Use. VALIDATION FIXTURE ONLY.",
            "recommendations": [],
            "limitations": ["Validation fixture — not literature-backed"],
            "subtypes": {
                "EDH": {
                    "label_id": 1,
                    "status": "configured",
                    "citations": ["VALIDATION FIXTURE — not for clinical use"],
                    "rules": [
                        {
                            "name": "validation_volume_ge",
                            "field": "volume_ml",
                            "op": "ge",
                            "value": 1.0,
                            "tier": "fixture_tier",
                            "citation": "VALIDATION FIXTURE — not for clinical use",
                        }
                    ],
                }
            },
        }
        configured_path = tmp_path / "configured_smoke.yaml"
        configured_path.write_text(yaml.safe_dump(configured), encoding="utf-8")
        configured_inference = _make_inference(
            total_volume_ml=3.0,
            per_class={1: 3.0},
            classes_present=[1],
            output_dir=out_root / "configured_smoke",
        )
        configured_result = ClinicalPipeline(rules_path=configured_path).run(configured_inference)
        _assert(configured_result.severity.status == "configured", "Configured fixture failed")
        _assert(configured_result.severity.tier == "fixture_tier", "Expected fixture tier")
        _assert(
            "VALIDATION FIXTURE" in configured_result.severity.citations[0],
            "Configured path must retain YAML citations",
        )
        print("  configured-rule branch (validation fixture only): OK")

    # Labels helper reuse spot-check.
    _assert(format_subtype_name(1) == "EDH", "Unexpected subtype label mapping")

    print("VALIDATION PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate clinical pipeline.")
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()
    try:
        code = run_validation(project_root=args.project_root)
    except Exception as exc:  # noqa: BLE001 — validation entrypoint
        print("VALIDATION FAILED")
        print(f"  - {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
