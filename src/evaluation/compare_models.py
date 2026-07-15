"""Compare MONAI and nnU-Net evaluation / summary outputs.

Does not retrain or reinvent metrics. Loads existing artifacts and writes
publication-oriented Markdown + CSV summaries.

Examples
--------
python -m evaluation.compare_models
python -m evaluation.compare_models \\
  --monai-eval-dir reports/evaluation/experiment_best30h_eval \\
  --nnunet-summary reports/nnunet/dataset501_fold0/model_summary.json \\
  --output-dir reports/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MONAI_EVAL = PROJECT_ROOT / "reports" / "evaluation" / "experiment_best30h_eval"
DEFAULT_NNUNET_EVAL = PROJECT_ROOT / "reports" / "evaluation" / "nnunet_dataset501_fold0_eval"
DEFAULT_NNUNET_SUMMARY = (
    PROJECT_ROOT / "reports" / "nnunet" / "dataset501_fold0" / "model_summary.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "comparison"

CLASS_NAMES = {1: "EDH", 2: "SDH", 3: "SAH", 4: "IPH", 5: "IVH"}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else str(value)
    return str(value)


def load_monai_eval(eval_dir: Path) -> dict[str, Any]:
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"MONAI metrics.json not found: {metrics_path}")

    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    global_metrics = metrics.get("global_metrics", {})
    per_class = global_metrics.get("per_class", {})

    hemorrhage_ious = []
    per_class_dice: dict[str, float | None] = {}
    per_class_iou: dict[str, float | None] = {}
    for class_id, name in CLASS_NAMES.items():
        entry = per_class.get(str(class_id), {})
        dice = entry.get("dice")
        iou = entry.get("iou")
        per_class_dice[name] = dice
        per_class_iou[name] = iou
        if isinstance(iou, (int, float)):
            hemorrhage_ious.append(float(iou))

    volume_mean = None
    volume_median = None
    volume_csv = eval_dir / "volume_metrics.csv"
    if volume_csv.is_file():
        # Prefer values already published in evaluation_report.md sidecar if present.
        report_path = eval_dir / "evaluation_report.md"
        if report_path.is_file():
            text = report_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "Mean total hemorrhage volume error:" in line:
                    volume_mean = float(line.split(":")[-1].replace("mL", "").strip())
                if "Median total hemorrhage volume error:" in line:
                    volume_median = float(line.split(":")[-1].replace("mL", "").strip())

    confidence_mean = None
    report_path = eval_dir / "evaluation_report.md"
    if report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "Mean study confidence:" in line:
                confidence_mean = float(line.split(":")[-1].strip())
                break

    return {
        "model_id": "monai_best30h",
        "framework": "MONAI",
        "checkpoint_path": metrics.get("checkpoint_path"),
        "checkpoint_epoch": metrics.get("checkpoint_epoch"),
        "val_macro_dice": metrics.get("checkpoint_val_macro_dice"),
        "split": metrics.get("split"),
        "scans_evaluated": metrics.get("scans_evaluated"),
        "test_macro_dice": global_metrics.get("macro_dice"),
        "test_micro_dice": global_metrics.get("micro_dice"),
        "test_mean_iou_hemorrhage": (
            sum(hemorrhage_ious) / len(hemorrhage_ious) if hemorrhage_ious else None
        ),
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
        "volume_error_mean_ml": volume_mean,
        "volume_error_median_ml": volume_median,
        "mean_study_confidence": confidence_mean,
        "hardware": (metrics.get("environment") or {}).get("platform"),
        "inference_speed": "See docs/results/monai_results.md (API demos ~19–22s CPU)",
    }


def load_nnunet_eval(eval_dir: Path) -> dict[str, Any]:
    """Load locked-test nnU-Net evaluation artifacts (preferred when present)."""
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"nnU-Net metrics.json not found: {metrics_path}")

    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    global_metrics = metrics.get("global_metrics", {})
    per_class = global_metrics.get("per_class", {})
    nnunet_meta = metrics.get("nnunet") or {}

    hemorrhage_ious: list[float] = []
    per_class_dice: dict[str, float | None] = {}
    per_class_iou: dict[str, float | None] = {}
    for class_id, name in CLASS_NAMES.items():
        entry = per_class.get(str(class_id), {})
        dice = entry.get("dice")
        iou = entry.get("iou")
        per_class_dice[name] = dice
        per_class_iou[name] = iou
        if isinstance(iou, (int, float)):
            hemorrhage_ious.append(float(iou))

    volume_mean = volume_median = confidence_mean = None
    report_path = eval_dir / "evaluation_report.md"
    if report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "Mean total hemorrhage volume error:" in line:
                volume_mean = float(line.split(":")[-1].replace("mL", "").strip())
            if "Median total hemorrhage volume error:" in line:
                volume_median = float(line.split(":")[-1].replace("mL", "").strip())
            if "Mean study confidence:" in line:
                confidence_mean = float(line.split(":")[-1].strip())

    mean_infer = nnunet_meta.get("mean_inference_seconds")
    if mean_infer is None or (isinstance(mean_infer, float) and mean_infer != mean_infer):
        times = nnunet_meta.get("inference_seconds_per_case") or {}
        valid = [
            float(v)
            for v in times.values()
            if isinstance(v, (int, float)) and float(v) == float(v)
        ]
        mean_infer = sum(valid) / len(valid) if valid else None

    return {
        "model_id": "nnunet_fold0",
        "framework": "nnU-Net",
        "checkpoint_path": metrics.get("checkpoint_path"),
        "checkpoint_epoch": nnunet_meta.get("best_ema_epoch")
        or metrics.get("checkpoint_val_ema_pseudo_dice"),
        "val_macro_dice": metrics.get("checkpoint_val_ema_pseudo_dice"),
        "split": metrics.get("split"),
        "scans_evaluated": metrics.get("scans_evaluated"),
        "test_macro_dice": global_metrics.get("macro_dice"),
        "test_micro_dice": global_metrics.get("micro_dice"),
        "test_mean_iou_hemorrhage": (
            sum(hemorrhage_ious) / len(hemorrhage_ious) if hemorrhage_ious else None
        ),
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
        "volume_error_mean_ml": volume_mean,
        "volume_error_median_ml": volume_median,
        "mean_study_confidence": confidence_mean,
        "hardware": nnunet_meta.get("device") or (metrics.get("environment") or {}).get("platform"),
        "inference_speed": (
            f"CPU mean {mean_infer:.2f}s/case" if isinstance(mean_infer, (int, float)) else "N/A"
        ),
        "epochs_completed": None,
        "configuration": nnunet_meta.get("configuration"),
        "parameters": nnunet_meta.get("parameters"),
    }


def load_nnunet_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"nnU-Net summary not found: {summary_path}")

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    locked = summary.get("locked_test") or {}
    pseudo = summary.get("pseudo_dice_at_best_ema_epoch") or {}
    per_class_dice = {k: pseudo.get(k) for k in CLASS_NAMES.values()}

    return {
        "model_id": summary.get("model_id", "nnunet_fold0"),
        "framework": summary.get("framework", "nnU-Net"),
        "checkpoint_path": summary.get("checkpoint_path"),
        "checkpoint_epoch": summary.get("best_ema_recorded_after_epoch"),
        "val_macro_dice": summary.get("best_val_ema_pseudo_dice"),
        "split": "val (EMA); locked test held out",
        "scans_evaluated": None,
        "test_macro_dice": locked.get("macro_dice"),
        "test_micro_dice": locked.get("micro_dice"),
        "test_mean_iou_hemorrhage": None,
        "per_class_dice": per_class_dice,
        "per_class_iou": {k: None for k in CLASS_NAMES.values()},
        "volume_error_mean_ml": (locked.get("volume_error_ml")),
        "volume_error_median_ml": None,
        "mean_study_confidence": locked.get("confidence"),
        "hardware": summary.get("hardware"),
        "inference_speed": (summary.get("inference_speed") or {}).get("reason")
        or _fmt((summary.get("inference_speed") or {}).get("value")),
        "epochs_completed": summary.get("epochs_completed"),
        "configuration": summary.get("configuration"),
        "metric_caveat": (
            "val_macro_dice field holds nnU-Net best EMA pseudo Dice — not project macro Dice"
        ),
    }


def build_rows(monai: dict[str, Any], nnunet: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {"metric": "framework", "monai": _fmt(monai["framework"]), "nnunet": _fmt(nnunet["framework"])},
        {
            "metric": "checkpoint",
            "monai": _fmt(monai["checkpoint_path"]),
            "nnunet": _fmt(nnunet["checkpoint_path"]),
        },
        {
            "metric": "best_epoch_or_ema_epoch",
            "monai": _fmt(monai["checkpoint_epoch"]),
            "nnunet": _fmt(nnunet["checkpoint_epoch"]),
        },
        {
            "metric": "val_headline_dice",
            "monai": _fmt(monai["val_macro_dice"]),
            "nnunet": _fmt(nnunet["val_macro_dice"]),
        },
        {
            "metric": "test_macro_dice",
            "monai": _fmt(monai["test_macro_dice"]),
            "nnunet": _fmt(nnunet["test_macro_dice"]),
        },
        {
            "metric": "test_micro_dice",
            "monai": _fmt(monai["test_micro_dice"]),
            "nnunet": _fmt(nnunet["test_micro_dice"]),
        },
        {
            "metric": "test_mean_iou_hemorrhage",
            "monai": _fmt(monai["test_mean_iou_hemorrhage"]),
            "nnunet": _fmt(nnunet["test_mean_iou_hemorrhage"]),
        },
        {
            "metric": "volume_error_mean_ml",
            "monai": _fmt(monai["volume_error_mean_ml"], digits=4),
            "nnunet": _fmt(nnunet["volume_error_mean_ml"]),
        },
        {
            "metric": "volume_error_median_ml",
            "monai": _fmt(monai["volume_error_median_ml"], digits=4),
            "nnunet": _fmt(nnunet["volume_error_median_ml"]),
        },
        {
            "metric": "mean_study_confidence",
            "monai": _fmt(monai["mean_study_confidence"], digits=4),
            "nnunet": _fmt(nnunet["mean_study_confidence"]),
        },
        {
            "metric": "hardware",
            "monai": _fmt(monai["hardware"]),
            "nnunet": _fmt(nnunet["hardware"]),
        },
        {
            "metric": "inference_speed",
            "monai": _fmt(monai["inference_speed"]),
            "nnunet": _fmt(nnunet["inference_speed"]),
        },
    ]

    for name in CLASS_NAMES.values():
        rows.append(
            {
                "metric": f"dice_{name}",
                "monai": _fmt((monai.get("per_class_dice") or {}).get(name)),
                "nnunet": _fmt((nnunet.get("per_class_dice") or {}).get(name)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "monai", "nnunet"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, str]],
    path: Path,
    monai: dict[str, Any],
    nnunet: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Comparison (generated)",
        "",
        "Generated by `src/evaluation/compare_models.py` from existing evaluation outputs.",
        "Does not retrain models. `N/A` means not available in source artifacts.",
        "",
        "## Caveats",
        "",
        "- MONAI and nnU-Net Dice/IoU/volume/confidence rows use the **locked test** evaluation reports when available.",
        "- nnU-Net `val_headline_dice` is **EMA pseudo Dice** from training (not project macro Dice).",
        "- Prefer `docs/results/publication_comparison_tables.md` for the full publication table set.",
        "",
        f"- MONAI scans evaluated: {_fmt(monai.get('scans_evaluated'))}",
        f"- nnU-Net scans evaluated: {_fmt(nnunet.get('scans_evaluated'))}",
        f"- nnU-Net configuration: {_fmt(nnunet.get('configuration'))}",
        "",
        "## Side-by-side table",
        "",
        "| Metric | MONAI | nnU-Net |",
        "|--------|------:|--------:|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['monai']} | {row['nnunet']} |")

    lines.extend(
        [
            "",
            "## Publication-ready bullets",
            "",
            f"- MONAI locked-test macro Dice: **{_fmt(monai.get('test_macro_dice'))}** "
            f"(best val macro Dice {_fmt(monai.get('val_macro_dice'))} @ epoch "
            f"{_fmt(monai.get('checkpoint_epoch'))}).",
            f"- nnU-Net locked-test macro Dice: **{_fmt(nnunet.get('test_macro_dice'))}** "
            f"(best val EMA pseudo Dice {_fmt(nnunet.get('val_macro_dice'))}).",
            "- Prefer nnU-Net for locked-test segmentation quality; MONAI remains the served demo path until dual serving.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_models(
    monai_eval_dir: Path,
    nnunet_summary: Path,
    output_dir: Path,
    nnunet_eval_dir: Path | None = None,
) -> dict[str, Path]:
    monai = load_monai_eval(monai_eval_dir)
    if nnunet_eval_dir is not None and (nnunet_eval_dir / "metrics.json").is_file():
        nnunet = load_nnunet_eval(nnunet_eval_dir)
    else:
        nnunet = load_nnunet_summary(nnunet_summary)
    rows = build_rows(monai, nnunet)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "monai_vs_nnunet.csv"
    md_path = output_dir / "monai_vs_nnunet.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, monai, nnunet)
    return {"csv": csv_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MONAI and nnU-Net evaluation outputs without retraining."
    )
    parser.add_argument(
        "--monai-eval-dir",
        type=Path,
        default=DEFAULT_MONAI_EVAL,
        help="Directory containing metrics.json / evaluation_report.md",
    )
    parser.add_argument(
        "--nnunet-eval-dir",
        type=Path,
        default=DEFAULT_NNUNET_EVAL,
        help="Locked-test nnU-Net evaluation directory (preferred when metrics.json exists)",
    )
    parser.add_argument(
        "--nnunet-summary",
        type=Path,
        default=DEFAULT_NNUNET_SUMMARY,
        help="Fallback JSON summary if locked-test eval is absent",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write Markdown and CSV summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = compare_models(
        monai_eval_dir=args.monai_eval_dir.resolve(),
        nnunet_summary=args.nnunet_summary.resolve(),
        output_dir=args.output_dir.resolve(),
        nnunet_eval_dir=args.nnunet_eval_dir.resolve(),
    )
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['csv']}")


if __name__ == "__main__":
    main()
