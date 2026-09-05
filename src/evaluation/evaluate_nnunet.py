"""Evaluate nnU-Net predictions on the locked BHSD test split.

Research adapter only — does not modify training, preprocessing, inference API,
frontend, or metric definitions. Reuses shared evaluation modules:

- evaluation.metrics
- evaluation.volume_metrics
- evaluation.confidence
- evaluation.plots
- evaluation.experiment_logger
- evaluation.evaluate_model.MetricAccumulator (same aggregation as MONAI)

Requires ``nnunetv2`` and the trained fold-0 checkpoint. Produces the same
artifact schema as ``reports/evaluation/experiment_best30h_eval/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from dataset.bhsd_dataset import load_split_filenames
from evaluation.confidence import (
    all_class_confidences,
    confidence_histogram,
    study_confidence,
)
from evaluation.evaluate_model import MetricAccumulator
from evaluation.experiment_logger import collect_environment_info, compute_split_hash
from evaluation.metrics import HEMORRHAGE_CLASSES, NUM_CLASSES, compute_segmentation_metrics
from evaluation.plots import (
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_per_class_dice,
    plot_scan_examples,
    plot_volume_error,
    plot_volume_error_distribution,
)
from evaluation.volume_metrics import (
    SUBTYPE_LABELS,
    absolute_volume_error_ml,
    format_subtype_name,
    per_subtype_volume_error,
    total_hemorrhage_volume_ml,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "nnunet_dataset501_fold0"
DEFAULT_IMAGES_TS = PROJECT_ROOT / "data" / "nnunet" / "nnUNet_raw" / "Dataset501_BHSD" / "imagesTs"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "ground truths"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "label_192" / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "evaluation" / "nnunet_dataset501_fold0_eval"
CLASS_LABELS = ["BG", "EDH", "SDH", "SAH", "IPH", "IVH"]


def _case_id_from_nnunet_image(path: Path) -> str:
    """Map ``ID_xxx_0000.nii.gz`` → ``ID_xxx``."""
    name = path.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
    else:
        stem = Path(name).stem
    if stem.endswith("_0000"):
        stem = stem[: -len("_0000")]
    return stem


def _gt_path_for_case(case_id: str, labels_dir: Path) -> Path:
    gz = labels_dir / f"{case_id}.nii.gz"
    if gz.is_file():
        return gz
    nii = labels_dir / f"{case_id}.nii"
    if nii.is_file():
        return nii
    raise FileNotFoundError(f"Ground-truth mask not found for {case_id} under {labels_dir}")


def _image_path_for_case(case_id: str, images_dir: Path) -> Path:
    gz = images_dir / f"{case_id}.nii.gz"
    if gz.is_file():
        return gz
    nii = images_dir / f"{case_id}.nii"
    if nii.is_file():
        return nii
    raise FileNotFoundError(f"CT image not found for {case_id} under {images_dir}")


def _first_existing_file(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    listed = ", ".join(names)
    raise FileNotFoundError(f"None of [{listed}] found in {directory}")


def prepare_nnunet_model_folder(
    checkpoint_dir: Path,
    staging_root: Path,
    *,
    use_folds: tuple[int, ...] = (0,),
) -> Path:
    """Build the nnU-Net results layout expected by ``nnUNetPredictor``.

    Does not alter the original checkpoint directory.

    Single-fold (default ``use_folds=(0,)``): ``checkpoint_dir`` contains
    ``checkpoint_best.pth``, ``nnUNetPlans.json`` / ``plans.json``, and
    ``dataset.json`` (V1 ``nnunet_fold0`` layout).

    Multi-fold: ``checkpoint_dir`` is either the SSL root
    ``checkpoints/nnunet_ssl_2fold`` or ``.../fold_0``. Fold weights live in
    ``fold_{n}/checkpoint_best.pth``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    folds = tuple(int(f) for f in use_folds)
    if not folds:
        raise ValueError("use_folds must contain at least one fold index")

    model_dir = (
        staging_root
        / "Dataset501_BHSD"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    if len(folds) == 1:
        src_ckpt = checkpoint_dir / "checkpoint_best.pth"
        if not src_ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {src_ckpt}")
        fold_dir = model_dir / f"fold_{folds[0]}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_ckpt, fold_dir / "checkpoint_best.pth")
        sidecar_root = checkpoint_dir
    else:
        root = (
            checkpoint_dir.parent
            if checkpoint_dir.name.startswith("fold_")
            else checkpoint_dir
        )
        for fold in folds:
            src_ckpt = root / f"fold_{fold}" / "checkpoint_best.pth"
            if not src_ckpt.is_file():
                raise FileNotFoundError(f"Missing fold {fold} checkpoint: {src_ckpt}")
            fold_dir = model_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_ckpt, fold_dir / "checkpoint_best.pth")
        sidecar_root = root

    plans_src = _first_existing_file(sidecar_root, ("nnUNetPlans.json", "plans.json"))
    shutil.copy2(plans_src, model_dir / "plans.json")

    dataset_src = sidecar_root / "dataset.json"
    if not dataset_src.is_file():
        raise FileNotFoundError(f"Missing dataset.json: {dataset_src}")
    dataset = json.loads(dataset_src.read_text(encoding="utf-8"))
    # Local raw conversion uses .nii.gz; training dataset.json may say .nii
    dataset["file_ending"] = ".nii.gz"
    (model_dir / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    return model_dir


def run_nnunet_prediction(
    model_dir: Path,
    images_ts: Path,
    pred_dir: Path,
    *,
    device: torch.device,
    use_mirroring: bool,
    case_limit: int | None,
    use_folds: tuple[int, ...] = (0,),
) -> dict[str, float]:
    """Run nnU-Net inference on locked test images. Returns per-case wall times."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    pred_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(images_ts.glob("*_0000.nii.gz"))
    if not image_files:
        image_files = sorted(images_ts.glob("*_0000.nii"))
    if not image_files:
        raise FileNotFoundError(f"No nnU-Net test images found in {images_ts}")
    if case_limit is not None:
        image_files = image_files[: int(case_limit)]

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=tuple(int(f) for f in use_folds),
        checkpoint_name="checkpoint_best.pth",
    )

    case_times: dict[str, float] = {}
    for image_path in image_files:
        case_id = _case_id_from_nnunet_image(image_path)
        out_seg = pred_dir / f"{case_id}.nii.gz"
        if out_seg.is_file() and (pred_dir / f"{case_id}.npz").is_file():
            logger.info("Skipping existing prediction for %s", case_id)
            case_times[case_id] = float("nan")
            continue
        list_of_lists = [[str(image_path)]]
        output_files = [str(pred_dir / case_id)]
        t0 = time.perf_counter()
        predictor.predict_from_files(
            list_of_lists,
            output_files,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        elapsed = time.perf_counter() - t0
        case_times[case_id] = elapsed
        logger.info("Predicted %s in %.1fs", case_id, elapsed)
    return case_times


def _load_probabilities(pred_dir: Path, case_id: str) -> np.ndarray | None:
    """Load softmax probabilities if exported (``*.npz``)."""
    npz = pred_dir / f"{case_id}.npz"
    if not npz.is_file():
        return None
    data = np.load(npz)
    # nnU-Net stores probabilities under 'probabilities' or 'softmax'
    for key in ("probabilities", "softmax", "prob"):
        if key in data:
            arr = np.asarray(data[key])
            # Expected (C, X, Y, Z) or (C, Z, Y, X) — nnU-Net uses (C, X, Y, Z)
            return arr.astype(np.float32)
    # Fallback: first array
    key = list(data.keys())[0]
    return np.asarray(data[key]).astype(np.float32)


def _load_segmentation(pred_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load predicted label map and return (labels, affine)."""
    for name in (f"{case_id}.nii.gz", f"{case_id}.nii"):
        path = pred_dir / name
        if path.is_file():
            nii = nib.load(str(path))
            return np.asarray(nii.get_fdata(), dtype=np.int64), np.asarray(nii.affine)
    raise FileNotFoundError(f"Segmentation not found for {case_id} in {pred_dir}")


def _align_probabilities(probabilities: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Align nnU-Net probability axes to the prediction / GT NIfTI layout."""
    if probabilities.ndim != 4:
        raise ValueError(f"probabilities must be 4D (C, ...), got {probabilities.shape}")
    if probabilities.shape[0] != NUM_CLASSES and probabilities.shape[-1] == NUM_CLASSES:
        probabilities = np.moveaxis(probabilities, -1, 0)
    if probabilities.shape[0] != NUM_CLASSES:
        raise ValueError(f"unexpected probability shape {probabilities.shape}")
    if probabilities.shape[1:] == prediction.shape:
        return probabilities.astype(np.float32, copy=False)

    # Common export: (C, Z, Y, X) while NIfTI labels are (X, Y, Z)
    candidate = np.transpose(probabilities, (0, 2, 3, 1))
    if candidate.shape[1:] == prediction.shape:
        return candidate.astype(np.float32, copy=False)

    # Try remaining permutations of spatial axes
    for order in (
        (0, 1, 3, 2),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (0, 3, 2, 1),
    ):
        candidate = np.transpose(probabilities, order)
        if candidate.shape[1:] == prediction.shape:
            return candidate.astype(np.float32, copy=False)

    raise ValueError(
        f"cannot align probability spatial {probabilities.shape[1:]} to pred {prediction.shape}"
    )


def evaluate_case(
    case_id: str,
    pred_dir: Path,
    labels_dir: Path,
    images_dir: Path,
    accumulator: MetricAccumulator,
) -> dict[str, Any]:
    """Score one locked-test case with shared metric modules."""
    prediction, affine = _load_segmentation(pred_dir, case_id)
    gt_nii = nib.load(str(_gt_path_for_case(case_id, labels_dir)))
    ground_truth = np.asarray(gt_nii.get_fdata(), dtype=np.int64)
    spacing = tuple(float(v) for v in gt_nii.header.get_zooms()[:3])

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"{case_id}: prediction shape {prediction.shape} != GT {ground_truth.shape}"
        )

    probabilities = _load_probabilities(pred_dir, case_id)
    if probabilities is None:
        eye = np.eye(NUM_CLASSES, dtype=np.float32)
        probabilities = eye[prediction].transpose(3, 0, 1, 2)
    else:
        probabilities = _align_probabilities(probabilities, prediction)

    pred_t = torch.from_numpy(prediction)
    gt_t = torch.from_numpy(ground_truth)
    probs_t = torch.from_numpy(probabilities)

    # Same formulas as MetricAccumulator.update_batch, vectorized for 3D volumes.
    from evaluation.metrics import binary_confusion

    for class_index in range(NUM_CLASSES):
        confusion = binary_confusion(pred_t == class_index, gt_t == class_index)
        accumulator.tp[class_index] += confusion.tp.cpu()
        accumulator.fp[class_index] += confusion.fp.cpu()
        accumulator.fn[class_index] += confusion.fn.cpu()

    gt_flat = ground_truth.ravel().astype(np.int64)
    pred_flat = prediction.ravel().astype(np.int64)
    flat_index = gt_flat * NUM_CLASSES + pred_flat
    cm_add = np.bincount(flat_index, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
        NUM_CLASSES, NUM_CLASSES
    )
    accumulator.confusion_matrix += cm_add

    counts, bin_edges = confidence_histogram(probs_t, bins=20)
    if accumulator.confidence_counts.shape[0] != counts.shape[0]:
        accumulator.confidence_counts = np.zeros_like(counts)
        accumulator.confidence_bin_edges = bin_edges
    accumulator.confidence_counts += counts.astype(np.int64)

    scan_metrics = compute_segmentation_metrics(pred_t, gt_t, ignore_background=True)
    study_conf = float(study_confidence(probs_t, pred_t).item())
    class_conf = all_class_confidences(probs_t, pred_t)
    volume_errors = per_subtype_volume_error(prediction, ground_truth, spacing)
    pred_total = total_hemorrhage_volume_ml(prediction, spacing)
    gt_total = total_hemorrhage_volume_ml(ground_truth, spacing)
    total_error = absolute_volume_error_ml(pred_total, gt_total)

    # Representative mid-depth axial slice for overlays (axis 2 for RAS BHSD)
    depth = prediction.shape[2]
    rep = depth // 2
    image = nib.load(str(_image_path_for_case(case_id, images_dir))).get_fdata()
    image_slice = image[:, :, rep]
    return {
        "filename": f"{case_id}.nii.gz",
        "case_id": case_id,
        "macro_dice": float(scan_metrics.macro_dice.item()),
        "per_class_dice": {
            i: float(scan_metrics.dice[i].item()) for i in range(NUM_CLASSES)
        },
        "study_confidence": study_conf,
        "class_confidences": class_conf,
        "total_volume_error_ml": total_error,
        "volume_errors": volume_errors,
        "representative_slice": rep,
        "image_slice": image_slice,
        "gt_slice": ground_truth[:, :, rep],
        "pred_slice": prediction[:, :, rep],
        "spacing": spacing,
        "affine": affine,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "probabilities": probabilities,
    }


def _global_metrics(accumulator: MetricAccumulator) -> dict[str, Any]:
    dice = accumulator.per_class_dice()
    iou = accumulator.per_class_iou()
    precision = accumulator.per_class_precision()
    recall = accumulator.per_class_recall()
    specificity = accumulator.per_class_specificity()
    hemorrhage = list(HEMORRHAGE_CLASSES)
    micro_tp = accumulator.tp[hemorrhage].sum()
    micro_fp = accumulator.fp[hemorrhage].sum()
    micro_fn = accumulator.fn[hemorrhage].sum()
    return {
        "macro_dice": accumulator.macro_dice(),
        "micro_dice": float((2.0 * micro_tp) / (2.0 * micro_tp + micro_fp + micro_fn + 1e-7)),
        "per_class": {
            str(i): {
                "dice": float(dice[i].item()),
                "iou": float(iou[i].item()),
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "sensitivity": float(recall[i].item()),
                "specificity": float(specificity[i].item()),
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": accumulator.confusion_matrix.tolist(),
        "confidence_histogram": {
            "counts": accumulator.confidence_counts.tolist(),
            "bin_edges": accumulator.confidence_bin_edges.tolist(),
        },
    }


def _failure_analysis(scan_results: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    if not scan_results:
        return {
            "worst_dice_scans": [],
            "best_dice_scans": [],
            "largest_volume_error_scans": [],
            "lowest_confidence_scans": [],
        }

    def summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "filename": item["filename"],
            "macro_dice": item["macro_dice"],
            "total_volume_error_ml": item["total_volume_error_ml"],
            "study_confidence": item["study_confidence"],
        }

    by_dice = sorted(scan_results, key=lambda x: x["macro_dice"])
    by_vol = sorted(scan_results, key=lambda x: x["total_volume_error_ml"], reverse=True)
    by_conf = sorted(scan_results, key=lambda x: x["study_confidence"])
    k = min(top_k, len(scan_results))
    return {
        "worst_dice_scans": [summary(x) for x in by_dice[:k]],
        "best_dice_scans": [summary(x) for x in by_dice[-k:][::-1]],
        "largest_volume_error_scans": [summary(x) for x in by_vol[:k]],
        "lowest_confidence_scans": [summary(x) for x in by_conf[:k]],
    }


def write_artifacts(
    run_dir: Path,
    global_metrics: dict[str, Any],
    scan_results: list[dict[str, Any]],
    failure_analysis: dict[str, Any],
    *,
    checkpoint_path: Path,
    environment: dict[str, Any],
    case_times: dict[str, float],
    param_count: int | None,
    use_mirroring: bool,
    device: str,
) -> None:
    """Write MONAI-compatible evaluation artifact set."""
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    predictions_dir = run_dir / "predictions"
    overlays_dir = run_dir / "overlays"
    masks_dir = run_dir / "prediction_masks"
    for path in (plots_dir, predictions_dir, overlays_dir, masks_dir):
        path.mkdir(parents=True, exist_ok=True)

    valid_times = [
        float(v)
        for v in case_times.values()
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(float(v)))
    ]
    mean_inference = float(sum(valid_times) / len(valid_times)) if valid_times else None

    payload = {
        "run_id": run_dir.name,
        "split": "test",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": None,
        "checkpoint_val_macro_dice": None,
        "checkpoint_val_ema_pseudo_dice": 0.47909998893737793,
        "environment": environment,
        "nnunet": {
            "configuration": "3d_fullres",
            "fold": 0,
            "use_mirroring": use_mirroring,
            "device": device,
            "parameters": param_count,
            "inference_seconds_per_case": {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in case_times.items()
            },
            "mean_inference_seconds": mean_inference,
            "timed_cases": len(valid_times),
        },
        "global_metrics": global_metrics,
        "failure_analysis": failure_analysis,
        "scans_evaluated": len(scan_results),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with (run_dir / "failure_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(failure_analysis, handle, indent=2)

    with (run_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_index",
                "subtype",
                "dice",
                "iou",
                "precision",
                "recall",
                "sensitivity",
                "specificity",
            ],
        )
        writer.writeheader()
        for class_index in range(NUM_CLASSES):
            metrics = global_metrics["per_class"][str(class_index)]
            writer.writerow(
                {
                    "class_index": class_index,
                    "subtype": format_subtype_name(class_index) if class_index else "BG",
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                }
            )

    volume_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for result in scan_results:
        for label, metrics in result["volume_errors"].items():
            volume_rows.append(
                {
                    "filename": result["filename"],
                    "label": label,
                    "subtype": format_subtype_name(label),
                    "pred_ml": metrics["pred_ml"],
                    "gt_ml": metrics["gt_ml"],
                    "abs_error_ml": metrics["abs_error_ml"],
                    "rel_error": metrics["rel_error"],
                }
            )
        conf_row: dict[str, Any] = {
            "filename": result["filename"],
            "study_confidence": result["study_confidence"],
        }
        for label, value in result["class_confidences"].items():
            conf_row[f"class_{label}_confidence"] = value
        confidence_rows.append(conf_row)
        summary_rows.append(
            {
                "filename": result["filename"],
                "macro_dice": result["macro_dice"],
                "total_volume_error_ml": result["total_volume_error_ml"],
                "study_confidence": result["study_confidence"],
                "inference_seconds": case_times.get(result["case_id"]),
            }
        )

        np.savez_compressed(
            predictions_dir / f"{result['filename']}.npz",
            prediction=result["prediction"],
            ground_truth=result["ground_truth"],
            spacing=np.array(result["spacing"]),
            affine=result["affine"],
            macro_dice=result["macro_dice"],
        )
        mask_nii = nib.Nifti1Image(
            result["prediction"].astype(np.uint8),
            result["affine"],
        )
        nib.save(mask_nii, str(masks_dir / f"{result['case_id']}_pred.nii.gz"))

        # Overlay PNG (CT + prediction outline using mid-slice)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.clip(result["image_slice"].T, -40, 120), cmap="gray", origin="lower")
        axes[0].set_title("CT")
        axes[0].axis("off")
        axes[1].imshow(result["gt_slice"].T, cmap="nipy_spectral", origin="lower", vmin=0, vmax=5)
        axes[1].set_title("GT")
        axes[1].axis("off")
        axes[2].imshow(np.clip(result["image_slice"].T, -40, 120), cmap="gray", origin="lower")
        overlay = np.ma.masked_where(result["pred_slice"] == 0, result["pred_slice"])
        axes[2].imshow(overlay.T, cmap="nipy_spectral", origin="lower", alpha=0.5, vmin=0, vmax=5)
        axes[2].set_title("Pred overlay")
        axes[2].axis("off")
        fig.suptitle(result["filename"])
        fig.tight_layout()
        fig.savefig(overlays_dir / f"{result['case_id']}_overlay.png", dpi=120)
        plt.close(fig)

    pd.DataFrame(volume_rows).to_csv(run_dir / "volume_metrics.csv", index=False)
    pd.DataFrame(confidence_rows).to_csv(run_dir / "confidence_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(run_dir / "prediction_summary.csv", index=False)

    # Plots (shared)
    hemorrhage_dice = {
        i: global_metrics["per_class"][str(i)]["dice"] for i in HEMORRHAGE_CLASSES
    }
    plot_per_class_dice(
        hemorrhage_dice,
        output_path=plots_dir / "per_class_dice.png",
        title="nnU-Net Locked Test Per-Class Dice",
    )
    plot_confusion_matrix(
        np.array(global_metrics["confusion_matrix"]),
        class_labels=CLASS_LABELS,
        output_path=plots_dir / "confusion_matrix.png",
        title="nnU-Net Locked Test Confusion Matrix",
    )
    counts = np.array(global_metrics["confidence_histogram"]["counts"])
    edges = np.array(global_metrics["confidence_histogram"]["bin_edges"])
    plot_confidence_histogram(
        counts,
        edges,
        output_path=plots_dir / "confidence_histogram.png",
        title="nnU-Net Locked Test Confidence Histogram",
    )
    volume_errors = [r["total_volume_error_ml"] for r in scan_results]
    plot_volume_error_distribution(
        volume_errors,
        output_path=plots_dir / "volume_error_distribution.png",
        title="nnU-Net Total Volume Absolute Error (mL)",
    )
    mean_subtype: dict[int, float] = {}
    for label in HEMORRHAGE_CLASSES:
        vals = [
            r["volume_errors"][label]["abs_error_ml"]
            for r in scan_results
            if label in r["volume_errors"]
        ]
        if vals:
            mean_subtype[label] = float(np.mean(vals))
    if mean_subtype:
        plot_volume_error(
            mean_subtype,
            output_path=plots_dir / "mean_subtype_volume_error.png",
            title="nnU-Net Mean Per-Subtype Volume Absolute Error (mL)",
        )

    worst_names = {x["filename"] for x in failure_analysis["worst_dice_scans"][:5]}
    best_names = {x["filename"] for x in failure_analysis["best_dice_scans"][:5]}
    worst_examples = [
        {
            "filename": f"{r['filename']} (Dice={r['macro_dice']:.3f})",
            "image": r["image_slice"],
            "ground_truth": r["gt_slice"],
            "prediction": r["pred_slice"],
        }
        for r in scan_results
        if r["filename"] in worst_names
    ]
    best_examples = [
        {
            "filename": f"{r['filename']} (Dice={r['macro_dice']:.3f})",
            "image": r["image_slice"],
            "ground_truth": r["gt_slice"],
            "prediction": r["pred_slice"],
        }
        for r in scan_results
        if r["filename"] in best_names
    ]
    if worst_examples:
        plot_scan_examples(
            worst_examples,
            output_path=plots_dir / "worst_predictions.png",
            title="nnU-Net Worst Dice Scans",
        )
    if best_examples:
        plot_scan_examples(
            best_examples,
            output_path=plots_dir / "best_predictions.png",
            title="nnU-Net Best Dice Scans",
        )

    # Markdown report
    vol_err = [r["total_volume_error_ml"] for r in scan_results]
    confs = [r["study_confidence"] for r in scan_results]
    lines = [
        "# nnU-Net Locked-Test Evaluation Report",
        "",
        "## Dataset",
        "- Split: test (locked)",
        f"- Test scans in manifest: {len(load_split_filenames('test'))}",
        f"- Test scans evaluated: {len(scan_results)}",
        f"- Split hash: `{compute_split_hash()}`",
        "",
        "## Model",
        "- Framework: nnU-Net v2",
        "- Configuration: `3d_fullres`",
        "- Fold: 0",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Test-time mirroring: {use_mirroring}",
        f"- Device: {device}",
        f"- Parameters: {param_count if param_count is not None else 'N/A'}",
        "",
        "## Global Test Metrics",
        f"- Macro Dice (classes 1-5): **{global_metrics['macro_dice']:.6f}**",
        f"- Micro Dice: {global_metrics['micro_dice']:.6f}",
        "",
        "### Per-Class Metrics",
        "| Class | Subtype | Dice | IoU | Precision | Recall |",
        "|------:|---------|-----:|----:|----------:|-------:|",
    ]
    for class_index in range(NUM_CLASSES):
        m = global_metrics["per_class"][str(class_index)]
        subtype = "BG" if class_index == 0 else SUBTYPE_LABELS.get(class_index, str(class_index))
        lines.append(
            f"| {class_index} | {subtype} | {m['dice']:.4f} | {m['iou']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} |"
        )
    if scan_results:
        lines.extend(
            [
                "",
                "## Volume Metrics (native spacing, mL)",
                f"- Mean total hemorrhage volume error: {float(np.mean(vol_err)):.4f} mL",
                f"- Median total hemorrhage volume error: {float(np.median(vol_err)):.4f} mL",
                f"- Max total hemorrhage volume error: {float(np.max(vol_err)):.4f} mL",
                "",
                "## Confidence Statistics",
                f"- Mean study confidence: {float(np.mean(confs)):.4f}",
                f"- Median study confidence: {float(np.median(confs)):.4f}",
                f"- Min study confidence: {float(np.min(confs)):.4f}",
                "",
                "## Inference Speed",
                f"- Mean seconds / case: "
                f"{(float(np.mean(valid_times)) if valid_times else float('nan')):.2f} "
                f"(timed cases={len(valid_times)})",
                f"- Cases timed: {len(valid_times)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- Metrics use the same shared modules as MONAI locked-test evaluation.",
            "- Volumes use native NIfTI spacing from BHSD ground-truth headers.",
            "- No retraining; checkpoint_best.pth reused as-is.",
        ]
    )
    (run_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_parameters(model_dir: Path, device: torch.device) -> int | None:
    """Load predictor network and count parameters."""
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=(0,),
            checkpoint_name="checkpoint_best.pth",
        )
        total = 0
        for net in predictor.network if isinstance(predictor.network, (list, tuple)) else [predictor.network]:
            total += sum(p.numel() for p in net.parameters())
        return int(total)
    except Exception as exc:  # noqa: BLE001 — research probe; report N/A on failure
        logger.warning("Could not count parameters: %s", exc)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked-test nnU-Net evaluation (research adapter).")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--images-ts", type=Path, default=DEFAULT_IMAGES_TS)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--use-mirroring",
        action="store_true",
        help="Enable nnU-Net test-time mirroring (slower). Default off for parity with MONAI (no TTA).",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Run prediction only (skip metric writing).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score existing predictions in output-dir/nnunet_preds (skip predict).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    device = torch.device(args.device)

    # Verify locked test split identity with nnU-Net imagesTs
    test_files = load_split_filenames("test")
    test_ids = {Path(name).name.replace(".nii.gz", "").replace(".nii", "") for name in test_files}
    ts_ids = {_case_id_from_nnunet_image(p) for p in args.images_ts.glob("*_0000.nii*")}
    if test_ids != ts_ids and args.case_limit is None:
        missing = sorted(test_ids - ts_ids)
        extra = sorted(ts_ids - test_ids)
        raise RuntimeError(
            f"imagesTs does not match locked test split. missing={missing[:5]} extra={extra[:5]}"
        )

    staging = args.output_dir / "_nnunet_model_staging"
    model_dir = prepare_nnunet_model_folder(args.checkpoint_dir, staging)
    pred_dir = args.output_dir / "nnunet_preds"

    # nnU-Net env vars (prediction may consult them)
    os.environ.setdefault("nnUNet_raw", str(PROJECT_ROOT / "data" / "nnunet" / "nnUNet_raw"))
    os.environ.setdefault(
        "nnUNet_preprocessed", str(PROJECT_ROOT / "data" / "nnunet" / "nnUNet_preprocessed")
    )
    os.environ.setdefault("nnUNet_results", str(staging.parent / "nnUNet_results"))

    param_count = count_parameters(model_dir, device)
    case_times: dict[str, float] = {}

    if not args.score_only:
        case_times = run_nnunet_prediction(
            model_dir,
            args.images_ts,
            pred_dir,
            device=device,
            use_mirroring=bool(args.use_mirroring),
            case_limit=args.case_limit,
        )
        timing_path = args.output_dir / "inference_times.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(case_times, indent=2), encoding="utf-8")
    else:
        timing_path = args.output_dir / "inference_times.json"
        if timing_path.is_file():
            case_times = json.loads(timing_path.read_text(encoding="utf-8"))

    if args.predictions_only:
        print(f"Predictions written under {pred_dir}")
        return

    # Score all predicted cases that have GT
    pred_cases = sorted(
        {
            p.name.replace(".nii.gz", "").replace(".nii", "").replace(".npz", "")
            for p in pred_dir.iterdir()
            if p.suffix in {".nii", ".gz", ".npz"} or p.name.endswith(".nii.gz")
        }
    )
    # Filter to segmentation niftis primarily
    seg_ids = []
    for p in sorted(pred_dir.glob("*.nii*")):
        if p.name.endswith("_0000.nii.gz"):
            continue
        seg_ids.append(p.name.replace(".nii.gz", "").replace(".nii", ""))
    if not seg_ids:
        raise RuntimeError(f"No segmentations found in {pred_dir}")

    accumulator = MetricAccumulator()
    scan_results: list[dict[str, Any]] = []
    for case_id in seg_ids:
        if case_id not in test_ids and args.case_limit is None:
            continue
        logger.info("Scoring %s", case_id)
        result = evaluate_case(
            case_id,
            pred_dir,
            args.labels_dir,
            args.images_dir,
            accumulator,
        )
        scan_results.append(result)

    global_metrics = _global_metrics(accumulator)
    failure = _failure_analysis(scan_results)
    environment = collect_environment_info(seed=42)
    write_artifacts(
        args.output_dir,
        global_metrics,
        scan_results,
        failure,
        checkpoint_path=args.checkpoint_dir / "checkpoint_best.pth",
        environment=environment,
        case_times=case_times,
        param_count=param_count,
        use_mirroring=bool(args.use_mirroring),
        device=str(device),
    )
    print(f"Wrote evaluation artifacts to {args.output_dir}")
    print(f"Macro Dice: {global_metrics['macro_dice']:.6f}")
    print(f"Scans evaluated: {len(scan_results)}")


if __name__ == "__main__":
    main()
