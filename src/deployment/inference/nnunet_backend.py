"""nnU-Net inference backend — reuses evaluation prediction helpers."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from deployment.inference.base import InferenceBackend, UnifiedInferenceResult
from deployment.inference.common import (
    PROJECT_ROOT,
    artifact_output_dir,
    compute_confidence,
    compute_volumes,
    largest_hemorrhage_slice,
    render_overlays,
    scan_output_name,
    write_standard_artifacts,
)
from evaluation.evaluate_nnunet import (
    _align_probabilities,
    _load_probabilities,
    _load_segmentation,
    prepare_nnunet_model_folder,
    run_nnunet_prediction,
)


class NnUNetBackend(InferenceBackend):
    """Adapt existing nnU-Net prediction utilities to the unified interface."""

    def __init__(
        self,
        model_id: str,
        checkpoint_path: Path,
        *,
        device: str | None = None,
        use_mirroring: bool = False,
        staging_root: Path | None = None,
        use_folds: tuple[int, ...] = (0,),
    ) -> None:
        self._model_id = model_id
        self._checkpoint_path = Path(checkpoint_path)
        self._checkpoint_dir = self._checkpoint_path.parent
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._use_mirroring = bool(use_mirroring)
        self._use_folds = tuple(int(f) for f in use_folds) or (0,)
        self._staging_root = Path(staging_root) if staging_root else (
            PROJECT_ROOT / "reports" / "inference" / "_nnunet_staging" / model_id
        )
        self._model_dir: Path | None = None
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def framework(self) -> str:
        return "nnunet"

    def _ensure_sidecars(self) -> None:
        """Copy dataset/plans JSON from config/models/<model_id> when missing."""
        src = PROJECT_ROOT / "config" / "models" / self._model_id
        if not src.is_dir():
            return
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name in ("dataset.json", "nnUNetPlans.json", "plans.json"):
            candidate = src / name
            target = self._checkpoint_dir / name
            if candidate.is_file() and not target.is_file():
                shutil.copy2(candidate, target)

    def load_model(self) -> None:
        if self._loaded and self._model_dir is not None:
            return
        self._ensure_sidecars()
        os.environ.setdefault("nnUNet_raw", str(PROJECT_ROOT / "data" / "nnunet" / "nnUNet_raw"))
        os.environ.setdefault(
            "nnUNet_preprocessed",
            str(PROJECT_ROOT / "data" / "nnunet" / "nnUNet_preprocessed"),
        )
        os.environ.setdefault(
            "nnUNet_results",
            str(self._staging_root.parent / "nnUNet_results"),
        )
        self._model_dir = prepare_nnunet_model_folder(
            self._checkpoint_dir,
            self._staging_root,
            use_folds=self._use_folds,
        )
        self._loaded = True

    def predict_volume(self, ct_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self.load_model()
        assert self._model_dir is not None

        ct_path = Path(ct_path)
        scan_name = scan_output_name(ct_path)
        work = self._staging_root / "single_case" / scan_name
        images_ts = work / "imagesTs"
        pred_dir = work / "preds"
        if work.exists():
            shutil.rmtree(work)
        images_ts.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        staged = images_ts / f"{scan_name}_0000.nii.gz"
        if ct_path.resolve() != staged.resolve():
            shutil.copy2(ct_path, staged)

        run_nnunet_prediction(
            self._model_dir,
            images_ts,
            pred_dir,
            device=self._device,
            use_mirroring=self._use_mirroring,
            case_limit=None,
            use_folds=self._use_folds,
        )

        prediction, affine = _load_segmentation(pred_dir, scan_name)
        probabilities = _load_probabilities(pred_dir, scan_name)
        if probabilities is None:
            eye = np.eye(6, dtype=np.float32)
            probabilities = np.moveaxis(eye[prediction], -1, 0)
        else:
            probabilities = _align_probabilities(probabilities, prediction)

        image_nii = nib.load(str(ct_path))
        raw_hu = image_nii.get_fdata().astype(np.float32)
        spacing = tuple(float(v) for v in image_nii.header.get_zooms()[:3])
        shape = tuple(int(v) for v in prediction.shape)

        geometry = {
            "spacing": spacing,
            "shape": shape,
            "affine": affine,
            "raw_hu": raw_hu,
            "scan_name": scan_name,
            "pred_dir": pred_dir,
        }
        return prediction.astype(np.int64), probabilities.astype(np.float32), geometry

    def calculate_volumes(
        self,
        prediction_mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> dict[str, Any]:
        return compute_volumes(prediction_mask, spacing)

    def calculate_confidence(
        self,
        prediction_mask: np.ndarray,
        probability_volume: np.ndarray,
    ) -> dict[str, Any]:
        return compute_confidence(prediction_mask, probability_volume)

    def generate_overlay(
        self,
        raw_hu: np.ndarray,
        prediction_mask: np.ndarray,
        *,
        spacing: tuple[float, float, float],
        output_dir: Path,
        ground_truth: np.ndarray | None = None,
        largest_slice: int | None = None,
    ) -> dict[str, Path]:
        return render_overlays(
            raw_hu=raw_hu,
            prediction_mask=prediction_mask,
            spacing=spacing,
            output_dir=output_dir,
            ground_truth=ground_truth,
            largest_slice=largest_slice,
        )

    def generate_outputs(
        self,
        *,
        ct_path: Path,
        output_dir: Path,
        prediction_mask: np.ndarray,
        probability_volume: np.ndarray,
        geometry: dict[str, Any],
        volumes: dict[str, Any],
        confidence: dict[str, Any],
        overlay_paths: dict[str, Path],
        processing_time_sec: float,
        mask_path: Path | None = None,
    ) -> UnifiedInferenceResult:
        scan_name = str(geometry.get("scan_name") or scan_output_name(ct_path))
        spacing = tuple(float(v) for v in geometry["spacing"])
        shape = tuple(int(v) for v in geometry["shape"])
        affine = np.asarray(geometry["affine"])

        artifacts = write_standard_artifacts(
            output_dir=output_dir,
            scan_name=scan_name,
            ct_path=ct_path,
            checkpoint_path=self._checkpoint_path,
            model_name="nnunet_3d_fullres",
            device=str(self._device),
            processing_time_sec=processing_time_sec,
            prediction_mask=prediction_mask,
            probability_volume=probability_volume,
            affine=affine,
            shape=shape,
            spacing=spacing,
            volumes=volumes,
            confidence=confidence,
            overlay_paths=overlay_paths,
        )
        return UnifiedInferenceResult(
            model_id=self.model_id,
            framework=self.framework,
            prediction_mask=prediction_mask,
            overlay_paths={
                key: path
                for key, path in overlay_paths.items()
                if "overlay" in key
            }
            or ({"overlay": artifacts["overlay"]} if "overlay" in artifacts else {}),
            volumes=volumes,
            confidence=confidence,
            artifacts=artifacts,
            report_path=artifacts["prediction_report"],
            metadata={
                "scan_name": scan_name,
                "checkpoint_path": str(self._checkpoint_path),
                "output_dir": str(output_dir),
                "device": str(self._device),
                "processing_time_sec": processing_time_sec,
                "mask_path": str(mask_path) if mask_path else None,
                "shape": list(shape),
                "spacing": list(spacing),
                "use_mirroring": self._use_mirroring,
                "use_folds": list(self._use_folds),
            },
        )

    def predict(
        self,
        ct_path: Path,
        *,
        mask_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> UnifiedInferenceResult:
        start = time.perf_counter()
        ct_path = Path(ct_path)
        scan_name = scan_output_name(ct_path)
        out = Path(output_dir) if output_dir else artifact_output_dir(scan_name, self.model_id)

        prediction_mask, probability_volume, geometry = self.predict_volume(ct_path)
        spacing = tuple(float(v) for v in geometry["spacing"])
        volumes = self.calculate_volumes(prediction_mask, spacing)
        confidence = self.calculate_confidence(prediction_mask, probability_volume)

        ground_truth = None
        if mask_path is not None:
            ground_truth = nib.load(str(mask_path)).get_fdata().astype(np.int64)

        overlay_paths = self.generate_overlay(
            geometry["raw_hu"],
            prediction_mask,
            spacing=spacing,
            output_dir=out,
            ground_truth=ground_truth,
            largest_slice=largest_hemorrhage_slice(prediction_mask),
        )
        elapsed = time.perf_counter() - start
        return self.generate_outputs(
            ct_path=ct_path,
            output_dir=out,
            prediction_mask=prediction_mask,
            probability_volume=probability_volume,
            geometry=geometry,
            volumes=volumes,
            confidence=confidence,
            overlay_paths=overlay_paths,
            processing_time_sec=elapsed,
            mask_path=mask_path,
        )
