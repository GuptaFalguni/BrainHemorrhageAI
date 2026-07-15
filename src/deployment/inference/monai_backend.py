"""MONAI inference backend — wraps ``inference.InferencePipeline``."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

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
from inference.pipeline import InferencePipeline, preprocess_nifti_scan


class MonaiBackend(InferenceBackend):
    """Adapt the existing MONAI production pipeline to the unified interface."""

    def __init__(
        self,
        model_id: str,
        checkpoint_path: Path,
        *,
        device: str = "auto",
        model_config_path: Path | None = None,
        preprocessing_config_path: Path | None = None,
    ) -> None:
        self._model_id = model_id
        self._checkpoint_path = Path(checkpoint_path)
        self._device_arg = device
        self._model_config_path = model_config_path
        self._preprocessing_config_path = preprocessing_config_path or (
            PROJECT_ROOT / "config" / "preprocessing.yaml"
        )
        self._pipeline: InferencePipeline | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def framework(self) -> str:
        return "monai"

    def load_model(self) -> None:
        if self._pipeline is not None:
            return
        self._pipeline = InferencePipeline(
            checkpoint_path=self._checkpoint_path,
            model_config_path=self._model_config_path,
            preprocessing_config_path=self._preprocessing_config_path,
            device=self._device_arg,
        )

    def _require_pipeline(self) -> InferencePipeline:
        self.load_model()
        assert self._pipeline is not None
        return self._pipeline

    def predict_volume(self, ct_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        pipeline = self._require_pipeline()
        volume, _raw_hu = preprocess_nifti_scan(
            input_path=ct_path,
            mask_path=None,
            preprocessing_config_path=self._preprocessing_config_path,
        )
        buffer = pipeline._run_model(volume)
        from evaluation.evaluate_model import stack_scan_predictions

        prediction_volume, _gt = stack_scan_predictions(buffer)
        probability_volume = buffer.stack_volume(buffer.prob_slices, channel_first=True)
        geometry = {
            "spacing": volume.original_spacing,
            "shape": volume.original_shape,
            "affine": volume.affine,
            "raw_hu": _raw_hu,
            "scan_name": scan_output_name(Path(ct_path)),
        }
        return prediction_volume.astype(np.int64), probability_volume.astype(np.float32), geometry

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
            preprocessing_config_path=Path(self._preprocessing_config_path),
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
        pipeline = self._require_pipeline()
        scan_name = str(geometry.get("scan_name") or scan_output_name(ct_path))
        spacing = tuple(float(v) for v in geometry["spacing"])
        shape = tuple(int(v) for v in geometry["shape"])
        affine = np.asarray(geometry["affine"])

        artifacts = write_standard_artifacts(
            output_dir=output_dir,
            scan_name=scan_name,
            ct_path=ct_path,
            checkpoint_path=self._checkpoint_path,
            model_name=str(pipeline.model_config.get("model_name", "monai_2d_unet")),
            device=str(pipeline.device),
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
                if key.startswith("prediction_overlay") or key == "overlay"
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
                "device": str(pipeline.device),
                "processing_time_sec": processing_time_sec,
                "mask_path": str(mask_path) if mask_path else None,
                "shape": list(shape),
                "spacing": list(spacing),
            },
        )

    def predict(
        self,
        ct_path: Path,
        *,
        mask_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> UnifiedInferenceResult:
        """Full MONAI path via existing ``InferencePipeline.predict`` plus schema adapt."""
        start = time.perf_counter()
        pipeline = self._require_pipeline()
        ct_path = Path(ct_path)
        scan_name = scan_output_name(ct_path)
        out = Path(output_dir) if output_dir else artifact_output_dir(scan_name, self.model_id)

        # Prefer the full existing pipeline for identical MONAI artifacts.
        result = pipeline.predict(
            input_path=ct_path,
            mask_path=mask_path,
            output_dir=out,
        )

        volumes = {
            "total_volume_ml": float(result.total_volume_ml),
            "per_class_volume_ml": {int(k): float(v) for k, v in result.per_class_volume_ml.items()},
            "classes_present": list(result.classes_present),
            "spacing": tuple(float(v) for v in result.spacing),
        }
        confidence = {
            "study_confidence": float(result.study_confidence),
            "per_class_confidence": {int(k): float(v) for k, v in result.class_confidences.items()},
            "mean_softmax_confidence": float(result.mean_softmax_confidence),
            "confidence_histogram": result.confidence_histogram,
        }

        artifacts = dict(result.artifact_paths)
        pred_path = artifacts.get("prediction_nifti") or (out / "prediction.nii.gz")
        mask_alias = out / "prediction_mask.nii.gz"
        if pred_path.is_file() and not mask_alias.is_file():
            import shutil

            shutil.copy2(pred_path, mask_alias)
        artifacts["prediction_mask"] = mask_alias
        artifacts["prediction"] = Path(pred_path)
        overlay_src = artifacts.get("prediction_overlay") or (out / "prediction_overlay.png")
        overlay_alias = out / "overlay.png"
        if Path(overlay_src).is_file() and not overlay_alias.is_file():
            import shutil

            shutil.copy2(overlay_src, overlay_alias)
        artifacts["overlay"] = overlay_alias

        processing_time = float(result.processing_time_sec)
        if processing_time <= 0:
            processing_time = time.perf_counter() - start

        overlay_paths = {
            key: Path(path)
            for key, path in artifacts.items()
            if "overlay" in key
        }

        return UnifiedInferenceResult(
            model_id=self.model_id,
            framework=self.framework,
            prediction_mask=result.prediction_volume.astype(np.int64),
            overlay_paths=overlay_paths,
            volumes=volumes,
            confidence=confidence,
            artifacts={key: Path(path) for key, path in artifacts.items()},
            report_path=Path(artifacts.get("prediction_report", out / "prediction_report.md")),
            metadata={
                "scan_name": result.scan_name,
                "checkpoint_path": str(self._checkpoint_path),
                "output_dir": str(out),
                "device": result.device,
                "processing_time_sec": processing_time,
                "mask_path": str(mask_path) if mask_path else None,
                "shape": list(result.shape),
                "spacing": list(result.spacing),
            },
        )
