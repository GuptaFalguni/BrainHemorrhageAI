"""Abstract inference backend and unified result schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class UnifiedInferenceResult:
    """Identical schema returned by every deployable backend."""

    model_id: str
    framework: str
    prediction_mask: np.ndarray
    overlay_paths: dict[str, Path]
    volumes: dict[str, Any]
    confidence: dict[str, Any]
    artifacts: dict[str, Path]
    report_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def schema_fingerprint(self) -> dict[str, Any]:
        """Return a backend-agnostic description for validation."""
        return {
            "keys": sorted(
                [
                    "model_id",
                    "framework",
                    "prediction_mask",
                    "overlay_paths",
                    "volumes",
                    "confidence",
                    "artifacts",
                    "report_path",
                    "metadata",
                ]
            ),
            "volume_keys": sorted(self.volumes.keys()),
            "confidence_keys": sorted(self.confidence.keys()),
            "required_artifacts": sorted(
                {
                    "prediction_mask",
                    "overlay",
                    "volume_report",
                    "confidence",
                    "prediction_summary",
                    "prediction_report",
                }
                & set(self.artifacts)
            ),
            "mask_ndim": int(self.prediction_mask.ndim),
            "mask_dtype": str(self.prediction_mask.dtype),
        }


class InferenceBackend(ABC):
    """Common interface every deployed model backend must implement."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable deployable model identifier."""

    @property
    @abstractmethod
    def framework(self) -> str:
        """Framework name (``monai`` or ``nnunet``)."""

    @abstractmethod
    def load_model(self) -> None:
        """Load weights / predictor into memory."""

    @abstractmethod
    def predict(
        self,
        ct_path: Path,
        *,
        mask_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> UnifiedInferenceResult:
        """Run full inference and write standardized artifacts."""

    @abstractmethod
    def predict_volume(self, ct_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return ``(prediction_mask, probability_volume, geometry_meta)``."""

    @abstractmethod
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
        """Write overlay figures and return paths."""

    @abstractmethod
    def calculate_volumes(
        self,
        prediction_mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> dict[str, Any]:
        """Compute native-space volume statistics."""

    @abstractmethod
    def calculate_confidence(
        self,
        prediction_mask: np.ndarray,
        probability_volume: np.ndarray,
    ) -> dict[str, Any]:
        """Compute study / class confidence statistics."""

    @abstractmethod
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
        """Persist artifacts and build ``UnifiedInferenceResult``."""
