"""Unified inference pipeline — single entry point for all deployable models."""

from __future__ import annotations

from pathlib import Path

from deployment.inference.base import UnifiedInferenceResult
from deployment.inference.common import artifact_output_dir, scan_output_name
from deployment.inference.factory import InferenceFactory
from deployment.model_registry import ModelRegistry


class UnifiedInferencePipeline:
    """Registry → factory → backend → standardized ``UnifiedInferenceResult``."""

    def __init__(
        self,
        *,
        factory: InferenceFactory | None = None,
        model_registry: ModelRegistry | None = None,
        project_root: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        self._factory = factory or InferenceFactory(
            model_registry=model_registry,
            project_root=project_root,
            device=device,
        )

    @property
    def factory(self) -> InferenceFactory:
        return self._factory

    def run(
        self,
        ct_path: Path | str,
        *,
        model_id: str | None = None,
        mask_path: Path | str | None = None,
        output_dir: Path | str | None = None,
    ) -> UnifiedInferenceResult:
        """Execute inference for ``model_id`` (default interactive model if omitted)."""
        ct_path = Path(ct_path)
        if not ct_path.is_file():
            raise FileNotFoundError(f"CT scan not found: {ct_path}")

        resolved_id = self._factory.resolve_model_id(model_id)
        backend = self._factory.create(resolved_id)
        backend.load_model()

        scan_name = scan_output_name(ct_path)
        out = Path(output_dir) if output_dir else artifact_output_dir(scan_name, resolved_id)
        return backend.predict(
            ct_path,
            mask_path=Path(mask_path) if mask_path else None,
            output_dir=out,
        )
