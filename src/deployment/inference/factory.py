"""Factory that resolves ``model_id`` → concrete inference backend."""

from __future__ import annotations

from pathlib import Path

from deployment.inference.base import InferenceBackend
from deployment.inference.monai_backend import MonaiBackend
from deployment.inference.nnunet_backend import NnUNetBackend
from deployment.model_registry import ModelRegistry
from deployment.registry import load_deployment_registries


class InferenceFactory:
    """Create the correct backend for a deployable ``model_id``."""

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        *,
        project_root: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        if model_registry is None:
            _experiments, model_registry = load_deployment_registries(project_root=project_root)
        self._models = model_registry
        self._device = device
        self._cache: dict[str, InferenceBackend] = {}

    @property
    def model_registry(self) -> ModelRegistry:
        return self._models

    def resolve_model_id(self, model_id: str | None = None) -> str:
        if model_id is None or not str(model_id).strip():
            return self._models.get_default().model_id
        return self._models.get(model_id).model_id

    def create(self, model_id: str | None = None) -> InferenceBackend:
        resolved = self.resolve_model_id(model_id)
        if resolved in self._cache:
            return self._cache[resolved]

        model = self._models.get(resolved)
        info = self._models.deployment_info(resolved)
        checkpoint = Path(info["checkpoint_path"])
        framework = str(model.framework).lower()

        if framework == "monai":
            backend: InferenceBackend = MonaiBackend(
                model_id=resolved,
                checkpoint_path=checkpoint,
                device=self._device or "auto",
            )
        elif framework == "nnunet":
            folds: tuple[int, ...] = (0, 1) if resolved == "nnunet_ssl_2fold" else (0,)
            backend = NnUNetBackend(
                model_id=resolved,
                checkpoint_path=checkpoint,
                device=self._device,
                use_folds=folds,
            )
        else:
            raise ValueError(f"Unsupported framework '{framework}' for model '{resolved}'.")

        self._cache[resolved] = backend
        return backend

    def __call__(self, model_id: str | None = None) -> InferenceBackend:
        return self.create(model_id)
