"""Health endpoint."""

from __future__ import annotations

import sys

from fastapi import APIRouter, Depends

from api.v1.dependencies import get_model_registry
from api.v1.schemas.response import HealthResponse
from deployment.model_registry import ModelRegistry

router = APIRouter(tags=["health"])

API_VERSION = "1.0.0"


def _torch_info() -> tuple[str, str, bool]:
    import torch

    gpu = bool(torch.cuda.is_available())
    device = "cuda" if gpu else "cpu"
    return torch.__version__, device, gpu


def _monai_version() -> str | None:
    try:
        import monai

        return str(monai.__version__)
    except Exception:  # noqa: BLE001 — optional reporting only
        return None


@router.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_model_registry)) -> HealthResponse:
    pytorch_version, device, gpu_available = _torch_info()
    available = sorted(registry.records.keys())
    default_id = registry.get_default().model_id if registry.records else None
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        available_models=available,
        default_model_id=default_id,
        device=device,
        gpu_available=gpu_available,
        python_version=sys.version.split()[0],
        pytorch_version=pytorch_version,
        monai_version=_monai_version(),
        registry_loaded=True,
    )


__all__ = ["API_VERSION", "router"]
