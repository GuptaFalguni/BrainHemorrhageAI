"""Models registry endpoint — read-only."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.v1.dependencies import get_model_registry
from api.v1.schemas.response import (
    ModelCapabilitiesResponse,
    ModelResponse,
    ModelsListResponse,
)
from deployment.model_registry import ModelRecord, ModelRegistry

router = APIRouter(tags=["models"])


def _to_model_response(record: ModelRecord) -> ModelResponse:
    meta = record.metadata()
    caps = meta["capabilities"]
    return ModelResponse(
        model_id=meta["model_id"],
        display_name=meta["display_name"],
        experiment_id=meta["experiment_id"],
        category=meta["category"],
        framework=meta["framework"],
        version=meta["version"],
        default=bool(meta["default"]),
        description=meta.get("description") or "",
        paper=meta.get("paper") or "",
        training_dataset=meta.get("training_dataset") or "",
        macro_dice=meta.get("macro_dice"),
        volume_error_mean_ml=meta.get("volume_error_mean_ml"),
        speed=meta.get("speed") or "",
        limitations=meta.get("limitations") or "",
        recommended_use=meta.get("recommended_use") or "",
        model_card=meta.get("model_card") or "",
        capabilities=ModelCapabilitiesResponse(**caps),
    )


@router.get("/models", response_model=ModelsListResponse)
def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> ModelsListResponse:
    models = [
        _to_model_response(registry.records[model_id])
        for model_id in sorted(registry.records.keys())
    ]
    default_id = registry.get_default().model_id
    return ModelsListResponse(default_model_id=default_id, models=models)


__all__ = ["router"]
