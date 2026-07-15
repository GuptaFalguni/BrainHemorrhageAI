"""Request schemas for API v1."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Optional predict fields (multipart also accepts the CT upload)."""

    model_id: str | None = Field(
        default=None,
        description="Deployable model_id from the model registry. Omit for default.",
    )


__all__ = ["PredictRequest"]
