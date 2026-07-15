"""FastAPI application for BrainHemorrhageAI API v1."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.routers import health, inference, models, reports
from api.v1.routers.health import API_VERSION


def create_app() -> FastAPI:
    """Create the versioned production API (orchestration only)."""
    application = FastAPI(
        title="Brain Hemorrhage AI API",
        description=(
            "Production REST API v1. Orchestrates UnifiedInferencePipeline and "
            "ClinicalPipeline. Research Software — Not for Clinical Use."
        ),
        version=API_VERSION,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    application.include_router(health.router, prefix=prefix)
    application.include_router(models.router, prefix=prefix)
    application.include_router(inference.router, prefix=prefix)
    application.include_router(reports.router, prefix=prefix)
    return application


app = create_app()
