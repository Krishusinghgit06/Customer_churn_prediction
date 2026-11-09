"""FastAPI application factory for production deployment and testing."""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.service import ModelManager, model_manager as default_model_manager
from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(
    manager: Optional[ModelManager] = None,
    settings: Optional[Settings] = None,
) -> FastAPI:
    """Application factory method creating and configuring a FastAPI instance."""
    app_settings = settings or get_settings()
    active_manager = manager or default_model_manager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifecycle manager to load models into memory during app startup."""
        logger.info("Initializing application and loading ML models...")
        try:
            active_manager.load()
        except Exception as e:
            logger.error("Failed to load models at startup: %s", e)
        yield
        logger.info("Application shutting down...")

    app = FastAPI(
        title=app_settings.app_name,
        description="Production-grade API serving churn predictions and SHAP local feature attributions.",
        version=app_settings.app_version,
        lifespan=lifespan,
    )

    # Attach model manager to app state for dependency injection
    app.state.model_manager = active_manager
    app.state.settings = app_settings

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(router)

    @app.get("/", tags=["Monitoring"])
    def root():
        return {
            "service": app_settings.app_name,
            "version": app_settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
        }

    return app

