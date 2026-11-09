"""FastAPI service layer, schemas, routes, and application factory."""

from src.api.app import create_app
from src.api.schemas import (
    CustomerPayload,
    FeatureAttribution,
    HealthResponse,
    PredictionResponse,
)
from src.api.service import ModelManager, ModelServiceFactory, model_manager

__all__ = [
    "create_app",
    "CustomerPayload",
    "FeatureAttribution",
    "HealthResponse",
    "PredictionResponse",
    "ModelManager",
    "ModelServiceFactory",
    "model_manager",
]
