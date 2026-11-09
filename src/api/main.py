"""FastAPI application entrypoint with lifespan model caching."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes import router
from src.api.service import model_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load models into memory during app startup."""
    logger.info("Initializing application and loading ML models...")
    try:
        model_manager.load()
    except Exception as e:
        logger.error("Failed to load models at startup: %s", e)
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="Customer Churn Prediction Engine",
    description="Production-grade API serving churn predictions and SHAP local feature attributions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", tags=["Monitoring"])
def root():
    return {
        "service": "Customer Churn Prediction Engine",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
