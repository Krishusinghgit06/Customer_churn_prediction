"""FastAPI router defining /health and /predict endpoints."""

from fastapi import APIRouter, HTTPException, status
from src.api.schemas import HealthResponse, PredictionResponse
from src.api.service import model_manager
from src.data.schema import CustomerPayload

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check and model metadata",
    tags=["Monitoring"],
)
def get_health() -> HealthResponse:
    """Returns the operational status and loaded model details."""
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable",
        )

    return HealthResponse(
        status="healthy",
        model_version=model_manager.model_version,
        model_type=model_manager.model_type,
        threshold=model_manager.threshold,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn and return SHAP top-3 driving features",
    tags=["Inference"],
)
def predict_churn(customer: CustomerPayload) -> PredictionResponse:
    """Predicts probability of customer churn with SHAP local feature attribution."""
    try:
        return model_manager.predict(customer)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )
