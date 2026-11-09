"""FastAPI router defining /health and /predict endpoints."""

from fastapi import APIRouter, HTTPException, Request, status
from src.api.schemas import CustomerPayload, HealthResponse, PredictionResponse
from src.api.service import model_manager
from src.core.exceptions import InferenceError, ModelNotLoadedError

router = APIRouter()


def get_active_model_manager(request: Request = None):
    """Retrieves model manager from app.state if injected, else uses default singleton."""
    if request and hasattr(request.app.state, "model_manager") and request.app.state.model_manager:
        return request.app.state.model_manager
    return model_manager


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check and model metadata",
    tags=["Monitoring"],
)
def get_health(request: Request) -> HealthResponse:
    """Returns the operational status and loaded model details."""
    manager = get_active_model_manager(request)
    if not manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable",
        )

    return HealthResponse(
        status="healthy",
        model_version=manager.model_version,
        model_type=manager.model_type,
        threshold=manager.threshold,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn and return SHAP top-3 driving features",
    tags=["Inference"],
)
def predict_churn(customer: CustomerPayload, request: Request) -> PredictionResponse:
    """Predicts probability of customer churn with SHAP local feature attribution."""
    manager = get_active_model_manager(request)
    try:
        return manager.predict(customer)
    except ModelNotLoadedError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(err),
        )
    except (InferenceError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )
