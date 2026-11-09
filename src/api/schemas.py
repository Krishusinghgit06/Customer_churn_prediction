"""Pydantic schemas for FastAPI request and response validation."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from src.data.schema import CustomerPayload


class FeatureAttribution(BaseModel):
    """Local SHAP feature attribution contribution."""
    feature: str = Field(..., description="Name of the contributing feature")
    attribution: float = Field(..., description="Shapley attribution value impact")
    impact: str = Field(..., description="Direction of impact ('Increases churn risk' or 'Decreases churn risk')")


class PredictionResponse(BaseModel):
    """Schema for customer churn prediction and explainability response."""
    customer_id: str = Field(..., description="Customer identifier")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of churning")
    churn_prediction: int = Field(..., ge=0, le=1, description="Binary decision flag (1 = Churned, 0 = Retained)")
    threshold_used: float = Field(..., description="Decision threshold applied to generate churn_prediction")
    top_driving_features: List[FeatureAttribution] = Field(..., description="Top 3 driving features explaining the prediction")


class HealthResponse(BaseModel):
    """Schema for service health status."""
    status: str = Field(..., description="Overall service status (e.g. 'healthy')")
    model_version: str = Field(..., description="Semantic version of the deployed model pipeline")
    model_type: str = Field(..., description="Underlying model classifier type")
    threshold: float = Field(..., description="Active churn decision threshold")
