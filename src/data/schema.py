"""Schema definitions and feature constants for Telco Customer Churn."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class CustomerPayload(BaseModel):
    """Pydantic model representing incoming customer data for inference."""
    customerID: Optional[str] = Field(default="UNKNOWN", description="Unique customer identifier")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal[0, 1] = Field(..., description="Whether customer is a senior citizen (1, 0)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed with company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Whether customer has phone service")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Multiple lines status")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet service provider")
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["No internet service", "No", "Yes"] = Field(..., description="Online backup add-on")
    DeviceProtection: Literal["No internet service", "No", "Yes"] = Field(..., description="Device protection add-on")
    TechSupport: Literal["No internet service", "No", "Yes"] = Field(..., description="Tech support add-on")
    StreamingTV: Literal["No internet service", "No", "Yes"] = Field(..., description="Streaming TV add-on")
    StreamingMovies: Literal["No internet service", "No", "Yes"] = Field(..., description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing status")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges amount")
    TotalCharges: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total charges over lifetime. If None, imputed from tenure * MonthlyCharges",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }
