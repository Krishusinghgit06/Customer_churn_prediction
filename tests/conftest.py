"""Shared pytest fixtures for Customer Churn Engine test suite."""

from typing import Dict, Any
import joblib
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import model_manager


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with initialized lifespan models."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def trained_pipeline():
    """Loads trained inference pipeline."""
    return joblib.load("models/churn_pipeline.joblib")


@pytest.fixture(scope="session")
def trained_explainer():
    """Loads trained SHAP explainer."""
    return joblib.load("models/shap_explainer.joblib")


@pytest.fixture
def high_risk_customer() -> Dict[str, Any]:
    """Sample payload representing a high churn risk customer."""
    return {
        "customerID": "9999-RISK1",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 95.50,
    }


@pytest.fixture
def low_risk_customer() -> Dict[str, Any]:
    """Sample payload representing a loyal, low churn risk customer."""
    return {
        "customerID": "0001-LOYAL",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.20,
        "TotalCharges": 2938.00,
    }
