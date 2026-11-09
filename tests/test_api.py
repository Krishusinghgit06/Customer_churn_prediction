"""Integration tests for FastAPI endpoints (/health and /predict)."""

import pytest
from fastapi import status


def test_root_endpoint(client):
    """Verify root endpoint returns service info and status 200."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "service" in data
    assert data["docs"] == "/docs"


def test_health_endpoint(client):
    """Verify GET /health returns 200 OK and model metadata."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["threshold"] == 0.45
    assert data["model_version"] == "1.0.0"
    assert "model_type" in data
    assert data["model_type"] in ["XGBClassifier", "LogisticRegression"]


def test_predict_endpoint_high_risk(client, high_risk_customer):
    """Verify POST /predict returns 200 OK, probability, binary churn prediction, and 3 SHAP features."""
    response = client.post("/predict", json=high_risk_customer)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["customer_id"] == high_risk_customer["customerID"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["threshold_used"] == 0.45

    top_features = data["top_driving_features"]
    assert len(top_features) == 3
    for feat in top_features:
        assert "feature" in feat
        assert "attribution" in feat
        assert "impact" in feat
        assert isinstance(feat["attribution"], float)
        assert feat["impact"] in ["Increases churn risk", "Decreases churn risk"]


def test_predict_endpoint_low_risk(client, low_risk_customer):
    """Verify POST /predict on low-risk customer yields low probability and prediction 0."""
    response = client.post("/predict", json=low_risk_customer)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["customer_id"] == low_risk_customer["customerID"]
    assert data["churn_prediction"] == 0
    assert data["churn_probability"] < 0.45


def test_predict_endpoint_validation_error(client):
    """Verify POST /predict with invalid data returns 422 Unprocessable Entity."""
    # Invalid tenure (negative), invalid gender enum, missing required fields
    malformed_payload = {
        "customerID": "INVALID-01",
        "gender": "UnknownGender",
        "tenure": -5,
        "MonthlyCharges": -10.0,
    }
    response = client.post("/predict", json=malformed_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
