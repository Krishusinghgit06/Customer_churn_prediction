"""Unit tests for Scikit-learn preprocessing pipeline and leakage prevention."""

import numpy as np
import pandas as pd
import pytest
from src.features.preprocessor import (
    TotalChargesCleaner,
    build_preprocessor,
    get_feature_names_from_preprocessor,
)


def test_total_charges_cleaner():
    """Verify TotalChargesCleaner converts blank strings and whitespace to NaN."""
    cleaner = TotalChargesCleaner()
    df = pd.DataFrame({
        "TotalCharges": ["100.5", " ", "", "  250.0 ", "invalid"],
    })
    cleaned = cleaner.transform(df)
    assert np.isnan(cleaned.iloc[1]["TotalCharges"])
    assert np.isnan(cleaned.iloc[2]["TotalCharges"])
    assert cleaned.iloc[0]["TotalCharges"] == 100.5
    assert cleaned.iloc[3]["TotalCharges"] == 250.0
    assert np.isnan(cleaned.iloc[4]["TotalCharges"])


def test_preprocessor_handles_missing_values(high_risk_customer):
    """Verify ColumnTransformer pipeline handles missing or None fields without errors."""
    preprocessor = build_preprocessor()

    # Create dummy training data
    df_train = pd.DataFrame([high_risk_customer, high_risk_customer]).drop(columns=["customerID"], errors="ignore")
    preprocessor.fit(df_train)

    # Test inference payload with missing TotalCharges and None values
    df_test = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,
        "TotalCharges": None,  # Missing
    }])

    transformed = preprocessor.transform(df_test)
    assert not np.isnan(transformed).any(), "Transformed output must not contain NaNs"
    assert transformed.shape[0] == 1


def test_preprocessor_handles_unseen_categories(high_risk_customer):
    """Verify OneHotEncoder handles unknown categories gracefully without crashing."""
    preprocessor = build_preprocessor()

    df_train = pd.DataFrame([high_risk_customer]).drop(columns=["customerID"], errors="ignore")
    preprocessor.fit(df_train)

    df_unseen = df_train.copy()
    df_unseen["InternetService"] = "Satellite-6G"  # Brand new unseen category

    transformed = preprocessor.transform(df_unseen)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_feature_names_retrieval(high_risk_customer):
    """Verify feature names extraction generates non-empty strings matching transformed dimensions."""
    preprocessor = build_preprocessor()
    df = pd.DataFrame([high_risk_customer, high_risk_customer]).drop(columns=["customerID"], errors="ignore")
    transformed = preprocessor.fit_transform(df)

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    assert len(feature_names) == transformed.shape[1]
    assert all(isinstance(name, str) for name in feature_names)
    assert "tenure" in feature_names
    assert "MonthlyCharges" in feature_names
