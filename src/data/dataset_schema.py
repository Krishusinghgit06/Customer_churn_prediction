"""Dataset schema definitions and feature constants for Telco Customer Churn."""

from typing import List

ID_COLUMN: str = "customerID"
TARGET_COLUMN: str = "Churn"

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

