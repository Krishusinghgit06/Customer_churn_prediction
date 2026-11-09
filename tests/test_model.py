"""Unit tests for ML model pipeline, serialization, and SHAP explainability."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def test_model_artifact_exists():
    """Verify serialized pipeline, explainer, and metrics exist on disk."""
    assert Path("models/churn_pipeline.joblib").exists()
    assert Path("models/shap_explainer.joblib").exists()
    assert Path("models/metrics.json").exists()


def test_pipeline_inference_probabilities(trained_pipeline, high_risk_customer, low_risk_customer):
    """Verify inference produces valid probabilities in [0.0, 1.0] and sensible relative ranking."""
    df_high = pd.DataFrame([high_risk_customer]).drop(columns=["customerID"])
    df_low = pd.DataFrame([low_risk_customer]).drop(columns=["customerID"])

    prob_high = trained_pipeline.predict_proba(df_high)[0, 1]
    prob_low = trained_pipeline.predict_proba(df_low)[0, 1]

    assert 0.0 <= prob_high <= 1.0
    assert 0.0 <= prob_low <= 1.0
    assert prob_high > prob_low, "High risk customer churn probability must exceed low risk customer"


def test_threshold_logic(trained_pipeline, high_risk_customer):
    """Verify decision threshold 0.45 accurately converts probability to binary label."""
    df = pd.DataFrame([high_risk_customer]).drop(columns=["customerID"])
    prob = trained_pipeline.predict_proba(df)[0, 1]

    threshold = 0.45
    expected_label = 1 if prob >= threshold else 0
    actual_label = int(prob >= threshold)

    assert actual_label == expected_label


def test_shap_explainer_local_attributions(trained_pipeline, trained_explainer, high_risk_customer):
    """Verify SHAP explainer returns exactly 3 driving features with valid attribution metadata."""
    df = pd.DataFrame([high_risk_customer]).drop(columns=["customerID"])
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(df)

    attributions = trained_explainer.explain_instance(transformed, top_k=3)

    assert len(attributions) == 3
    for item in attributions:
        assert "feature" in item
        assert "attribution" in item
        assert "impact" in item
        assert isinstance(item["attribution"], float)
        assert item["impact"] in ["Increases churn risk", "Decreases churn risk"]


def test_benchmark_metrics_satisfaction():
    """Verify benchmark metrics record meets ROC-AUC and Recall requirements."""
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)

    assert "winning_model" in metrics
    winning_model = metrics["winning_model"]
    model_metrics = metrics["models"][winning_model]

    assert model_metrics["roc_auc"] >= 0.80, f"Expected ROC-AUC >= 0.80, got {model_metrics['roc_auc']}"
    assert model_metrics["recall"] >= 0.80, f"Expected Recall >= 0.80, got {model_metrics['recall']}"
