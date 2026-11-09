"""Configuration loader and settings management."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Dataset configuration parameters."""
    raw_data_path: str = "data/raw/telco_customer_churn.csv"
    download_urls: List[str] = Field(default_factory=lambda: [
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/churn.csv",
    ])
    test_size: float = 0.20
    random_state: int = 42
    target_column: str = "Churn"
    id_column: str = "customerID"


class ModelArtifactConfig(BaseModel):
    """Paths and threshold for model artifacts."""
    decision_threshold: float = 0.45
    output_pipeline_path: str = "artifacts/models/churn_pipeline.joblib"
    output_explainer_path: str = "artifacts/models/shap_explainer.joblib"
    output_metrics_path: str = "artifacts/metrics/metrics.json"
    shap_plot_path: str = "artifacts/reports/shap_summary.png"
    # Fallback paths for root models/ directory compatibility
    legacy_pipeline_path: str = "models/churn_pipeline.joblib"
    legacy_explainer_path: str = "models/shap_explainer.joblib"
    legacy_metrics_path: str = "models/metrics.json"


class FeatureConfig(BaseModel):
    """Categorical and numerical feature definitions."""
    numerical: List[str] = Field(default_factory=lambda: [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ])
    categorical: List[str] = Field(default_factory=lambda: [
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
    ])


class Settings(BaseModel):
    """Global application settings."""
    app_name: str = "Customer Churn Prediction Engine"
    app_version: str = "1.0.0"
    api_prefix: str = ""
    debug: bool = False
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelArtifactConfig = Field(default_factory=ModelArtifactConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)


def load_yaml_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads raw YAML configuration file if it exists."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return data or {}


@lru_cache()
def get_settings(config_path: Optional[str] = None) -> Settings:
    """Factory function to instantiate cached Settings object from config.yaml and environment."""
    resolved_path = config_path or os.getenv("CONFIG_PATH", "config/config.yaml")
    raw_dict = load_yaml_config(resolved_path)

    # Initialize Settings with parsed yaml dictionary
    return Settings(
        dataset=DatasetConfig(**raw_dict.get("dataset", {})),
        model=ModelArtifactConfig(**raw_dict.get("model", {})),
        features=FeatureConfig(**raw_dict.get("features", {})),
    )

