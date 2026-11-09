"""ModelManager service for managing inference and SHAP explainability."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

from src.api.schemas import FeatureAttribution, PredictionResponse
from src.data.schema import CustomerPayload, ID_COLUMN

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton service to load and cache models and perform predictions with SHAP attribution."""

    def __init__(
        self,
        pipeline_path: str = "models/churn_pipeline.joblib",
        explainer_path: str = "models/shap_explainer.joblib",
        threshold: float = 0.45,
        model_version: str = "1.0.0",
    ):
        self.pipeline_path = Path(pipeline_path)
        self.explainer_path = Path(explainer_path)
        self.threshold = threshold
        self.model_version = model_version
        self.pipeline = None
        self.explainer = None
        self.model_type = "Unknown"

    def load(self) -> None:
        """Loads serialized pipeline and explainer into memory."""
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Model pipeline not found at {self.pipeline_path}")
        if not self.explainer_path.exists():
            raise FileNotFoundError(f"SHAP explainer not found at {self.explainer_path}")

        logger.info("Loading churn pipeline from %s", self.pipeline_path)
        self.pipeline = joblib.load(self.pipeline_path)

        classifier = self.pipeline.named_steps.get("classifier")
        self.model_type = type(classifier).__name__ if classifier else "Unknown"

        logger.info("Loading SHAP explainer from %s", self.explainer_path)
        self.explainer = joblib.load(self.explainer_path)
        logger.info("ModelManager successfully loaded %s (threshold=%.2f)", self.model_type, self.threshold)

    def is_loaded(self) -> bool:
        """Returns True if both pipeline and explainer are in memory."""
        return self.pipeline is not None and self.explainer is not None

    def predict(self, customer: CustomerPayload) -> PredictionResponse:
        """Runs inference and SHAP local attribution for an incoming customer payload."""
        if not self.is_loaded():
            raise RuntimeError("ModelManager is not initialized or models are not loaded.")

        data_dict = customer.model_dump()
        customer_id = data_dict.get("customerID") or "UNKNOWN"

        # Drop ID column if present to match training feature set
        feature_dict = {k: v for k, v in data_dict.items() if k != ID_COLUMN}

        # If TotalCharges is None or missing, populate with tenure * MonthlyCharges
        if feature_dict.get("TotalCharges") is None:
            feature_dict["TotalCharges"] = feature_dict["tenure"] * feature_dict["MonthlyCharges"]

        df_input = pd.DataFrame([feature_dict])

        # Inference
        probs = self.pipeline.predict_proba(df_input)
        churn_prob = float(probs[0, 1])
        prediction = 1 if churn_prob >= self.threshold else 0

        # SHAP attribution
        preprocessor = self.pipeline.named_steps["preprocessor"]
        transformed_input = preprocessor.transform(df_input)

        raw_attributions = self.explainer.explain_instance(transformed_input, top_k=3)
        top_features = [
            FeatureAttribution(
                feature=attr["feature"],
                attribution=attr["attribution"],
                impact=attr["impact"],
            )
            for attr in raw_attributions
        ]

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(churn_prob, 4),
            churn_prediction=prediction,
            threshold_used=self.threshold,
            top_driving_features=top_features,
        )


# Global instance for app lifecycle
model_manager = ModelManager()
