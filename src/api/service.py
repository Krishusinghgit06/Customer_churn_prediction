"""ModelManager service for managing inference and SHAP explainability with Factory Method."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

from src.api.schemas import CustomerPayload, FeatureAttribution, PredictionResponse
from src.core.config import get_settings
from src.core.exceptions import InferenceError, ModelNotFoundError, ModelNotLoadedError
from src.data.dataset_schema import ID_COLUMN

logger = logging.getLogger(__name__)


class ModelManager:
    """Service to load and cache models and perform predictions with SHAP attribution."""

    def __init__(
        self,
        pipeline_path: Optional[str] = None,
        explainer_path: Optional[str] = None,
        threshold: float = 0.45,
        model_version: str = "1.0.0",
    ):
        settings = get_settings()
        self.pipeline_path = self._resolve_path(
            primary=pipeline_path,
            candidates=[
                settings.model.output_pipeline_path,
                settings.model.legacy_pipeline_path,
                "artifacts/models/churn_pipeline.joblib",
                "models/churn_pipeline.joblib",
            ],
        )
        self.explainer_path = self._resolve_path(
            primary=explainer_path,
            candidates=[
                settings.model.output_explainer_path,
                settings.model.legacy_explainer_path,
                "artifacts/models/shap_explainer.joblib",
                "models/shap_explainer.joblib",
            ],
        )
        self.threshold = threshold
        self.model_version = model_version
        self.pipeline = None
        self.explainer = None
        self.model_type = "Unknown"

    @staticmethod
    def _resolve_path(primary: Optional[str], candidates: List[str]) -> Path:
        if primary:
            return Path(primary)
        for c in candidates:
            p = Path(c)
            if p.exists():
                return p
        return Path(candidates[0])

    def load(self) -> None:
        """Loads serialized pipeline and explainer into memory."""
        if not self.pipeline_path.exists():
            raise ModelNotFoundError(f"Model pipeline not found at {self.pipeline_path}")
        if not self.explainer_path.exists():
            raise ModelNotFoundError(f"SHAP explainer not found at {self.explainer_path}")

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
            raise ModelNotLoadedError("ModelManager is not initialized or models are not loaded.")

        try:
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
        except Exception as exc:
            logger.error("Inference execution failed: %s", exc)
            raise InferenceError(f"Inference computation failed: {str(exc)}") from exc


class ModelServiceFactory:
    """Factory for creating configured ModelManager instances."""

    @classmethod
    def create_service(
        cls,
        pipeline_path: Optional[str] = None,
        explainer_path: Optional[str] = None,
        threshold: Optional[float] = None,
        model_version: Optional[str] = None,
    ) -> ModelManager:
        """Constructs a ModelManager with explicit parameters or fallback to configuration."""
        settings = get_settings()
        th = threshold if threshold is not None else settings.model.decision_threshold
        version = model_version or settings.app_version

        manager = ModelManager(
            pipeline_path=pipeline_path,
            explainer_path=explainer_path,
            threshold=th,
            model_version=version,
        )
        return manager


# Default singleton instance for app lifecycle
model_manager = ModelServiceFactory.create_service()
