"""Factory method pattern for machine learning models and pipelines."""

import logging
from typing import Any, Callable, Dict, Optional, Type
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessor import PreprocessorFactory

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for instantiating machine learning classifiers with standardized configurations."""

    _registry: Dict[str, Type[BaseEstimator]] = {
        "xgboost": XGBClassifier,
        "xgb": XGBClassifier,
        "logistic_regression": LogisticRegression,
        "lr": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "rf": RandomForestClassifier,
    }

    _default_params: Dict[str, Dict[str, Any]] = {
        "xgboost": {
            "n_estimators": 75,
            "max_depth": 3,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "eval_metric": "logloss",
            "random_state": 42,
        },
        "logistic_regression": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 6,
            "class_weight": "balanced",
            "random_state": 42,
        },
    }

    @classmethod
    def register_classifier(
        cls,
        name: str,
        classifier_class: Type[BaseEstimator],
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registers a new model classifier in the factory."""
        key = name.lower()
        cls._registry[key] = classifier_class
        if default_params:
            cls._default_params[key] = default_params

    @classmethod
    def create_classifier(cls, model_type: str, **kwargs: Any) -> BaseEstimator:
        """Factory method to construct a classifier instance with merged parameters."""
        key = model_type.lower()
        if key not in cls._registry:
            supported = list(cls._registry.keys())
            raise ValueError(f"Unknown model_type '{model_type}'. Supported models: {supported}")

        classifier_class = cls._registry[key]
        canonical_key = "xgboost" if key in ("xgboost", "xgb") else (
            "logistic_regression" if key in ("logistic_regression", "lr") else key
        )
        params = cls._default_params.get(canonical_key, {}).copy()
        params.update(kwargs)

        logger.info("Instantiating classifier '%s' (%s) with params: %s", key, classifier_class.__name__, params)
        return classifier_class(**params)


class PipelineFactory:
    """Factory for constructing full end-to-end scikit-learn Pipelines."""

    @classmethod
    def create_pipeline(
        cls,
        model_type: str = "xgboost",
        preprocessor: Optional[ColumnTransformer] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Builds a complete Scikit-Learn Pipeline combining Preprocessor and Classifier."""
        # 1. Obtain or create Preprocessor
        transformer = preprocessor or PreprocessorFactory.create_preprocessor()

        # 2. Obtain Classifier from ModelFactory
        kwargs = model_kwargs or {}
        classifier = ModelFactory.create_classifier(model_type, **kwargs)

        # 3. Assemble and return Pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("classifier", classifier),
            ]
        )
        logger.info("Assembled Pipeline: preprocessor -> %s", type(classifier).__name__)
        return pipeline

