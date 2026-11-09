"""Model training, benchmarking, and artifact serialization pipeline using Factories."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.core.config import get_settings
from src.data.dataset_schema import ID_COLUMN, TARGET_COLUMN
from src.data.loader_factory import DataLoaderFactory
from src.features.preprocessor import PreprocessorFactory, get_feature_names_from_preprocessor
from src.modeling.evaluate import compute_metrics
from src.modeling.explain import ChurnExplainer
from src.modeling.factory import PipelineFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepares stratified train and test splits, isolating target and dropping ID."""
    df_clean = df.copy()

    # Target mapping: 'Yes' / '1' -> 1, 'No' / '0' -> 0
    target_series = df_clean[TARGET_COLUMN].astype(str).str.strip().str.lower()
    if target_series.isin(["yes", "no"]).any():
        y = (target_series == "yes").astype(int).to_numpy()
    else:
        y = pd.to_numeric(df_clean[TARGET_COLUMN]).astype(int).to_numpy()

    # Drop ID and target from features
    drop_cols = [col for col in [ID_COLUMN, TARGET_COLUMN] if col in df_clean.columns]
    X = df_clean.drop(columns=drop_cols)

    logger.info("Total samples: %d, Churn rate: %.2f%%", len(y), float(y.mean() * 100))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info("Train samples: %d, Test samples: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_and_evaluate(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Executes end-to-end model training, comparison, SHAP generation, and serialization."""
    settings = get_settings(config_path)

    # 1. Ingest Data via DataLoaderFactory
    loader = DataLoaderFactory.create_loader(
        source_type="auto",
        file_path=settings.dataset.raw_data_path,
        remote_urls=settings.dataset.download_urls,
        random_state=settings.dataset.random_state,
    )
    df = loader.load_data()

    # 2. Split Data (Strictly before preprocessor fitting)
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        test_size=settings.dataset.test_size,
        random_state=settings.dataset.random_state,
    )

    threshold = settings.model.decision_threshold
    results: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # 3. Model 1: Baseline Logistic Regression via PipelineFactory
    # -------------------------------------------------------------
    logger.info("Training Baseline Model: Logistic Regression (balanced) via PipelineFactory...")
    lr_pipeline = PipelineFactory.create_pipeline(
        model_type="logistic_regression",
        model_kwargs={
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": settings.dataset.random_state,
        },
    )
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = compute_metrics(y_test, lr_probs, threshold=threshold)
    logger.info(
        "Logistic Regression: ROC-AUC=%.4f, Recall(%.2f)=%.4f",
        results["logistic_regression"]["roc_auc"],
        threshold,
        results["logistic_regression"]["recall"],
    )

    # -------------------------------------------------------------
    # 4. Model 2: XGBoost Classifier via PipelineFactory
    # -------------------------------------------------------------
    logger.info("Training Challenger Model: XGBoost Classifier via PipelineFactory...")
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / max(pos_count, 1))
    logger.info("Computed XGBoost scale_pos_weight: %.2f", scale_pos_weight)

    xgb_pipeline = PipelineFactory.create_pipeline(
        model_type="xgboost",
        model_kwargs={
            "n_estimators": 75,
            "max_depth": 3,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": scale_pos_weight,
            "random_state": settings.dataset.random_state,
            "eval_metric": "logloss",
        },
    )
    xgb_pipeline.fit(X_train, y_train)
    xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
    results["xgboost"] = compute_metrics(y_test, xgb_probs, threshold=threshold)
    logger.info(
        "XGBoost: ROC-AUC=%.4f, Recall(%.2f)=%.4f",
        results["xgboost"]["roc_auc"],
        threshold,
        results["xgboost"]["recall"],
    )

    # -------------------------------------------------------------
    # 5. Model Selection (Prioritize ROC-AUC & Recall)
    # -------------------------------------------------------------
    if results["xgboost"]["roc_auc"] >= results["logistic_regression"]["roc_auc"] - 0.005:
        winning_model_name = "xgboost"
        winning_pipeline = xgb_pipeline
    else:
        winning_model_name = "logistic_regression"
        winning_pipeline = lr_pipeline

    logger.info("Selected winning model: %s", winning_model_name)

    # -------------------------------------------------------------
    # 6. SHAP Explainer & Global Plot
    # -------------------------------------------------------------
    fitted_preprocessor = winning_pipeline.named_steps["preprocessor"]
    fitted_estimator = winning_pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(fitted_preprocessor)

    X_train_trans = fitted_preprocessor.transform(X_train)
    X_test_trans = fitted_preprocessor.transform(X_test)

    explainer = ChurnExplainer.fit_explainer(
        model=fitted_estimator,
        X_train_transformed=X_train_trans,
        feature_names=feature_names,
        background_samples=100,
    )

    plot_paths = [
        settings.model.shap_plot_path,
        "reports/figures/shap_summary.png",
    ]
    for p in plot_paths:
        explainer.generate_global_summary_plot(X_test_trans[:200], output_path=p)

    # -------------------------------------------------------------
    # 7. Serialization of Artifacts (Dual location for backward compatibility)
    # -------------------------------------------------------------
    pipeline_targets = [
        Path(settings.model.output_pipeline_path),
        Path(settings.model.legacy_pipeline_path),
    ]
    for pt in pipeline_targets:
        pt.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(winning_pipeline, pt)
        logger.info("Exported winning pipeline to %s", pt)

    explainer_targets = [
        Path(settings.model.output_explainer_path),
        Path(settings.model.legacy_explainer_path),
    ]
    for et in explainer_targets:
        et.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(explainer, et)
        logger.info("Exported SHAP explainer to %s", et)

    metrics_payload = {
        "winning_model": winning_model_name,
        "decision_threshold": threshold,
        "models": results,
    }
    metrics_targets = [
        Path(settings.model.output_metrics_path),
        Path(settings.model.legacy_metrics_path),
    ]
    for mt in metrics_targets:
        mt.parent.mkdir(parents=True, exist_ok=True)
        with open(mt, "w") as f:
            json.dump(metrics_payload, f, indent=2)
        logger.info("Exported benchmark metrics to %s", mt)

    return metrics_payload


if __name__ == "__main__":
    metrics = train_and_evaluate()
    print("\n=== FINAL BENCHMARK METRICS ===")
    print(json.dumps(metrics, indent=2))

