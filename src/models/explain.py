"""Explainability module using SHAP for global summary plots and local feature attribution."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class ChurnExplainer:
    """Wrapper around SHAP explainer to provide global plots and local JSON attributions."""

    def __init__(self, explainer: Any, feature_names: List[str]):
        self.explainer = explainer
        self.feature_names = feature_names

    @classmethod
    def fit_explainer(
        cls,
        model: Any,
        X_train_transformed: np.ndarray,
        feature_names: List[str],
        background_samples: int = 100,
    ) -> "ChurnExplainer":
        """Initializes and calibrates SHAP explainer on tree model."""
        np.random.seed(42)
        n_samples = min(background_samples, X_train_transformed.shape[0])
        background_indices = np.random.choice(X_train_transformed.shape[0], n_samples, replace=False)
        background = X_train_transformed[background_indices]

        logger.info("Initializing SHAP TreeExplainer for model...")
        try:
            # TreeExplainer using tree structure
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            logger.info("TreeExplainer without data failed (%s), trying with background...", e)
            try:
                explainer = shap.TreeExplainer(model, data=background)
            except Exception as e2:
                logger.warning("TreeExplainer failed (%s). Falling back to shap.Explainer.", e2)
                explainer = shap.Explainer(model, background)

        return cls(explainer=explainer, feature_names=feature_names)

    def generate_global_summary_plot(
        self,
        X_test_transformed: np.ndarray,
        output_path: str = "reports/figures/shap_summary.png",
        max_display: int = 15,
    ) -> str:
        """Generates and saves SHAP beeswarm/summary plot on test sample."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Computing SHAP values for global summary plot (%d samples)...", len(X_test_transformed))
        shap_values = self.explainer(X_test_transformed)

        # Handle multi-dimensional output if binary classification returns 2 slices
        if len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
            values_to_plot = shap_values[:, :, 1]
        else:
            values_to_plot = shap_values

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            values_to_plot.values,
            features=X_test_transformed,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary plot to %s", out_file)
        return str(out_file)

    def explain_instance(
        self,
        x_single_transformed: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generates local feature attribution for a single transformed customer vector.

        Returns top_k features sorted by absolute SHAP impact on positive churn prediction.
        """
        # Ensure 2D shape (1, num_features)
        if x_single_transformed.ndim == 1:
            x_single_transformed = x_single_transformed.reshape(1, -1)

        raw_shap = self.explainer(x_single_transformed)

        # Extract values for positive class (Churn = 1)
        if len(raw_shap.shape) == 3 and raw_shap.shape[-1] == 2:
            vals = raw_shap.values[0, :, 1]
        elif len(raw_shap.shape) == 2:
            vals = raw_shap.values[0, :]
        else:
            vals = np.array(raw_shap.values).ravel()

        # Sort indices by absolute contribution
        sorted_indices = np.argsort(np.abs(vals))[::-1][:top_k]

        attributions = []
        for idx in sorted_indices:
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            attr_val = float(vals[idx])
            impact = "Increases churn risk" if attr_val > 0 else "Decreases churn risk"

            attributions.append({
                "feature": feat_name,
                "attribution": round(attr_val, 4),
                "impact": impact,
            })

        return attributions
