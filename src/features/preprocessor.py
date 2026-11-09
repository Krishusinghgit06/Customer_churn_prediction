"""Preprocessing pipeline builder using Scikit-Learn ColumnTransformer."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to coerce TotalCharges column to numeric float, converting blanks to NaN."""

    def __init__(self, column_name: str = "TotalCharges"):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            if self.column_name in X_copy.columns:
                X_copy[self.column_name] = pd.to_numeric(
                    X_copy[self.column_name].replace(r"^\s*$", np.nan, regex=True),
                    errors="coerce",
                )
            return X_copy
        elif isinstance(X, np.ndarray):
            # If passed as 2D numpy array, attempt numeric conversion
            cleaned = []
            for val in X.ravel():
                try:
                    if str(val).strip() == "" or val is None:
                        cleaned.append(np.nan)
                    else:
                        cleaned.append(float(val))
                except (ValueError, TypeError):
                    cleaned.append(np.nan)
            return np.array(cleaned, dtype=float).reshape(-1, 1)
        return X


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Builds a leak-free Scikit-learn ColumnTransformer for numerical and categorical columns.

    - Numerical: TotalCharges coercion -> SimpleImputer(median) -> StandardScaler
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(ignore unknown, dense)
    """
    num_cols = numerical_features or NUMERICAL_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("cleaner", TotalChargesCleaner()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-interpretable feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for manual reconstruction if get_feature_names_out fails
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                if hasattr(trans.named_steps["onehot"], "get_feature_names_out"):
                    feature_names.extend(trans.named_steps["onehot"].get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)
        return feature_names
