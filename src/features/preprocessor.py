"""Preprocessing pipeline builder using Scikit-Learn ColumnTransformer and Factory Method."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.dataset_schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


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

    def get_feature_names_out(self, input_features=None):
        """Pass through feature names for Scikit-learn Pipeline compatibility."""
        if input_features is None:
            return np.array([self.column_name], dtype=object)
        return np.asarray(input_features, dtype=object)


class PreprocessorFactory:
    """Factory for building feature transformation pipelines and ColumnTransformers."""

    @staticmethod
    def create_numeric_pipeline(
        cleaner_column: str = "TotalCharges",
        impute_strategy: str = "median",
    ) -> Pipeline:
        """Constructs preprocessing pipeline for numerical features."""
        return Pipeline(
            steps=[
                ("cleaner", TotalChargesCleaner(column_name=cleaner_column)),
                ("imputer", SimpleImputer(strategy=impute_strategy)),
                ("scaler", StandardScaler()),
            ]
        )

    @staticmethod
    def create_categorical_pipeline(
        impute_strategy: str = "most_frequent",
        handle_unknown: str = "ignore",
    ) -> Pipeline:
        """Constructs preprocessing pipeline for categorical features."""
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=impute_strategy)),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown=handle_unknown,
                        sparse_output=False,
                    ),
                ),
            ]
        )

    @classmethod
    def create_preprocessor(
        cls,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> ColumnTransformer:
        """Builds a leak-free Scikit-learn ColumnTransformer for numerical and categorical columns.

        - Numerical: TotalChargesCleaner -> SimpleImputer(median) -> StandardScaler
        - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(ignore unknown, dense)
        """
        num_cols = numerical_features or NUMERICAL_FEATURES
        cat_cols = categorical_features or CATEGORICAL_FEATURES

        numeric_transformer = cls.create_numeric_pipeline()
        categorical_transformer = cls.create_categorical_pipeline()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        return preprocessor


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Backward-compatible entry point delegating to PreprocessorFactory."""
    return PreprocessorFactory.create_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-interpretable feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                try:
                    if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                        feature_names.extend(trans.named_steps["onehot"].get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)
                except Exception:
                    feature_names.extend(cols)
            else:
                try:
                    feature_names.extend(trans.get_feature_names_out(cols))
                except Exception:
                    feature_names.extend(cols)
        return feature_names
