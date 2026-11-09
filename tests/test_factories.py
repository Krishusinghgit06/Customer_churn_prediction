"""Tests for Factory Method implementations across the application."""

import pytest
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.api.app import create_app
from src.api.service import ModelManager, ModelServiceFactory
from src.data.loader_factory import (
    DataLoaderFactory,
    LocalCsvDataLoader,
    SyntheticDataLoader,
)
from src.features.preprocessor import PreprocessorFactory
from src.modeling.factory import ModelFactory, PipelineFactory


class DummyCustomClassifier(BaseEstimator):
    """Dummy classifier for factory registration tests."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return [0] * len(X)


class TestModelFactory:
    """Test suite for ModelFactory and PipelineFactory."""

    def test_create_xgboost_classifier(self):
        clf = ModelFactory.create_classifier("xgboost", max_depth=5)
        assert isinstance(clf, XGBClassifier)
        assert clf.max_depth == 5

    def test_create_logistic_regression(self):
        clf = ModelFactory.create_classifier("logistic_regression", max_iter=500)
        assert isinstance(clf, LogisticRegression)
        assert clf.max_iter == 500

    def test_unknown_classifier_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            ModelFactory.create_classifier("non_existent_model")

    def test_register_custom_classifier(self):
        ModelFactory.register_classifier("custom_dummy", DummyCustomClassifier)
        clf = ModelFactory.create_classifier("custom_dummy")
        assert isinstance(clf, DummyCustomClassifier)

    def test_pipeline_factory_assembly(self):
        pipeline = PipelineFactory.create_pipeline(model_type="logistic_regression")
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


class TestPreprocessorFactory:
    """Test suite for PreprocessorFactory."""

    def test_create_preprocessor(self):
        preprocessor = PreprocessorFactory.create_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names

    def test_create_sub_pipelines(self):
        num_pipe = PreprocessorFactory.create_numeric_pipeline()
        cat_pipe = PreprocessorFactory.create_categorical_pipeline()
        assert isinstance(num_pipe, Pipeline)
        assert isinstance(cat_pipe, Pipeline)
        assert "cleaner" in num_pipe.named_steps
        assert "onehot" in cat_pipe.named_steps


class TestDataLoaderFactory:
    """Test suite for DataLoaderFactory."""

    def test_create_synthetic_loader(self):
        loader = DataLoaderFactory.create_loader(source_type="synthetic", num_samples=10)
        assert isinstance(loader, SyntheticDataLoader)
        df = loader.load_data()
        assert len(df) == 10
        assert "Churn" in df.columns

    def test_create_local_loader(self):
        loader = DataLoaderFactory.create_loader(
            source_type="local",
            file_path="data/raw/telco_customer_churn.csv",
        )
        assert isinstance(loader, LocalCsvDataLoader)

    def test_unknown_source_type_raises(self):
        with pytest.raises(ValueError, match="Unknown data loader"):
            DataLoaderFactory.create_loader(source_type="unknown_source")


class TestApplicationFactory:
    """Test suite for FastAPI application factory create_app and ModelServiceFactory."""

    def test_create_service(self):
        service = ModelServiceFactory.create_service(threshold=0.50)
        assert isinstance(service, ModelManager)
        assert service.threshold == 0.50

    def test_create_app_instance(self):
        custom_app = create_app()
        with TestClient(custom_app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert "service" in data
            assert data["health"] == "/health"
