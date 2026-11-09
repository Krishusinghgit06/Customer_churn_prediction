"""Data ingestion, loaders, and dataset schema definitions."""

from src.data.dataset_schema import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.data.fetch_data import fetch_telco_data, generate_synthetic_telco_data
from src.data.loader_factory import (
    BaseDataLoader,
    DataLoaderFactory,
    LocalCsvDataLoader,
    RemoteCsvDataLoader,
    SyntheticDataLoader,
)

__all__ = [
    "ID_COLUMN",
    "TARGET_COLUMN",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_FEATURE_COLUMNS",
    "BaseDataLoader",
    "DataLoaderFactory",
    "LocalCsvDataLoader",
    "RemoteCsvDataLoader",
    "SyntheticDataLoader",
    "fetch_telco_data",
    "generate_synthetic_telco_data",
]
