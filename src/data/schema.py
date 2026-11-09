"""Legacy schema bridge preserving backward compatibility.

NOTE: For new code, import tabular constants from `src.data.dataset_schema`
and API payload models from `src.api.schemas`.
"""

from src.api.schemas import CustomerPayload
from src.data.dataset_schema import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)

__all__ = [
    "ID_COLUMN",
    "TARGET_COLUMN",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_FEATURE_COLUMNS",
    "CustomerPayload",
]
