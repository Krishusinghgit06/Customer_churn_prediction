"""Dataset acquisition module with remote download and realistic synthetic fallback.

Uses DataLoaderFactory under the hood while preserving backward compatibility.
"""

import logging
from typing import List, Optional
import pandas as pd

from src.data.loader_factory import DataLoaderFactory, SyntheticDataLoader

logger = logging.getLogger(__name__)

DEFAULT_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/churn.csv",
]


def generate_synthetic_telco_data(num_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic Telco Customer Churn data with exact schema and distributions."""
    loader = SyntheticDataLoader(num_samples=num_samples, random_state=random_state)
    return loader.load_data()


def fetch_telco_data(
    save_path: str = "data/raw/telco_customer_churn.csv",
    remote_urls: Optional[List[str]] = None,
    num_samples: int = 7043,
    random_state: int = 42,
) -> pd.DataFrame:
    """Acquires the Telco Customer Churn dataset using DataLoaderFactory."""
    urls = remote_urls or DEFAULT_URLS
    loader = DataLoaderFactory.create_loader(
        source_type="auto",
        file_path=save_path,
        remote_urls=urls,
        num_samples=num_samples,
        random_state=random_state,
    )
    return loader.load_data()
