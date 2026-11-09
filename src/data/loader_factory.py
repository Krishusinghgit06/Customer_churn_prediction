"""Factory pattern implementation for data loaders and dataset ingestion."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type
import numpy as np
import pandas as pd
import requests

from src.core.exceptions import DataIngestionError
from src.data.dataset_schema import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for customer churn data loaders."""

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Loads and returns the telco churn dataset as a pandas DataFrame."""
        pass


class LocalCsvDataLoader(BaseDataLoader):
    """Loads dataset from a local CSV file path."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise DataIngestionError(f"Local dataset file not found at: {self.file_path}")
        logger.info("Loading dataset from local path: %s", self.file_path)
        try:
            df = pd.read_csv(self.file_path)
            logger.info("Successfully loaded %d records from %s", len(df), self.file_path)
            return df
        except Exception as exc:
            raise DataIngestionError(f"Failed to read CSV at {self.file_path}: {exc}") from exc


class RemoteCsvDataLoader(BaseDataLoader):
    """Downloads dataset from remote URLs with automatic failover."""

    def __init__(self, urls: List[str], save_path: Optional[str] = None):
        self.urls = urls
        self.save_path = Path(save_path) if save_path else None

    def load_data(self) -> pd.DataFrame:
        for url in self.urls:
            try:
                logger.info("Attempting to download dataset from: %s", url)
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    if self._validate_schema(df):
                        logger.info("Successfully fetched %d records from remote: %s", len(df), url)
                        if self.save_path:
                            self.save_path.parent.mkdir(parents=True, exist_ok=True)
                            df.to_csv(self.save_path, index=False)
                            logger.info("Saved remote dataset locally to %s", self.save_path)
                        return df
            except Exception as e:
                logger.warning("Download failed for %s: %s", url, e)

        raise DataIngestionError("All remote dataset URLs failed to provide valid data.")

    def _validate_schema(self, df: pd.DataFrame) -> bool:
        required = [TARGET_COLUMN, "tenure", "MonthlyCharges"]
        return all(col in df.columns for col in required)


class SyntheticDataLoader(BaseDataLoader):
    """Generates realistic synthetic Telco customer churn dataset."""

    def __init__(self, num_samples: int = 7043, random_state: int = 42, save_path: Optional[str] = None):
        self.num_samples = num_samples
        self.random_state = random_state
        self.save_path = Path(save_path) if save_path else None

    def load_data(self) -> pd.DataFrame:
        logger.info("Generating %d synthetic Telco records (random_state=%d)...", self.num_samples, self.random_state)
        np.random.seed(self.random_state)

        customer_ids = [
            f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}"
            for _ in range(self.num_samples)
        ]

        gender = np.random.choice(["Male", "Female"], size=self.num_samples, p=[0.505, 0.495])
        senior_citizen = np.random.choice([0, 1], size=self.num_samples, p=[0.838, 0.162])
        partner = np.random.choice(["Yes", "No"], size=self.num_samples, p=[0.483, 0.517])

        dependents = []
        for p in partner:
            if p == "Yes":
                dependents.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            else:
                dependents.append(np.random.choice(["Yes", "No"], p=[0.10, 0.90]))

        contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=self.num_samples, p=[0.55, 0.21, 0.24])
        tenures = []
        for c in contracts:
            if c == "Month-to-month":
                t = int(np.random.exponential(scale=14))
                tenures.append(min(max(t, 0), 72))
            elif c == "One year":
                t = int(np.random.normal(loc=35, scale=15))
                tenures.append(min(max(t, 1), 72))
            else:
                t = int(np.random.normal(loc=56, scale=12))
                tenures.append(min(max(t, 1), 72))

        phone_service = np.random.choice(["Yes", "No"], size=self.num_samples, p=[0.903, 0.097])
        multiple_lines = []
        for ps in phone_service:
            if ps == "No":
                multiple_lines.append("No phone service")
            else:
                multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))

        internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=self.num_samples, p=[0.34, 0.44, 0.22])

        def add_on(service: str, yes_prob: float):
            if service == "No":
                return "No internet service"
            return np.random.choice(["Yes", "No"], p=[yes_prob, 1 - yes_prob])

        online_security = [add_on(s, 0.35) for s in internet_service]
        online_backup = [add_on(s, 0.40) for s in internet_service]
        device_protection = [add_on(s, 0.38) for s in internet_service]
        tech_support = [add_on(s, 0.34) for s in internet_service]
        streaming_tv = [add_on(s, 0.45) for s in internet_service]
        streaming_movies = [add_on(s, 0.46) for s in internet_service]

        paperless_billing = np.random.choice(["Yes", "No"], size=self.num_samples, p=[0.59, 0.41])
        payment_method = np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=self.num_samples,
            p=[0.33, 0.23, 0.22, 0.22],
        )

        monthly_charges = []
        total_charges = []
        for i in range(self.num_samples):
            base = 20.0
            if phone_service[i] == "Yes":
                base += 5.0
            if multiple_lines[i] == "Yes":
                base += 10.0
            if internet_service[i] == "DSL":
                base += 25.0
            elif internet_service[i] == "Fiber optic":
                base += 50.0

            if online_security[i] == "Yes":
                base += 8.0
            if online_backup[i] == "Yes":
                base += 8.0
            if device_protection[i] == "Yes":
                base += 8.0
            if tech_support[i] == "Yes":
                base += 8.0
            if streaming_tv[i] == "Yes":
                base += 10.0
            if streaming_movies[i] == "Yes":
                base += 10.0

            mc = base + np.random.uniform(-3.0, 3.0)
            mc = round(max(mc, 18.25), 2)
            monthly_charges.append(mc)

            t = tenures[i]
            if t == 0:
                total_charges.append(" ")
            else:
                tc = round(mc * t + np.random.uniform(-10.0, 10.0), 2)
                total_charges.append(str(max(tc, mc)))

        # Churn probability based on risk factors
        churn = []
        for i in range(self.num_samples):
            score = 0.0
            if contracts[i] == "Month-to-month":
                score += 1.3
            elif contracts[i] == "One year":
                score -= 0.6
            else:
                score -= 1.4

            if tenures[i] <= 12:
                score += 0.9
            elif tenures[i] >= 48:
                score -= 0.9

            if internet_service[i] == "Fiber optic":
                score += 0.7
            if tech_support[i] == "No" and internet_service[i] != "No":
                score += 0.6
            if payment_method[i] == "Electronic check":
                score += 0.5
            if online_security[i] == "No" and internet_service[i] != "No":
                score += 0.4
            if senior_citizen[i] == 1:
                score += 0.2

            prob = 1.0 / (1.0 + np.exp(-(score - 0.4)))
            churn.append("Yes" if np.random.rand() < prob else "No")

        data = {
            ID_COLUMN: customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenures,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contracts,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            TARGET_COLUMN: churn,
        }

        df = pd.DataFrame(data)
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.save_path, index=False)
            logger.info("Saved synthetic dataset to %s", self.save_path)

        return df


class DataLoaderFactory:
    """Factory method class for constructing data loader instances."""

    _loaders: Dict[str, Type[BaseDataLoader]] = {
        "local": LocalCsvDataLoader,
        "remote": RemoteCsvDataLoader,
        "synthetic": SyntheticDataLoader,
    }

    @classmethod
    def register_loader(cls, source_type: str, loader_class: Type[BaseDataLoader]) -> None:
        """Extensibility hook to register custom data loaders."""
        cls._loaders[source_type.lower()] = loader_class

    @classmethod
    def create_loader(
        cls,
        source_type: str = "auto",
        file_path: Optional[str] = None,
        remote_urls: Optional[List[str]] = None,
        num_samples: int = 7043,
        random_state: int = 42,
    ) -> BaseDataLoader:
        """Factory method returning the appropriate BaseDataLoader implementation."""
        st = source_type.lower()
        if st == "local":
            if not file_path:
                raise ValueError("file_path must be specified for 'local' data loader")
            return LocalCsvDataLoader(file_path=file_path)

        elif st == "remote":
            urls = remote_urls or [
                "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            ]
            return RemoteCsvDataLoader(urls=urls, save_path=file_path)

        elif st == "synthetic":
            return SyntheticDataLoader(
                num_samples=num_samples,
                random_state=random_state,
                save_path=file_path,
            )

        elif st == "auto":
            # Auto-detection strategy:
            # 1. Check local path if valid CSV exists
            if file_path and Path(file_path).exists() and Path(file_path).stat().st_size > 100:
                logger.info("Auto-loader selected: LocalCsvDataLoader (%s)", file_path)
                return LocalCsvDataLoader(file_path=file_path)

            # 2. Return a composite/fallback loader that attempts remote then synthetic
            return AutoFallbackDataLoader(
                file_path=file_path,
                remote_urls=remote_urls,
                num_samples=num_samples,
                random_state=random_state,
            )

        elif st in cls._loaders:
            loader_cls = cls._loaders[st]
            return loader_cls(file_path=file_path)  # type: ignore

        raise ValueError(f"Unknown data loader source_type: '{source_type}'. Supported: {list(cls._loaders.keys())} or 'auto'")


class AutoFallbackDataLoader(BaseDataLoader):
    """Composite loader that tries local -> remote -> synthetic fallback."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        remote_urls: Optional[List[str]] = None,
        num_samples: int = 7043,
        random_state: int = 42,
    ):
        self.file_path = file_path
        self.remote_urls = remote_urls
        self.num_samples = num_samples
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        if self.file_path and Path(self.file_path).exists() and Path(self.file_path).stat().st_size > 100:
            try:
                return LocalCsvDataLoader(self.file_path).load_data()
            except Exception as e:
                logger.warning("Local load failed (%s), attempting remote...", e)

        if self.remote_urls:
            try:
                return RemoteCsvDataLoader(self.remote_urls, save_path=self.file_path).load_data()
            except Exception as e:
                logger.warning("Remote download failed (%s), falling back to synthetic...", e)

        return SyntheticDataLoader(
            num_samples=self.num_samples,
            random_state=self.random_state,
            save_path=self.file_path,
        ).load_data()

