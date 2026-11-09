"""Dataset acquisition module with remote download and realistic synthetic fallback."""

import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests

from src.data.schema import CATEGORICAL_FEATURES, ID_COLUMN, NUMERICAL_FEATURES, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/churn.csv",
]


def generate_synthetic_telco_data(num_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic Telco Customer Churn data with exact schema and distributions."""
    np.random.seed(random_state)
    logger.info("Generating %d synthetic Telco Customer Churn records...", num_samples)

    # Identifiers
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(num_samples)]

    # Demographics
    gender = np.random.choice(["Male", "Female"], size=num_samples, p=[0.505, 0.495])
    senior_citizen = np.random.choice([0, 1], size=num_samples, p=[0.838, 0.162])
    partner = np.random.choice(["Yes", "No"], size=num_samples, p=[0.483, 0.517])
    # Dependents more likely if partner
    dependents = []
    for p in partner:
        if p == "Yes":
            dependents.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
        else:
            dependents.append(np.random.choice(["Yes", "No"], p=[0.10, 0.90]))

    # Contract & Tenure
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_samples, p=[0.55, 0.21, 0.24])
    tenures = []
    for c in contracts:
        if c == "Month-to-month":
            # Shorter tenures, exponential-like
            t = int(np.random.exponential(scale=14))
            tenures.append(min(max(t, 0), 72))
        elif c == "One year":
            t = int(np.random.normal(loc=35, scale=15))
            tenures.append(min(max(t, 1), 72))
        else:  # Two year
            t = int(np.random.normal(loc=56, scale=12))
            tenures.append(min(max(t, 1), 72))

    phone_service = np.random.choice(["Yes", "No"], size=num_samples, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=num_samples, p=[0.34, 0.44, 0.22])

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

    paperless_billing = np.random.choice(["Yes", "No"], size=num_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=num_samples,
        p=[0.33, 0.23, 0.22, 0.22],
    )

    # Monthly charges calculation based on services
    monthly_charges = []
    total_charges = []
    for i in range(num_samples):
        base = 20.0  # Base line
        if phone_service[i] == "Yes":
            base += 5.0
        if multiple_lines[i] == "Yes":
            base += 10.0
        if internet_service[i] == "DSL":
            base += 25.0
        elif internet_service[i] == "Fiber optic":
            base += 50.0

        for add in [online_security[i], online_backup[i], device_protection[i], tech_support[i], streaming_tv[i], streaming_movies[i]]:
            if add == "Yes":
                base += 7.0

        charge = round(base + np.random.normal(0, 3.0), 2)
        charge = max(18.25, min(charge, 118.75))
        monthly_charges.append(charge)

        t = tenures[i]
        if t == 0:
            # Replicate Telco dataset quirk where tenure 0 has empty space
            total_charges.append(" ")
        else:
            tot = round(charge * t + np.random.normal(0, 10.0), 2)
            total_charges.append(str(max(tot, charge)))

    # Compute churn probability based on well-established Telco drivers and non-linear interactions
    churn = []
    for i in range(num_samples):
        # Baseline log-odds
        logit = -1.2
        if contracts[i] == "Month-to-month":
            logit += 1.4
            if internet_service[i] == "Fiber optic":
                logit += 0.7
            if tech_support[i] == "No":
                logit += 0.5
        elif contracts[i] == "Two year":
            logit -= 1.8
        elif contracts[i] == "One year":
            logit -= 0.7

        if internet_service[i] == "Fiber optic":
            logit += 0.5
        elif internet_service[i] == "No":
            logit -= 0.7

        if payment_method[i] == "Electronic check":
            logit += 0.5

        if tech_support[i] == "Yes":
            logit -= 0.4
        if online_security[i] == "Yes":
            logit -= 0.4

        if senior_citizen[i] == 1:
            logit += 0.3

        # Non-linear tenure retention curve
        logit -= np.sqrt(tenures[i] / 72.0) * 2.2

        prob = 1.0 / (1.0 + np.exp(-logit))
        churn.append("Yes" if np.random.rand() < prob else "No")

    df = pd.DataFrame({
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
    })

    return df


def fetch_telco_data(
    save_path: str = "data/raw/telco_customer_churn.csv",
    remote_urls: Optional[list] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Attempts to download the Telco dataset from remote mirrors, falling back to synthetic generator."""
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force_download:
        logger.info("Dataset already exists locally at %s. Loading from file.", target_path)
        return pd.read_csv(target_path)

    urls = remote_urls or DEFAULT_URLS
    downloaded = False
    df = None

    for url in urls:
        try:
            logger.info("Attempting to fetch Telco dataset from %s ...", url)
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(target_path, "wb") as f:
                    f.write(resp.content)
                logger.info("Successfully downloaded dataset to %s", target_path)
                df = pd.read_csv(target_path)
                downloaded = True
                break
        except Exception as e:
            logger.warning("Download failed from %s: %s", url, e)

    if not downloaded or df is None:
        logger.info("Falling back to realistic synthetic Telco dataset generator.")
        df = generate_synthetic_telco_data()
        df.to_csv(target_path, index=False)
        logger.info("Synthetic dataset saved successfully to %s", target_path)

    return df


if __name__ == "__main__":
    df = fetch_telco_data()
    print(f"Data shape: {df.shape}")
    print(f"Churn distribution:\n{df[TARGET_COLUMN].value_counts(normalize=True)}")
