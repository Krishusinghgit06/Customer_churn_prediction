# Customer Churn Prediction Engine with FastAPI & Explainability

A production-ready machine learning service that predicts customer churn on the Telco Customer Churn dataset, optimizes for high recall at a calibrated decision threshold of 0.45, provides model explanations via SHAP TreeExplainer, and serves real-time predictions through FastAPI.

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph Data & Feature Engineering
        RawData["Raw Telco Data\n(Local CSV or Synthetic)"] --> Split["Stratified Train/Test Split\n(80 / 20)"]
        Split --> XTrain["X_train"]
        Split --> XTest["X_test"]
        XTrain --> Preproc["ColumnTransformer\n(Clean + Impute + Scaler + OHE)"]
    end

    subgraph Model Pipeline & SHAP
        Preproc --> Model["XGBoost Classifier\n(scale_pos_weight=2.48)"]
        Model --> Joblib["churn_pipeline.joblib"]
        Model --> Explainer["SHAP TreeExplainer\n(shap_explainer.joblib)"]
        Explainer --> Plot["shap_summary.png"]
    end

    subgraph Serving Layer
        Joblib & Explainer --> FastAPI["FastAPI Service\n(GET /health, POST /predict)"]
        FastAPI --> JSONOut["Prediction JSON\n(Prob + Label @ 0.45 + Top 3 Features)"]
    end
```

---

## Key Features

1. **Leak-Free Preprocessing Pipeline**:
   - `ColumnTransformer` fitted strictly on training data post-split.
   - Robust `TotalChargesCleaner` handling whitespace, string coercion, and median imputation.
   - Standard scaling for continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`).
   - Dense one-hot encoding with `handle_unknown='ignore'` for categorical features.

2. **Model Comparison & Benchmark**:
   - Baseline: `LogisticRegression(class_weight='balanced')`
   - Challenger: `XGBClassifier(scale_pos_weight=2.48)`
   - Winning Model: **XGBoost** achieving **0.9004 ROC-AUC** and **92.59% Recall** at decision threshold `0.45`.

3. **Explainable AI (SHAP)**:
   - Global feature importance beeswarm summary plot generated at `reports/figures/shap_summary.png`.
   - Real-time local JSON attribution extracting the top 3 driving features with direction (`"Increases churn risk"` vs `"Decreases churn risk"`).

4. **FastAPI Serving Layer**:
   - Pydantic v2 schemas validating customer input attributes and enumerations.
   - Startup model caching using FastAPI `lifespan` context manager.
   - Endpoints: `GET /health`, `POST /predict`, and auto-generated Swagger UI at `/docs`.

5. **Pytest Verification Suite**:
   - 14 automated unit and integration tests covering preprocessing, model serialization, probability bounds, and API contracts.

---

## Project Layout

```
customer_churn_engine/
├── config/
│   └── config.yaml              # Hyperparameters, paths, and feature configurations
├── data/
│   └── raw/
│       └── telco_customer_churn.csv # Raw/synthetic Telco dataset
├── models/
│   ├── churn_pipeline.joblib    # Serialized Scikit-learn Pipeline
│   ├── shap_explainer.joblib    # Serialized SHAP TreeExplainer
│   └── metrics.json             # Benchmark comparison metrics
├── reports/
│   └── figures/
│       └── shap_summary.png     # Global SHAP beeswarm plot
├── src/
│   ├── data/
│   │   ├── fetch_data.py        # Robust fetcher with synthetic fallback
│   │   └── schema.py            # Feature definitions & Pydantic models
│   ├── features/
│   │   └── preprocessor.py      # ColumnTransformer pipeline builder
│   ├── models/
│   │   ├── train.py             # Model training, comparison & export
│   │   ├── evaluate.py          # Classification metrics computation
│   │   └── explain.py           # SHAP TreeExplainer integration
│   └── api/
│       ├── main.py              # FastAPI application entrypoint
│       ├── routes.py            # Endpoint handlers (/health, /predict)
│       ├── schemas.py           # API request & response schemas
│       └── service.py           # ModelManager inference singleton
└── tests/
    ├── conftest.py              # Shared fixtures
    ├── test_api.py              # API contract integration tests
    ├── test_model.py            # Pipeline and explainability unit tests
    └── test_preprocessing.py    # Transformer & leakage tests
```

---

## Getting Started

### 1. Prerequisites & Virtual Environment

```bash
cd /Users/krishukumar/.gemini/antigravity/scratch/customer_churn_engine
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Train and Benchmark the Models

```bash
python -m src.models.train
```

Outputs:
- Serializes `models/churn_pipeline.joblib`
- Serializes `models/shap_explainer.joblib`
- Generates `models/metrics.json`
- Saves global summary plot `reports/figures/shap_summary.png`

### 3. Run the Test Suite

```bash
pytest -v
```

All 14 tests will execute, validating preprocessing, model logic, and FastAPI endpoints.

### 4. Start the FastAPI Service

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at `http://localhost:8000/docs`.

---

## API Reference

### `GET /health`
Returns service status, model version, model type, and active threshold.

**Sample Response**:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "model_type": "XGBClassifier",
  "threshold": 0.45
}
```

### `POST /predict`
Submits a customer payload for churn risk evaluation and local SHAP feature explanations.

**Sample Request**:
```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.50,
  "TotalCharges": 95.50
}
```

**Sample Response**:
```json
{
  "customer_id": "7590-VHVEG",
  "churn_probability": 0.8124,
  "churn_prediction": 1,
  "threshold_used": 0.45,
  "top_driving_features": [
    {
      "feature": "Contract_Month-to-month",
      "attribution": 1.4521,
      "impact": "Increases churn risk"
    },
    {
      "feature": "InternetService_Fiber optic",
      "attribution": 0.6214,
      "impact": "Increases churn risk"
    },
    {
      "feature": "tenure",
      "attribution": 0.5103,
      "impact": "Increases churn risk"
    }
  ]
}
```
