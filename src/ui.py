"""Streamlit Dashboard for real-time customer churn prediction and SHAP explainability."""

import json
from typing import Any, Dict
import requests
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .risk-high {
        color: #d90429;
        font-weight: 700;
        font-size: 1.4rem;
    }
    .risk-low {
        color: #2b9348;
        font-weight: 700;
        font-size: 1.4rem;
    }
    .badge-pos {
        background-color: #ffe3e3;
        color: #c92a2a;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-neg {
        background-color: #e6fcf5;
        color: #0ca678;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.title("⚙️ Service Settings")
api_base = st.sidebar.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")
predict_endpoint = f"{api_base.rstrip('/')}/predict"
health_endpoint = f"{api_base.rstrip('/')}/health"

if st.sidebar.button("Test API Connection"):
    try:
        resp = requests.get(health_endpoint, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            st.sidebar.success(f"Connected! Model: {data.get('model_type')} (v{data.get('model_version')})")
        else:
            st.sidebar.error(f"Health check failed: HTTP {resp.status_code}")
    except Exception as e:
        st.sidebar.error(f"Could not connect to FastAPI server at {health_endpoint}: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Presets")
preset = st.sidebar.radio(
    "Load Scenario:",
    ["Custom", "High Churn Risk Customer", "Loyal Long-Term Customer"],
)

# Header
st.title("📊 Customer Churn Prediction & Explainability Engine")
st.markdown(
    "Evaluate customer retention risk powered by **XGBoost** and interpret individual decisions via **SHAP TreeExplainer**."
)

# Set defaults based on preset
if preset == "High Churn Risk Customer":
    default_gender = "Female"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 1
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 95.50
    default_phone = "Yes"
    default_lines = "No"
    default_internet = "Fiber optic"
    default_sec = "No"
    default_backup = "No"
    default_device = "No"
    default_tech = "No"
    default_tv = "Yes"
    default_movies = "Yes"
elif preset == "Loyal Long-Term Customer":
    default_gender = "Male"
    default_senior = 0
    default_partner = "Yes"
    default_dependents = "Yes"
    default_tenure = 65
    default_contract = "Two year"
    default_paperless = "No"
    default_payment = "Bank transfer (automatic)"
    default_monthly = 45.20
    default_phone = "Yes"
    default_lines = "Yes"
    default_internet = "DSL"
    default_sec = "Yes"
    default_backup = "Yes"
    default_device = "Yes"
    default_tech = "Yes"
    default_tv = "No"
    default_movies = "No"
else:
    default_gender = "Female"
    default_senior = 0
    default_partner = "No"
    default_dependents = "No"
    default_tenure = 12
    default_contract = "Month-to-month"
    default_paperless = "Yes"
    default_payment = "Electronic check"
    default_monthly = 70.00
    default_phone = "Yes"
    default_lines = "No"
    default_internet = "Fiber optic"
    default_sec = "No"
    default_backup = "No"
    default_device = "No"
    default_tech = "No"
    default_tv = "No"
    default_movies = "No"

# Input Form in 3 Columns
st.markdown("### Customer Profile Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Demographics & Account")
    customer_id = st.text_input("Customer ID", value="CUST-7590")
    gender = st.selectbox("Gender", ["Female", "Male"], index=0 if default_gender == "Female" else 1)
    senior_citizen_val = st.selectbox("Senior Citizen", ["No", "Yes"], index=default_senior)
    senior_citizen = 1 if senior_citizen_val == "Yes" else 0
    partner = st.selectbox("Partner", ["No", "Yes"], index=0 if default_partner == "No" else 1)
    dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if default_dependents == "No" else 1)
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=default_tenure)

with col2:
    st.subheader("2. Contract & Billing")
    contract = st.selectbox(
        "Contract Term",
        ["Month-to-month", "One year", "Two year"],
        index=["Month-to-month", "One year", "Two year"].index(default_contract),
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_paperless == "Yes" else 1)
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        index=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ].index(default_payment),
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.0,
        max_value=130.0,
        value=float(default_monthly),
        step=1.0,
    )
    total_charges = st.number_input(
        "Total Lifetime Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(round(monthly_charges * max(tenure, 1), 2)),
        step=10.0,
    )

with col3:
    st.subheader("3. Subscribed Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_phone == "Yes" else 1)
    multiple_lines_options = ["No phone service", "No", "Yes"] if phone_service == "No" else ["No", "Yes"]
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"],
        index=["No phone service", "No", "Yes"].index(default_lines if phone_service == "Yes" else "No phone service"),
    )
    internet_service = st.selectbox(
        "Internet Service Provider",
        ["Fiber optic", "DSL", "No"],
        index=["Fiber optic", "DSL", "No"].index(default_internet),
    )

    addon_options = ["No", "Yes", "No internet service"] if internet_service != "No" else ["No internet service"]
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_sec if internet_service != "No" else "No internet service"))
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_backup if internet_service != "No" else "No internet service"))
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_device if internet_service != "No" else "No internet service"))
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tech if internet_service != "No" else "No internet service"))
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_tv if internet_service != "No" else "No internet service"))
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(default_movies if internet_service != "No" else "No internet service"))

st.markdown("---")

# Prediction Action
evaluate_btn = st.button("🚀 Evaluate Churn Risk", type="primary", use_container_width=True)

if evaluate_btn:
    payload = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    with st.spinner("Calling FastAPI inference service and calculating SHAP attributions..."):
        try:
            response = requests.post(predict_endpoint, json=payload, timeout=8)
            if response.status_code == 200:
                result = response.json()
                churn_prob = result["churn_probability"]
                churn_pred = result["churn_prediction"]
                threshold = result["threshold_used"]
                top_features = result.get("top_driving_features", [])

                st.markdown("## 🎯 Prediction & Risk Analysis")

                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    st.metric(
                        label="Predicted Churn Probability",
                        value=f"{churn_prob * 100:.1f}%",
                        delta=f"Threshold: {threshold * 100:.0f}%",
                        delta_color="off",
                    )
                    st.progress(float(churn_prob))

                    if churn_pred == 1:
                        st.error("🚨 **HIGH RISK**: Customer is likely to churn. Immediate retention intervention recommended.")
                    else:
                        st.success("✅ **LOW RISK**: Customer is expected to stay with current retention strategy.")

                with res_col2:
                    st.markdown("### 🔍 Top 3 Contributing Factors (SHAP Explanations)")
                    st.caption("Feature attribution shows the direction and magnitude pushing the risk higher or lower.")

                    for i, feat in enumerate(top_features, 1):
                        feature_name = feat["feature"]
                        attribution = feat["attribution"]
                        impact = feat["impact"]
                        is_risk_increase = "Increases" in impact

                        badge_class = "badge-pos" if is_risk_increase else "badge-neg"
                        icon = "🔺" if is_risk_increase else "🔻"
                        sign = "+" if attribution > 0 else ""

                        st.markdown(
                            f"""
                            <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #ffffff;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 600; font-size: 1.05rem;">{i}. {feature_name}</span>
                                    <span class="{badge_class}">{icon} {impact} ({sign}{attribution:.4f})</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with st.expander("📦 Raw API Response Payload"):
                    st.json(result)

            elif response.status_code == 422:
                st.error("Validation Error: The input payload did not match the expected API schema.")
                st.json(response.json())
            else:
                st.error(f"Inference server returned error: HTTP {response.status_code}")
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Connection Error: Unable to reach FastAPI service at `{predict_endpoint}`.\n\n"
                "Please verify that the FastAPI backend is running using:\n"
                "```bash\n"
                "uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n"
                "```"
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ Request Timeout: The inference service took too long to respond.")
        except Exception as ex:
            st.error(f"Unexpected error: {ex}")
