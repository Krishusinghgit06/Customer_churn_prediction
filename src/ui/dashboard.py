"""Streamlit Dashboard for real-time customer churn prediction and SHAP explainability."""

import os
from typing import Any, Dict
import requests
import streamlit as st

# Configure wide layout without sidebar
st.set_page_config(
    page_title="Customer Churn Prediction Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API Configuration (internal default with env override, no sidebar)
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"


def check_api_status():
    """Pings FastAPI /health endpoint to check server availability."""
    try:
        r = requests.get(HEALTH_URL, timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


# Presets Definition
PRESET_CUSTOM = "Custom"
PRESET_HIGH_RISK = "🔴 High Churn Risk (Month-to-month, Fiber, No Support)"
PRESET_LOYAL = "🟢 Loyal Customer (Two-year, DSL, High Tenure)"

PRESET_DATA = {
    PRESET_HIGH_RISK: {
        "customerID": "9999-RISK1",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 95.50,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
    },
    PRESET_LOYAL: {
        "customerID": "0001-LOYAL",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.20,
        "TotalCharges": 2938.00,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    },
    PRESET_CUSTOM: {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.00,
        "TotalCharges": 840.00,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    },
}

# Custom CSS for Dark Mode & Clean Header
st.markdown(
    """
    <style>
    /* 1. Hide Deploy Button & Top Toolbar completely */
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* 2. Global Container Sizing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* 3. Card Title with high-contrast text and subtle divider */
    .ui-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-color, #f8fafc);
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        padding-bottom: 8px;
    }

    /* 4. Subtitle Divider */
    .sub-divider {
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin: 1.2rem 0;
    }

    /* 5. Presets Container with dark-mode friendly glassmorphism */
    .preset-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 20px;
    }

    /* 6. Badges */
    .badge-threshold {
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 5px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-online {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
        padding: 5px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 5px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .dot-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
    }
    .dot-red {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ef4444;
    }

    /* 7. Risk Badges & Callouts */
    .risk-badge-high {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .risk-badge-low {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.45);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pill-inc {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 9px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .pill-dec {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 3px 9px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Header with High-Contrast Typography & Live API Status Badge
is_online, health_data = check_api_status()
if is_online:
    status_html = f'<span class="badge-online"><span class="dot-green"></span> API Online ({health_data.get("model_type", "XGBoost")})</span>'
else:
    status_html = '<span class="badge-offline"><span class="dot-red"></span> API Offline (127.0.0.1:8000)</span>'

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
        <div>
            <h1 style="margin: 0; font-size: 2.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                📊 Customer Churn Prediction Engine
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                XGBoost Binary Classifier with real-time SHAP TreeExplainer feature attributions
            </p>
        </div>
        <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px;">
            <span class="badge-threshold">Threshold: 0.45</span>
            {status_html}
        </div>
    </div>
    <div class="sub-divider"></div>
    """,
    unsafe_allow_html=True,
)

# 2. Main Page Scenario Preset Selector
st.markdown('<div class="preset-container">', unsafe_allow_html=True)
selected_preset = st.radio(
    "⚡ Select Profile Preset:",
    [PRESET_CUSTOM, PRESET_HIGH_RISK, PRESET_LOYAL],
    horizontal=True,
    key="active_preset",
)
st.markdown("</div>", unsafe_allow_html=True)

# Update session state fields when preset selection changes
if "last_preset" not in st.session_state or st.session_state.last_preset != selected_preset:
    st.session_state.last_preset = selected_preset
    defaults = PRESET_DATA[selected_preset]
    for k, v in defaults.items():
        st.session_state[f"field_{k}"] = v

# Fallback initialization if first load
curr_defaults = PRESET_DATA[selected_preset]
for k, v in curr_defaults.items():
    if f"field_{k}" not in st.session_state:
        st.session_state[f"field_{k}"] = v

# 3. Modern 3-Card Layout for Input Fields
card_col1, card_col2, card_col3 = st.columns(3, gap="medium")

# --- CARD 1: Customer & Demographics ---
with card_col1:
    with st.container(border=True):
        st.markdown('<div class="ui-card-title">👤 Customer & Demographics</div>', unsafe_allow_html=True)
        customer_id = st.text_input("Customer ID", value=st.session_state["field_customerID"])
        
        gender_idx = 0 if st.session_state["field_gender"] == "Female" else 1
        gender = st.selectbox("Gender", ["Female", "Male"], index=gender_idx)
        
        senior_idx = 1 if st.session_state["field_SeniorCitizen"] == 1 else 0
        senior_citizen_val = st.selectbox("Senior Citizen", ["No", "Yes"], index=senior_idx)
        senior_citizen = 1 if senior_citizen_val == "Yes" else 0
        
        partner_idx = 0 if st.session_state["field_Partner"] == "No" else 1
        partner = st.selectbox("Partner", ["No", "Yes"], index=partner_idx)
        
        dependents_idx = 0 if st.session_state["field_Dependents"] == "No" else 1
        dependents = st.selectbox("Dependents", ["No", "Yes"], index=dependents_idx)

# --- CARD 2: Contract & Billing ---
with card_col2:
    with st.container(border=True):
        st.markdown('<div class="ui-card-title">📑 Contract & Billing</div>', unsafe_allow_html=True)
        tenure = st.slider(
            "Tenure (Months with Company)",
            min_value=0,
            max_value=72,
            value=int(st.session_state["field_tenure"]),
        )
        
        contract_opts = ["Month-to-month", "One year", "Two year"]
        contract_idx = contract_opts.index(st.session_state["field_Contract"])
        contract = st.selectbox("Contract Term", contract_opts, index=contract_idx)
        
        paperless_idx = 0 if st.session_state["field_PaperlessBilling"] == "Yes" else 1
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=paperless_idx)
        
        payment_opts = [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
        payment_idx = payment_opts.index(st.session_state["field_PaymentMethod"])
        payment_method = st.selectbox("Payment Method", payment_opts, index=payment_idx)
        
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=18.0,
            max_value=130.0,
            value=float(st.session_state["field_MonthlyCharges"]),
            step=1.0,
        )
        
        total_charges = st.number_input(
            "Total Lifetime Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(st.session_state["field_TotalCharges"]),
            step=10.0,
        )

# --- CARD 3: Subscribed Services ---
with card_col3:
    with st.container(border=True):
        st.markdown('<div class="ui-card-title">🌐 Subscribed Services</div>', unsafe_allow_html=True)
        phone_idx = 0 if st.session_state["field_PhoneService"] == "Yes" else 1
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=phone_idx)
        
        lines_opts = ["No phone service", "No", "Yes"]
        lines_idx = lines_opts.index(st.session_state["field_MultipleLines"])
        multiple_lines = st.selectbox("Multiple Lines", lines_opts, index=lines_idx)
        
        internet_opts = ["Fiber optic", "DSL", "No"]
        internet_idx = internet_opts.index(st.session_state["field_InternetService"])
        internet_service = st.selectbox("Internet Service Provider", internet_opts, index=internet_idx)
        
        addon_opts = ["No", "Yes", "No internet service"]
        
        tech_idx = addon_opts.index(st.session_state["field_TechSupport"])
        tech_support = st.selectbox("Tech Support", addon_opts, index=tech_idx)
        
        sec_idx = addon_opts.index(st.session_state["field_OnlineSecurity"])
        online_security = st.selectbox("Online Security", addon_opts, index=sec_idx)
        
        backup_idx = addon_opts.index(st.session_state["field_OnlineBackup"])
        online_backup = st.selectbox("Online Backup", addon_opts, index=backup_idx)
        
        device_idx = addon_opts.index(st.session_state["field_DeviceProtection"])
        device_protection = st.selectbox("Device Protection", addon_opts, index=device_idx)
        
        tv_idx = addon_opts.index(st.session_state["field_StreamingTV"])
        streaming_tv = st.selectbox("Streaming TV", addon_opts, index=tv_idx)
        
        movie_idx = addon_opts.index(st.session_state["field_StreamingMovies"])
        streaming_movies = st.selectbox("Streaming Movies", addon_opts, index=movie_idx)

# 4. Prominent Centered Predict Action
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 1.6, 1])

with btn_col2:
    run_assessment = st.button("🚀 Run Churn Assessment", type="primary", use_container_width=True)

# 5. Polished Results Section with Dark-Mode Contrast
if run_assessment:
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

    with st.spinner("Executing XGBoost pipeline and extracting SHAP TreeExplainer attributions..."):
        try:
            resp = requests.post(PREDICT_URL, json=payload, timeout=6.0)
            if resp.status_code == 200:
                result = resp.json()
                churn_prob = result["churn_probability"]
                churn_pred = result["churn_prediction"]
                threshold = result["threshold_used"]
                top_features = result.get("top_driving_features", [])

                st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
                st.markdown("### 🎯 Assessment Results")

                res_card1, res_card2 = st.columns([1.1, 1.9], gap="medium")

                # Left: Risk Summary Card
                with res_card1:
                    with st.container(border=True):
                        st.markdown('<div class="ui-card-title">📈 Churn Risk Profile</div>', unsafe_allow_html=True)
                        
                        prob_pct = churn_prob * 100
                        st.metric(
                            label="Predicted Churn Probability",
                            value=f"{prob_pct:.1f}%",
                            delta=f"Threshold: {threshold * 100:.0f}%",
                            delta_color="off",
                        )
                        st.progress(float(churn_prob))
                        
                        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                        if churn_pred == 1:
                            st.markdown(
                                '<div class="risk-badge-high">🚨 HIGH RISK (Churn Likely)</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                """
                                <div style="background: rgba(239, 68, 68, 0.12); border-left: 4px solid #ef4444; padding: 12px; margin-top: 14px; border-radius: 4px; font-size: 0.9rem; color: #fca5a5;">
                                    <strong style="color: #f87171;">Automated Action:</strong> Customer is at high risk of churning. Proactively trigger retention offer: 12-month contract lock discount, loyalty credit, or free premium tech support.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="risk-badge-low">✅ LOW RISK (Retained)</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                """
                                <div style="background: rgba(34, 197, 94, 0.12); border-left: 4px solid #22c55e; padding: 12px; margin-top: 14px; border-radius: 4px; font-size: 0.9rem; color: #86efac;">
                                    <strong style="color: #4ade80;">Automated Action:</strong> Customer shows strong retention indicators. Maintain standard engagement and explore cross-sell opportunities (e.g., streaming add-ons).
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Right: Interactive SHAP Feature Drivers
                with res_card2:
                    with st.container(border=True):
                        st.markdown(
                            '<div class="ui-card-title">🔍 Top 3 Contributing Factors (SHAP TreeExplainer)</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("Individual feature contributions driving the prediction above or below baseline log-odds:")

                        max_abs_attr = max([abs(f["attribution"]) for f in top_features] + [1.0])

                        for i, feat in enumerate(top_features, 1):
                            fname = feat["feature"]
                            attr = feat["attribution"]
                            impact = feat["impact"]
                            is_inc = "Increases" in impact

                            pill_class = "pill-inc" if is_inc else "pill-dec"
                            pill_label = "Increases Risk" if is_inc else "Decreases Risk"
                            bar_color = "#ef4444" if is_inc else "#10b981"
                            sign = "+" if attr > 0 else ""
                            bar_pct = min(max(int((abs(attr) / max_abs_attr) * 100), 10), 100)

                            st.markdown(
                                f"""
                                <div style="margin-bottom: 14px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                                        <span style="font-weight: 600; color: #f1f5f9; font-size: 0.95rem;">
                                            {i}. {fname}
                                        </span>
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <span style="font-family: monospace; font-weight: 700; color: #cbd5e1;">{sign}{attr:.4f}</span>
                                            <span class="{pill_class}">{pill_label}</span>
                                        </div>
                                    </div>
                                    <div style="background: rgba(255, 255, 255, 0.12); border-radius: 6px; height: 9px; width: 100%; overflow: hidden;">
                                        <div style="background: {bar_color}; width: {bar_pct}%; height: 100%; border-radius: 6px;"></div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                with st.expander("📄 View Raw Prediction JSON"):
                    st.json(result)

            elif resp.status_code == 422:
                st.error("Validation Error: Input payload rejected by FastAPI schema.")
                st.json(resp.json())
            else:
                st.error(f"Inference error: HTTP {resp.status_code}")
                st.code(resp.text)

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Connection Error: Unable to reach FastAPI backend at `{PREDICT_URL}`.\n\n"
                "To test end-to-end, start the FastAPI server in another terminal:\n"
                "```bash\n"
                "cd /Users/krishukumar/customer_churn_engine\n"
                "uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n"
                "```"
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ Request Timeout: The backend took longer than 6 seconds to respond.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
