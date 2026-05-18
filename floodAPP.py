import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Flood Prediction System", layout="centered")

st.title("🌊 District-Level Flood Prediction System")
st.markdown("Predict **river discharge (m³/s)** and **flood warning level** using ML")

# =========================
# Load Models & Artifacts
# =========================
@st.cache_resource
def load_all_models():
    lstm_models = {
       "Sialkot": load_model("lstm_discharge_Sialkot.keras",compile=False),
        "Okara": load_model("lstm_discharge_Okara.keras",compile=False),
        "Multan": load_model("lstm_discharge_Multan.keras",compile=False),
    }
    

    X_scalers = joblib.load("X_scalers.pkl")
    y_scalers = joblib.load("y_scalers.pkl")
    lgbm_model = joblib.load("flood_classifier_lgbm.pkl")

    return lstm_models, X_scalers, y_scalers, lgbm_model


lstm_models, X_scalers, y_scalers, lgbm_model = load_all_models()

# =========================
# Feature Definitions (7 FEATURES – SAME AS TRAINING)
# =========================
DYNAMIC_FEATURES = [
    "rainfall",
    "total_runoff_mm",
    "API_7",
    "API_30",
    "evaporation_mm",
    "temperature_C",
    "discharge"
]

LOOKBACK = 14

# =========================
# User Inputs
# =========================
district = st.selectbox("Select District", ["Sialkot", "Okara", "Multan"])

st.subheader("Enter Last 14 Days Inputs")

rainfall_values = []
temperature_values = []

for i in range(LOOKBACK):
    col1, col2 = st.columns(2)

    with col1:
        rainfall_values.append(
            st.number_input(
                f"Rainfall Day -{LOOKBACK - i} (mm)",
                min_value=0.0,
                step=0.1,
                key=f"rain_{i}"
            )
        )

    with col2:
        temperature_values.append(
            st.number_input(
                f"Temperature Day -{LOOKBACK - i} (°C)",
                min_value=-5.0,
                max_value=50.0,
                step=0.1,
                key=f"temp_{i}"
            )
        )

rainfall_values = np.array(rainfall_values)
temperature_values = np.array(temperature_values)

# =========================
# Build LSTM Input (MATCHES TRAINING)
# =========================
def build_lstm_input(rainfall, temperature):
    seq = []
    for r, t in zip(rainfall, temperature):
        seq.append([
            r,          # rainfall
            0.0,        # total_runoff_mm (placeholder)
            0.0,        # API_7
            0.0,        # API_30
            0.0,        # evaporation_mm
            t,          # temperature_C (USER INPUT)
            0.0         # discharge (lag)
        ])
    return np.array(seq).reshape(1, LOOKBACK, len(DYNAMIC_FEATURES))

# =========================
# Prediction Button
# =========================
if st.button("🔮 Predict Flood Risk"):

    # Build input
    X_input = build_lstm_input(rainfall_values, temperature_values)

    # Scale X
    X_scaled = X_scalers[district].transform(
        X_input.reshape(-1, X_input.shape[-1])
    ).reshape(X_input.shape)

    # LSTM prediction (scaled)
    y_pred_scaled = lstm_models[district].predict(X_scaled, verbose=0)[0][0]

    # Inverse-scale discharge
    y_pred = y_scalers[district].inverse_transform(
        np.array([[y_pred_scaled]])
    )[0][0]

    y_pred = max(0, y_pred)  # clip negatives

    # =========================
    # LightGBM Flood Classification
    # =========================
    lgbm_input = pd.DataFrame([{
        "predicted_discharge": y_pred,
        "rainfall": rainfall_values[-1],
        "API_7": np.sum(rainfall_values[-7:]),
        "API_30": np.sum(rainfall_values),
        "evaporation_mm": 0.0,
        "temperature_C": temperature_values[-1],
        "slope_mean": 0.0,
    }])

    flood_class = lgbm_model.predict(lgbm_input)[0]

    # =========================
    # Display Results
    # =========================
    st.subheader("📊 Prediction Results")

    st.metric(
        "Predicted Discharge",
        f"{y_pred:.2f} "
    )

    flood_labels = {
        0: ("🟢 No Flood", "green"),
        1: ("🟡 Low Alert", "gold"),
        2: ("🟠 Moderate Alert", "orange"),
        3: ("🔴 Severe Flood", "red")
    }

    label, color = flood_labels[flood_class]

    st.markdown(
        f"<h3 style='color:{color};'>{label}</h3>",
        unsafe_allow_html=True
    )
