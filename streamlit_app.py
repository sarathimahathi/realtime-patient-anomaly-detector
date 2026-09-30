"""
Realtime Patient Anomaly Detector — Streamlit Cloud Web Application
"""

import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_kaggle_dataset, simulate_patient_stream
from src.model import PatientAnomalyDetector
from src.preprocessor import VitalSignsPreprocessor

# Page Configuration
st.set_page_config(
    page_title="Real-Time Patient Anomaly Detector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical CSS Theme (Blended Frosted White & Wine Red)
st.markdown(
    """
    <style>
    .main { 
        background: radial-gradient(ellipse at 12% 12%, rgba(255, 255, 255, 0.95) 0%, rgba(255, 241, 243, 0.8) 35%, transparent 65%),
                    radial-gradient(ellipse at 88% 12%, rgba(225, 29, 72, 0.16) 0%, rgba(254, 205, 211, 0.35) 40%, transparent 65%),
                    radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.9) 0%, rgba(255, 228, 235, 0.45) 45%, transparent 70%),
                    #fcf6f8; 
        color: #4c0519; 
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(255, 248, 250, 0.85) 50%, rgba(254, 226, 232, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-bottom: 1px solid rgba(225, 29, 72, 0.22);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 10px 30px -5px rgba(136, 19, 55, 0.08), 0 3px 14px rgba(255, 255, 255, 0.9), inset 0 1px 2px #fff;
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stMetric:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: rgba(225, 29, 72, 0.45);
        box-shadow: 0 20px 42px -8px rgba(136, 19, 55, 0.16), 0 0 25px rgba(255, 255, 255, 0.95), 0 0 18px rgba(225, 29, 72, 0.25);
    }
    .metric-card-critical {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 228, 230, 0.85) 100%) !important;
        border: 1px solid #ff1744 !important;
        box-shadow: 0 0 25px rgba(225, 29, 72, 0.3) !important;
    }
    .metric-card-warning {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.94) 0%, rgba(254, 243, 199, 0.75) 100%) !important;
        border: 1px solid #d97706 !important;
        box-shadow: 0 0 20px rgba(217, 119, 6, 0.2) !important;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.85rem;
    }
    .status-normal { background: rgba(255, 255, 255, 0.9); color: #be123c; border: 1px solid rgba(225, 29, 72, 0.3); }
    .status-warning { background: rgba(254, 243, 199, 0.85); color: #b45309; border: 1px solid #d97706; }
    .status-critical { background: linear-gradient(135deg, #ff1744 0%, #be123c 100%); color: #fff; border: 1px solid #ff1744; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    """Load or train model and preprocessor artifacts."""
    model_path = Path("models/anomaly_detector.joblib")
    scaler_path = Path("models/scaler.joblib")

    if not model_path.exists() or not scaler_path.exists():
        # Auto-train if artifacts are missing
        df = load_kaggle_dataset()
        prep = VitalSignsPreprocessor().fit(df.head(50000), persist_path=scaler_path)
        X = prep.transform(df.head(50000))
        detector = PatientAnomalyDetector(contamination=0.05, n_estimators=100, preprocessor=prep).fit(X)
        detector.save(model_path)
    else:
        prep = VitalSignsPreprocessor.load(scaler_path)
        detector = PatientAnomalyDetector.load(model_path)
        detector.preprocessor = prep

    return detector, prep


detector, preprocessor = load_pipeline()

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/fluency/96/heart-monitor.png", width=64)
st.sidebar.title("ICU Anomaly Engine")
st.sidebar.caption("IsolationForest ML + Medical Rule Guardrails")
st.sidebar.markdown("---")

selected_patient = st.sidebar.selectbox(
    "Select Patient ID:",
    ["PATIENT-0001", "PATIENT-0002", "PATIENT-0003", "PATIENT_001", "PATIENT_002"],
    index=0,
)

stream_interval = st.sidebar.slider("Stream Refresh Interval (seconds):", 0.5, 3.0, 1.0, 0.5)
simulated_anomaly_rate = st.sidebar.slider("Anomaly Injection Frequency:", 0.0, 0.5, 0.15, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Deterministic Safety Guardrails:**
    - 🩸 **HR**: < 45 or > 130 bpm
    - 🫁 **SpO2**: < 90%
    - 🩺 **Systolic BP**: < 85 or > 170 mmHg
    - 🌡️ **Temp**: > 39.0°C
    """
)

# --- Main App Header ---
st.title("🏥 Real-Time Patient Anomaly Detector")
st.markdown("Continuous ICU vital signs telemetry monitoring, physiological index derivation, and multi-tier anomaly detection.")

tab_live, tab_manual, tab_info = st.tabs(["📡 Live Telemetry Stream", "🩺 Interactive Patient Diagnostic", "📊 Model & Dataset Analytics"])

# =========================================================================
# TAB 1: Live Telemetry Stream
# =========================================================================
with tab_live:
    col_ctrl1, col_ctrl2 = st.columns([1, 4])
    with col_ctrl1:
        run_stream = st.toggle("▶️ Start Live Telemetry Stream", value=True)
    with col_ctrl2:
        inject_crisis = st.button("🚨 Inject Acute Clinical Emergency", type="primary")

    # State containers
    status_placeholder = st.empty()
    vitals_placeholder = st.empty()
    chart_placeholder = st.empty()
    alerts_placeholder = st.empty()

    if "stream_history" not in st.session_state:
        st.session_state.stream_history = []
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []

    if run_stream:
        # Simulate a single live tick
        now = time.time()
        
        # Determine if emergency button was triggered
        force_anomaly = inject_crisis or (np.random.random() < simulated_anomaly_rate)

        if not force_anomaly:
            reading = {
                "patient_id": selected_patient,
                "timestamp": now,
                "heart_rate": round(float(np.clip(np.random.normal(76.0, 4.0), 55.0, 105.0)), 1),
                "spo2": round(float(np.clip(np.random.normal(98.2, 0.8), 94.0, 100.0)), 1),
                "systolic_bp": round(float(np.clip(np.random.normal(120.0, 5.0), 105.0, 138.0)), 1),
                "diastolic_bp": round(float(np.clip(np.random.normal(78.0, 3.5), 65.0, 88.0)), 1),
                "respiratory_rate": round(float(np.clip(np.random.normal(16.0, 1.5), 12.0, 20.0)), 1),
                "temperature": round(float(np.clip(np.random.normal(37.0, 0.2), 36.5, 37.4)), 2),
            }
        else:
            # Acute crisis patterns
            crisis_type = np.random.choice(["Hypoxia", "Tachycardia", "Shock", "Fever"])
            if crisis_type == "Hypoxia":
                hr, spo2, sbp, dbp, rr, temp = 122.0, 84.5, 118.0, 76.0, 28.0, 37.0
            elif crisis_type == "Tachycardia":
                hr, spo2, sbp, dbp, rr, temp = 158.0, 95.0, 138.0, 88.0, 22.0, 37.0
            elif crisis_type == "Shock":
                hr, spo2, sbp, dbp, rr, temp = 135.0, 91.0, 72.0, 42.0, 26.0, 38.6
            else:
                hr, spo2, sbp, dbp, rr, temp = 118.0, 96.0, 125.0, 80.0, 20.0, 39.6

            reading = {
                "patient_id": selected_patient,
                "timestamp": now,
                "heart_rate": hr,
                "spo2": spo2,
                "systolic_bp": sbp,
                "diastolic_bp": dbp,
                "respiratory_rate": rr,
                "temperature": temp,
            }

        # Run model inference
        prediction = detector.predict(reading)
        severity = prediction["severity"]
        anomaly_score = prediction["anomaly_score"]
        flagged_reasons = prediction["flagged_reasons"]
        is_anomaly = prediction["is_anomaly"]

        # Append to session history
        time_label = time.strftime("%H:%M:%S", time.localtime(now))
        st.session_state.stream_history.append({
            "Time": time_label,
            "Heart Rate": reading["heart_rate"],
            "SpO2": reading["spo2"],
            "Risk Score": anomaly_score,
            "Severity": severity,
        })
        if len(st.session_state.stream_history) > 30:
            st.session_state.stream_history.pop(0)

        # Log alerts
        if is_anomaly or severity != "NORMAL":
            reason_str = " • ".join(flagged_reasons) if flagged_reasons else f"Outlier Score: {anomaly_score:.2f}"
            st.session_state.alert_log.insert(0, f"[{time_label}] **{severity}**: {reason_str}")
            if len(st.session_state.alert_log) > 20:
                st.session_state.alert_log.pop()

        # Render Top Status Banner
        with status_placeholder.container():
            pill_class = "status-normal" if severity == "NORMAL" else ("status-warning" if severity == "WARNING" else "status-critical")
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15,23,42,0.6); padding: 12px 20px; border-radius: 10px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.06);">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">MONITORED PATIENT:</span>
                        <strong style="color: #38bdf8; margin-left: 6px;">{selected_patient}</strong>
                        <span style="color: #64748b; margin-left: 16px;">Last Sync: {time_label}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Anomaly Risk: <strong>{anomaly_score:.2f}</strong></span>
                        <span class="status-pill {pill_class}">{severity}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Derived metrics
        pp = reading["systolic_bp"] - reading["diastolic_bp"]
        map_val = reading["diastolic_bp"] + (pp / 3.0)
        shock_idx = reading["heart_rate"] / max(reading["systolic_bp"], 1.0)

        # Render 4 Vitals Cards
        with vitals_placeholder.container():
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(
                label="❤️ Heart Rate",
                value=f"{int(reading['heart_rate'])} bpm",
                delta=f"Safe: 45 - 130",
                delta_color="off",
            )
            kpi2.metric(
                label="🫁 Oxygen (SpO2)",
                value=f"{reading['spo2']:.1f} %",
                delta=f"Safe: ≥ 90%",
                delta_color="off",
            )
            kpi3.metric(
                label="🩺 Blood Pressure",
                value=f"{int(reading['systolic_bp'])}/{int(reading['diastolic_bp'])} mmHg",
                delta=f"MAP: {int(map_val)} mmHg",
                delta_color="off",
            )
            kpi4.metric(
                label="🌡️ Temperature",
                value=f"{reading['temperature']:.1f} °C",
                delta=f"Shock Index: {shock_idx:.2f}",
                delta_color="off",
            )

        # Render Trends Chart
        with chart_placeholder.container():
            st.subheader("📈 Real-Time Telemetry Trends (Last 30 Points)")
            if st.session_state.stream_history:
                chart_df = pd.DataFrame(st.session_state.stream_history).set_index("Time")
                st.line_chart(chart_df[["Heart Rate", "SpO2"]], height=240)

        # Render Alert Log
        with alerts_placeholder.container():
            st.subheader("🚨 Clinical Event & Anomaly Log")
            if st.session_state.alert_log:
                for alert in st.session_state.alert_log[:6]:
                    if "CRITICAL" in alert:
                        st.error(alert)
                    elif "WARNING" in alert:
                        st.warning(alert)
                    else:
                        st.info(alert)
            else:
                st.caption("No acute anomalies detected. Telemetry normal.")

        # Trigger auto-refresh loop
        time.sleep(stream_interval)
        st.rerun()

# =========================================================================
# TAB 2: Interactive Patient Diagnostic Tool
# =========================================================================
with tab_manual:
    st.subheader("🧪 Single Patient Parameter Assessment")
    st.markdown("Enter or adjust custom vital signs to test the machine learning model and medical safety guardrails on-demand.")

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        inp_hr = st.slider("Heart Rate (bpm):", 30, 220, 75)
        inp_spo2 = st.slider("Oxygen Saturation SpO2 (%):", 60.0, 100.0, 98.0, 0.5)
        inp_sbp = st.slider("Systolic Blood Pressure (mmHg):", 50, 240, 120)
    with diag_col2:
        inp_dbp = st.slider("Diastolic Blood Pressure (mmHg):", 30, 150, 80)
        inp_rr = st.slider("Respiratory Rate (breaths/min):", 6, 60, 16)
        inp_temp = st.slider("Body Temperature (°C):", 34.0, 42.0, 37.0, 0.1)

    if st.button("🔍 Run Anomaly Diagnostics", type="primary"):
        manual_record = {
            "heart_rate": inp_hr,
            "spo2": inp_spo2,
            "systolic_bp": inp_sbp,
            "diastolic_bp": inp_dbp,
            "respiratory_rate": inp_rr,
            "temperature": inp_temp,
        }

        res = detector.predict(manual_record)
        pp = inp_sbp - inp_dbp
        map_val = inp_dbp + (pp / 3.0)
        si = inp_hr / max(inp_sbp, 1)

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Severity Level", res["severity"])
            st.metric("Anomaly Risk Score", f"{res['anomaly_score']:.2f}")
            st.metric("Calculated MAP", f"{map_val:.1f} mmHg")
            st.metric("Shock Index", f"{si:.2f}")

        with res_col2:
            if res["is_anomaly"] or res["severity"] == "CRITICAL":
                st.error(f"### 🚨 Anomaly Detected: {res['severity']}")
            elif res["severity"] == "WARNING":
                st.warning(f"### ⚠️ Clinical Alert: {res['severity']}")
            else:
                st.success("### ✅ Physiological Parameters Normal")

            if res["flagged_reasons"]:
                st.markdown("**Triggered Clinical Reasons:**")
                for r in res["flagged_reasons"]:
                    st.write(f"- {r}")
            else:
                st.write("All parameters within baseline clinical safety boundaries.")

# =========================================================================
# TAB 3: Model & Dataset Analytics
# =========================================================================
with tab_info:
    st.subheader("📊 Model & 2024 Clinical Dataset Architecture")
    st.markdown(
        """
        - **Dataset**: Kaggle / Real-World Human Vital Signs Dataset 2024 (200,020 records across high and low clinical risk cohorts).
        - **Model Architecture**: Unsupervised `IsolationForest` (contamination=0.05, n_estimators=150) calibrated with deterministic rule guardrails.
        - **Feature Engineering**:
          - Mean Arterial Pressure (MAP): $DBP + \\frac{SBP - DBP}{3}$
          - Pulse Pressure: $SBP - DBP$
          - Shock Index: $\\frac{HR}{SBP}$
          - StandardScaler normalization across all 8 feature dimensions.
        """
    )
    st.info("System fully configured and ready for live production use.")
