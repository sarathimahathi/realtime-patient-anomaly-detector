# Real-Time Patient Anomaly Detector

A modular, production-ready machine learning framework for continuous ICU patient telemetry streaming, clinical feature extraction, and real-time anomaly detection.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A["Patient Telemetry Source / Streamer"] -->|Vitals JSON| B["VitalSignsPreprocessor"]
    B -->|StandardScaler + Derived Features (PP, MAP)| C["PatientAnomalyDetector"]
    C -->|IsolationForest + Clinical Guardrails| D["FastAPI REST & WebSocket Server"]
    D -->|Real-Time WS /ws/live/{patient_id}| E["Live Visualizer Dashboard"]
```

---

## 📂 Directory Structure

```text
realtime-patient-anomaly-detector/
├── data/
│   ├── raw/                  # Ingested patient telemetry logs (CSV, Parquet)
│   │   └── patient_vitals.csv
│   └── processed/            # Cleaned, standardized feature matrices
├── models/                   # Serialized model weights & preprocessors (.joblib)
│   ├── anomaly_detector.joblib
│   └── scaler.joblib
├── src/
│   ├── __init__.py           # Package exports and versioning
│   ├── data_loader.py        # Dataset loading & mock patient vital streamer
│   ├── preprocessor.py       # Clinical feature engineering & StandardScaler pipeline
│   ├── model.py              # Isolation Forest anomaly detector & deterministic guardrails
│   └── server.py             # FastAPI app with REST inference & WebSocket streaming
├── dashboard/                # Live single-page telemetry dashboard
│   ├── index.html            # Main visualizer UI
│   ├── style.css             # Glassmorphic dark-mode styling
│   └── app.js                # WebSocket client & real-time canvas chart
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Unit tests for preprocessing & ML detector
│   └── test_stream.py        # Integration tests for streamer, API & WebSockets
├── train.py                  # Standalone training and evaluation script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Virtual environment recommended (`venv` or `conda`)

### 2. Installation
```bash
# Navigate to the project directory
cd realtime-patient-anomaly-detector

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Training
Run the offline training script to fit the `StandardScaler` and `IsolationForest`, evaluate anomaly distributions, and persist artifacts to `models/`:
```bash
python train.py
```

### 4. Run the Server & Dashboard
```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```
Open your browser at:
- **Live Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Running Tests

Execute unit and integration tests using `pytest`:
```bash
pytest -v
```

---

## 📡 API & WebSocket Reference

### 1. REST Endpoint: Single Vital Prediction
- **Method**: `POST /api/v1/predict`
- **Request Body**:
```json
{
  "patient_id": "PATIENT-1001",
  "heart_rate": 145.0,
  "spo2": 85.0,
  "systolic_bp": 72.0,
  "diastolic_bp": 45.0,
  "respiratory_rate": 28.0,
  "temperature": 39.5
}
```
- **Response**:
```json
{
  "patient_id": "PATIENT-1001",
  "is_anomaly": true,
  "anomaly_score": 0.95,
  "severity": "CRITICAL",
  "flagged_reasons": [
    "Severe tachycardia: Heart Rate 145.0 bpm > 130 bpm",
    "Critical hypoxia: SpO2 85.0% < 90.0%",
    "Severe hypotensive shock: Systolic BP 72.0 mmHg < 85.0 mmHg",
    "Severe hyperthermia: Core Temperature 39.50°C > 39.0°C"
  ],
  "derived_metrics": {
    "pulse_pressure": 27.0,
    "mean_arterial_pressure": 54.0,
    "shock_index": 2.01
  }
}
```

### 2. WebSocket Telemetry Stream
- **Endpoint**: `ws://localhost:8000/ws/live/{patient_id}?interval=1.0`
- **Output Schema (Emitted every 1 second)**:
```json
{
  "raw_vitals": {
    "patient_id": "PATIENT-1001",
    "timestamp": 1727670000.0,
    "heart_rate": 74.2,
    "spo2": 98.4,
    "systolic_bp": 120.1,
    "diastolic_bp": 78.0,
    "respiratory_rate": 16.0,
    "temperature": 37.0
  },
  "is_anomaly": false,
  "severity": "NORMAL",
  "anomaly_score": 0.1873,
  "timestamp": 1727670000.0,
  "flagged_reasons": []
}
```
- **Client Commands**: Send `{"inject_anomaly": true}` through the socket to immediately trigger acute clinical crisis patterns for real-time testing.
