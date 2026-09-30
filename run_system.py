"""
Unified startup runner script for Real-Time Patient Anomaly Detector.

Pipeline steps:
1. Verify and execute train.py to ensure valid model artifacts in models/.
2. Run pytest suite across tests/ to confirm system integrity.
3. Start the FastAPI + WebSocket server on port 8000 in background mode.
4. Verify HTTP health status and print browser URLs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "anomaly_detector.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
HEALTH_URL = f"{SERVER_URL}/health"
DASHBOARD_URL = f"{SERVER_URL}/dashboard"


def print_banner(text: str) -> None:
    print(f"\n{'='*70}\n  {text}\n{'='*70}")


def run_training() -> None:
    """Step 1: Execute train.py and verify model artifacts in models/."""
    print_banner("STEP 1: Verifying & Executing Model Training (train.py)")
    
    cmd = [sys.executable, "train.py"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\n[ERROR] train.py failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    if not MODEL_FILE.exists() or MODEL_FILE.stat().st_size == 0:
        print(f"\n[ERROR] Expected model artifact missing or empty at {MODEL_FILE}")
        sys.exit(1)

    if not SCALER_FILE.exists() or SCALER_FILE.stat().st_size == 0:
        print(f"\n[ERROR] Expected scaler artifact missing or empty at {SCALER_FILE}")
        sys.exit(1)

    print(f"[SUCCESS] Model artifact verified:  {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1024:.1f} KB)")
    print(f"[SUCCESS] Scaler artifact verified: {SCALER_FILE} ({SCALER_FILE.stat().st_size} bytes)")


def run_tests() -> None:
    """Step 2: Execute pytest test suite."""
    print_banner("STEP 2: Running Test Suite (pytest tests/)")
    
    cmd = [sys.executable, "-m", "pytest", "-v"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\n[ERROR] Test suite failed with exit code {result.returncode}")
        sys.exit(result.returncode)
        
    print("[SUCCESS] All unit and integration tests passed successfully!")


def start_server_background() -> subprocess.Popen:
    """Step 3: Launch FastAPI server in background mode and verify /health."""
    print_banner("STEP 3: Launching FastAPI + WebSocket Server in Background")
    
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.server:app",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
        "--log-level",
        "info",
    ]
    print(f"Command: {' '.join(cmd)}")
    
    # Launch in background
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(f"Spawned server process (PID: {process.pid}). Waiting for health check...")
    
    # Poll /health up to 15 seconds
    max_retries = 30
    ready = False
    for i in range(max_retries):
        # Check if process terminated prematurely
        if process.poll() is not None:
            out, _ = process.communicate()
            print(f"[ERROR] Server terminated prematurely:\n{out}")
            sys.exit(1)

        try:
            resp = requests.get(HEALTH_URL, timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "online" and data.get("model_loaded"):
                    ready = True
                    break
        except requests.RequestException:
            pass
        time.sleep(0.5)

    if not ready:
        print(f"[ERROR] Server failed to become healthy at {HEALTH_URL} within timeout.")
        process.terminate()
        sys.exit(1)

    print(f"[SUCCESS] FastAPI Server is ONLINE and operational on port {SERVER_PORT}!")
    return process


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Startup Runner for Patient Anomaly Detector")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        default=True,
        help="Keep server running interactively until Ctrl+C (default: True)",
    )
    parser.add_argument(
        "--no-keep-alive",
        action="store_false",
        dest="keep_alive",
        help="Exit immediately after pipeline verification",
    )
    args = parser.parse_args()

    # 1. Verify / Train
    run_training()

    # 2. Run Test Suite
    run_tests()

    # 3. Start Server
    server_process = start_server_background()

    # 4. Print System Info
    print_banner("SYSTEM READY & VERIFIED")
    print(f"""
Exact Terminal Commands to Run the System:
------------------------------------------
1. Activate virtual environment:
   Windows:  .venv\\Scripts\\activate
   Linux/Mac: source .venv/bin/activate

2. Start the FastAPI & WebSocket Server:
   uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

Access URLs:
------------
* Live Telemetry Dashboard:  {DASHBOARD_URL}
* Alternative Dashboard URL: {SERVER_URL}/
* REST API Health Status:    {HEALTH_URL}
* Interactive OpenAPI Docs:  {SERVER_URL}/docs
* WebSocket Live Stream:     ws://localhost:8000/ws/live/PATIENT_001
""")

    if args.keep_alive:
        print("Server is actively running in background mode. Press Ctrl+C to terminate.")
        try:
            while server_process.poll() is None:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down background server...")
            server_process.terminate()
            server_process.wait()
            print("Server stopped cleanly.")
    else:
        print("Verification complete. Terminating background check server.")
        server_process.terminate()
        server_process.wait()


if __name__ == "__main__":
    main()
