import streamlit as st
import time
import sys
import os
import pandas as pd
import json

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from notebooks.pipeline_runner import run_pipeline

# -------------------------------------------------------
# Persistent Metrics Storage
# -------------------------------------------------------
METRICS_FILE = os.path.join(ROOT_DIR, "data", "training_metrics.json")


def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {
        "accuracy": "—",
        "analysis_time": "—",
        "last_update_time": "—"
    }


def save_metrics(accuracy, analysis_time, last_update_time):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "analysis_time": analysis_time,
            "last_update_time": last_update_time
        }, f)


def show_local_model_update():

    # -------------------------------------------------------
    # Load persisted metrics
    # -------------------------------------------------------
    stored_metrics = load_metrics()

    if "accuracy" not in st.session_state:
        st.session_state.accuracy = stored_metrics["accuracy"]

    if "analysis_time" not in st.session_state:
        st.session_state.analysis_time = stored_metrics["analysis_time"]

    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = stored_metrics["last_update_time"]

    if "progress" not in st.session_state:
        st.session_state.progress = 0

    # -------------------------------------------------------
    # CSS
    # -------------------------------------------------------
    st.markdown("""
        <style>
        .metric-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .step-wrapper {
            display: flex;
            justify-content: space-between;
            margin: 25px 0;
            gap: 8px;
        }
        .arrow-step {
            flex: 1;
            background: #2563eb;
            color: white;
            padding: 12px 5px;
            text-align: center;
            font-weight: bold;
            clip-path: polygon(90% 0%, 100% 50%, 90% 100%, 0% 100%, 10% 50%, 0% 0%);
        }
        .step-title { font-size: 11px; display: block; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔄 Federated Model Training")

    # -------------------------------------------------------
    # File Upload
    # -------------------------------------------------------
    with st.container(border=True):
        st.subheader("📁 Data Source")
        uploaded_file = st.file_uploader(
            "Upload CSV file for federated training",
            type=["csv"]
        )
        if uploaded_file:
            st.success(f"File '{uploaded_file.name}' uploaded successfully")

    st.markdown("---")

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metric-box"><p>Accuracy</p>'
            f'<h2>{st.session_state.accuracy}</h2></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-box"><p>Last Update</p>'
            f'<h2>{st.session_state.last_update_time}</h2></div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="metric-box"><p>Analysis Time</p>'
            f'<h2>{st.session_state.analysis_time}</h2></div>',
            unsafe_allow_html=True
        )

    # -------------------------------------------------------
    # Live Status (Complete Section)
    # -------------------------------------------------------
    st.subheader("📊 Live Status")

    with st.container(border=True):
        if uploaded_file:
            st.write("**Current Phase:** Federated Local Training → Secure Upload")
            st.progress(st.session_state.progress)
            st.code(
                ">>> Loading hospital data\n"
                ">>> Running federated local update\n"
                ">>> Applying privacy constraints\n"
                ">>> Sending update to global server",
                language="python"
            )
        else:
            st.info("Upload a CSV file to begin federated training")

    # -------------------------------------------------------
    # Execution
    # -------------------------------------------------------
    if st.button(" Run Training Round", type="primary", disabled=uploaded_file is None):

        start_time = time.time()
        st.session_state.progress = 20

        with st.spinner("Running Federated Training Pipeline..."):

            # Save uploaded CSV to pipeline data location
            data_dir = os.path.join(ROOT_DIR, "data")
            os.makedirs(data_dir, exist_ok=True)

            csv_path = os.path.join(data_dir, "uploaded_local_data.csv")
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.progress = 50

            original_cwd = os.getcwd()
            os.chdir(os.path.join(ROOT_DIR, "notebooks"))

            try:
                # Pipeline reads CSV internally
                accuracy = run_pipeline()
            finally:
                os.chdir(original_cwd)

            st.session_state.progress = 90
            end_time = time.time()

            # -------------------------------------------------------
            # Real metrics + persistence
            # -------------------------------------------------------
            accuracy_str = f"{accuracy:.2%}"
            analysis_time_str = f"{end_time - start_time:.2f} sec"
            last_update_str = time.strftime("%H:%M:%S")

            st.session_state.accuracy = accuracy_str
            st.session_state.analysis_time = analysis_time_str
            st.session_state.last_update_time = last_update_str

            save_metrics(
                accuracy=accuracy_str,
                analysis_time=analysis_time_str,
                last_update_time=last_update_str
            )

            st.session_state.progress = 100
            st.success("✅ Global Model Updated Successfully")
            st.rerun()