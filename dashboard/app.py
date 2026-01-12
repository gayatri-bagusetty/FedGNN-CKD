import streamlit as st
import pandas as pd

from pipeline_controller import run_pipeline
from status_tracker import get_steps
from privacy_guard import sanitize_metrics
# from xai_engine import generate_xai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="Clinical Federated Learning Dashboard")

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.markdown("👤 Dashboard")

with col2:
    st.markdown("<h2 style='text-align:center;'>Clinical Federated Learning Dashboard</h2>", unsafe_allow_html=True)

with col3:
    st.button("Login / Logout")

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv","xlsx"])

st.sidebar.subheader("Select Options")
attributes = st.sidebar.multiselect(
    "Attributes",
    ["Age", "BP", "Creatinine", "Hemoglobin", "Glucose"]
)

columns = st.sidebar.multiselect(
    "Columns",
    ["Column A", "Column B", "Column C"]
)

queries = st.sidebar.multiselect(
    "Queries",
    ["Query 1", "Query 2"]
)

st.sidebar.subheader("Selected Requests")
st.sidebar.write(attributes + columns + queries)

# ---------------- MAIN AREA ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    if st.button("🚀 Run Federated Pipeline"):
        raw_metrics = run_pipeline()
        metrics = sanitize_metrics(raw_metrics)

        # ---------- KPI CARDS ----------
        k1, k2, k3 = st.columns(3)

        k1.metric("Accuracy (Noised)", f"{metrics['noised_accuracy']}%")
        k2.metric("Time of Update", metrics["time_update"])
        k3.metric("Time of Analysis", metrics["time_analysis"])

        # ---------- PIPELINE STEPS ----------
        st.subheader("Federated Learning Pipeline")
        step_cols = st.columns(6)

        for i, col in enumerate(step_cols, start=1):
            status = get_steps()[i]
            col.info(f"Step {i}\n{status}")

        # ---------- RESULTS ----------
        st.subheader("Results / Prediction")
        st.success("Global Model Ready")
        st.write(f"**Diagnosis:** {raw_metrics['diagnosis']}")
        st.write(f"**Noised Accuracy:** {metrics['noised_accuracy']}% (ε={metrics['epsilon']})")

        # ---------- XAI ----------
        # st.subheader("Explainable AI (XAI)")
        # for exp in generate_xai():
        #     st.write("•", exp)
else:
    st.info("Please upload a dataset to begin.")