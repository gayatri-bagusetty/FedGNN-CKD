import streamlit as st
import pandas as pd
import os

DATA_PATH = "data/patient_history.csv"

def render_patient_history():
    st.title("📜 Patient History")

    if not os.path.exists(DATA_PATH):
        st.info("No history available.")
        return

    df = pd.read_csv(DATA_PATH)
    doctor_df = df[df["Doctor"] == st.session_state.username]

    st.dataframe(doctor_df)