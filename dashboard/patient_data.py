import streamlit as st
import pandas as pd
from datetime import datetime
import random
import os

DATA_PATH = "data/patient_history.csv"

def render_patient_data():
    st.title("🧑‍⚕️ Patient Data Input")

    with st.form("patient_form"):
        age = st.number_input("Age", 1, 120)
        creatinine = st.number_input("Creatinine", 0.0, 10.0)
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        submit = st.form_submit_button("Predict")

    if submit:
        prediction = random.choice(["Low Risk", "High Risk"])
        st.success(f"Prediction: {prediction}")

        record = {
            "Doctor": st.session_state.username,
            "Age": age,
            "Creatinine": creatinine,
            "Diabetes": diabetes,
            "Prediction": prediction,
            "Timestamp": datetime.now()
        }

        df = pd.DataFrame([record])
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, mode="a", header=not os.path.exists(DATA_PATH), index=False)
