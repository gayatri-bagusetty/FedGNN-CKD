import streamlit as st
from datetime import datetime, timedelta

def render_doctor_dashboard():
    st.title("🧑‍⚕️ Doctor Dashboard")
    st.divider()

    last_update = datetime(2026, 1, 10)
    next_update = last_update + timedelta(days=30)

    c1, c2 = st.columns(2)
    c1.info(f"Last Global Model Update\n\n{last_update.strftime('%d %b %Y')}")
    c2.warning(f"Next Update Expected\n\n{next_update.strftime('%d %b %Y')}")

    st.subheader("📘 Usage Guide")
    st.markdown("""
    - Enter patient data to get CKD risk prediction.
    - Predictions use the global federated model.
    - Patient data is not stored centrally.
    - Local model details are hidden for privacy.
    """)