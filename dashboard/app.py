import streamlit as st
from auth import login

from dashboard_admin import render_admin_dashboard
from dashboard_doctor import render_doctor_dashboard
from patient_data import render_patient_data
from patient_history import render_patient_history
from local_model_update import render_local_model_update

st.set_page_config(page_title="Clinical FL Dashboard", layout="wide")

# ------------------------------------
# SESSION INIT
# ------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------------------------
# SIDEBAR (ROLE-BASED)
# ------------------------------------
with st.sidebar:
    st.title("🧭 Navigation")

    if st.session_state.role == "doctor":
        if st.button("📊 Dashboard"):
            st.session_state.page = "doctor_dashboard"
        if st.button("🧑‍⚕️ Patient Data Input"):
            st.session_state.page = "patient_input"
        if st.button("📜 History"):
            st.session_state.page = "history"

    if st.session_state.role == "admin":
        if st.button("📊 Dashboard"):
            st.session_state.page = "admin_dashboard"
        if st.button("🔄 Local Model Update"):
            st.session_state.page = "model_update"

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# ------------------------------------
# PAGE ROUTING
# ------------------------------------
page = st.session_state.get("page", "")

if page == "doctor_dashboard":
    render_doctor_dashboard()

elif page == "admin_dashboard":
    render_admin_dashboard()

elif page == "patient_input":
    render_patient_data()

elif page == "history":
    render_patient_history()

elif page == "model_update":
    render_local_model_update()