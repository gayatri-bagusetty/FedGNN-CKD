import streamlit as st
from auth import login

from dashboard_admin import render_admin_dashboard
from dashboard_doctor import render_doctor_dashboard
from patient_data import render_patient_data
from patient_history import render_patient_history
from local_model_update import render_local_model_update

st.set_page_config(page_title="Clinical FL Dashboard", layout="wide")

# ------------------------------------
# SESSION INITIALIZATION
# ------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "page" not in st.session_state:
    st.session_state.page = None

# ------------------------------------
# LOGIN PAGE
# ------------------------------------
if not st.session_state.logged_in:
    login()
    if st.session_state.logged_in:
        # Redirect user to their default dashboard after login
        if st.session_state.role == "doctor":
            st.session_state.page = "doctor_dashboard"
        elif st.session_state.role == "admin":
            st.session_state.page = "admin_dashboard"
    st.stop()  # Stop here until login is done

# ------------------------------------
# ROLE-BASED SIDEBAR
# ------------------------------------
with st.sidebar:
    st.title("🧭 Navigation")

    if st.session_state.role == "doctor":
        if st.button("📊 Doctor Dashboard"):
            st.session_state.page = "doctor_dashboard"
        if st.button("🧑‍⚕️ Patient Data Input"):
            st.session_state.page = "patient_input"
        if st.button("📜 Patient History"):
            st.session_state.page = "history"

    elif st.session_state.role == "admin":
        if st.button("📊 Admin Dashboard"):
            st.session_state.page = "admin_dashboard"
        if st.button("🔄 Local Model Update"):
            st.session_state.page = "model_update"

    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# ------------------------------------
# PAGE ROUTING
# ------------------------------------
page = st.session_state.page

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