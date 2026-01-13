import streamlit as st
from auth import login
from dashboard_doctor import doctor_dashboard
from dashboard_admin import admin_dashboard

st.set_page_config(page_title="Clinical FL System", layout="wide")

# ---------------- SESSION INIT ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- ROLE-BASED ROUTING ----------------
if st.session_state.role == "Doctor":
    doctor_dashboard()

elif st.session_state.role == "Admin":
    admin_dashboard()

else:
    st.error("Unauthorized access")