import streamlit as st
import pandas as pd
from datetime import datetime

def admin_dashboard():
    # --- 1. Page Configuration ---
    st.set_page_config(page_title="Admin Panel | NephroCare AI", layout="wide")

    # --- 2. Enhanced CSS (Boxed Sidebar & Admin Cards) ---
    st.markdown("""
        <style>
        /* Sidebar Styling */
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        
        /* Boxed Radio Buttons for Sidebar */
        div.row-widget.stRadio > div { flex-direction: column; gap: 15px; padding-top: 20px; }
        div.row-widget.stRadio div[role="radiogroup"] > label {
            background-color: #ffffff; border: 1px solid #d1d5db; padding: 10px 15px;
            border-radius: 8px; cursor: pointer; width: 100%; display: flex;
            align-items: center; transition: all 0.2s ease-in-out; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        div.row-widget.stRadio div[role="radiogroup"] > label:hover { 
            background-color: #f9fafb; border-color: #9ca3af; transform: translateY(-1px); 
        }
        div.row-widget.stRadio div[role="radiogroup"] > label[data-selected="true"] {
            background-color: #e5efff !important; border: 2px solid #007bff !important; color: #007bff !important;
        }
        div.row-widget.stRadio div[role="radiogroup"] > label > div:first-child { display: none; }
        div.row-widget.stRadio div[role="radiogroup"] > label p { font-size: 18px !important; font-weight: 500 !important; margin: 0; }

        /* Admin Dashboard Cards */
        .card {
            background: white;
            padding: 20px;
            border-radius: 14px;
            border: 1px solid #e0e4e8;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .metric-icon { font-size: 28px; margin-right: 12px; }
        
        /* Progress Steps */
        .step {
            padding: 12px;
            border-radius: 30px;
            color: white;
            text-align: center;
            font-weight: 600;
            font-size: 14px;
        }
        .step1 { background: linear-gradient(90deg, #2DBEAA, #48D6C9); }
        .step2 { background: linear-gradient(90deg, #4C7EF3, #6FA8FF); }
        .step-inactive { background: linear-gradient(90deg, #A0AEC0, #CBD5E0); }
        .step-warning { background: linear-gradient(90deg, #F6AD55, #ED8936); }
        </style>
        """, unsafe_allow_html=True)

    # --- 3. Session State Initialization ---
    if "admin_page" not in st.session_state:
        st.session_state.admin_page = "Dashboard"
    
    admin_name = st.session_state.get("username", "Admin")

    # --- 4. Sidebar Navigation (Boxed Style) ---
    with st.sidebar:
        st.write(f"<p style='text-align: center;'>Welcome, <b>{admin_name}</b></p>", unsafe_allow_html=True)
        st.markdown("---")
        
        with st.sidebar:

            if st.button("📊 Dashboard"):
                st.session_state.admin_page = "Dashboard"

            if st.button("🔄 Local Model Update"):
                st.session_state.admin_page = "Local Model Update"

            if st.button("👨‍⚕️ Manage Doctors"):
                st.session_state.admin_page = "Manage Doctors"

            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.rerun()

    # --- 5. Main Panel Logic ---
    choice = st.session_state.admin_page

    # --- LOGOUT LOGIC ---
    if choice == "Logout":
        st.session_state.clear()
        st.rerun()

    # --- DASHBOARD PAGE ---
    elif choice == "Dashboard":
        
        st.markdown(f"""
        <div class="card">
            <h2>Welcome back, {admin_name} 👋</h2>
            <p style="color: #666;">
                This portal manages the Federated Learning pipeline. Monitor model performance, 
                track local updates from participating hospitals, and manage clinical staff access 
                while ensuring Differential Privacy compliance.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("System Guidelines")
                st.markdown("""
                - **Model Updates:** Use the 'Local Model Update' tab to trigger and monitor GNN training.
                - **Doctor Access:** Add or remove clinical staff in 'Manage Doctors'.
                - **Privacy:** All updates are noise-injected (ε=2.0) to maintain patient anonymity.
                """)
        with col2:
            st.info("**Global Model Version:** v4.2.1\n\n**Connected Clients:** 8 Hospitals")

    # --- LOCAL MODEL UPDATE ---
    elif choice == "Local Model Update":
        st.title("🔄 Federated Model Training")
        
        # Performance Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("""<div class="card"><b>Accuracy</b><br><span style="color:#28a745; font-size:20px;">98.7%</span><br><small>ε = 2.0 (DP)</small></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown("""<div class="card"><b>Last Update</b><br><span style="font-size:20px;">2 mins ago</span><br><small>Sync: Global Aggregator</small></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown("""<div class="card"><b>Compute Time</b><br><span style="font-size:20px;">15.4 secs</span><br><small>H100 Instance</small></div>""", unsafe_allow_html=True)

        st.subheader("Training Progress Pipeline")
        step_cols = st.columns(7)
        steps = [
            ("Pre-processing", "step1"),
            ("Local Training", "step2"),
            ("Noise Injection", "step-inactive"),
            ("Aggregation", "step-inactive"),
            ("Validation", "step-inactive"),
            ("Global Push", "step-warning"),
            ("Pending", "step-inactive")
        ]

        for col, (label, style) in zip(step_cols, steps):
            with col:
                st.markdown(f"<div class='step {style}'>{label}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("Trigger Global Re-Aggregation", type="primary"):
            st.toast("Aggregating local model weights...")

    # --- MANAGE DOCTORS ---
    elif choice == "Manage Doctors":
        st.title("👨‍⚕️ Manage Doctors")
        
        with st.container(border=True):
            st.subheader("Register New Clinical User")
            with st.form("add_doctor_form"):
                c1, c2 = st.columns(2)
                with c1:
                    d_name = st.text_input("Doctor Name")
                    d_id = st.text_input("Registration ID")
                    dept = st.text_input("Department", value="Nephrology")
                with c2:
                    hosp = st.text_input("Branch / Hospital")
                    pos = st.selectbox("Position", ["Junior Doctor", "Senior Doctor", "Consultant", "Specialist"])
                
                if st.form_submit_button("Add Doctor to System", type="primary"):
                    if d_name and d_id:
                        st.success(f"Doctor {d_name} (ID: {d_id}) has been granted access.")
                    else:
                        st.error("Please fill in the required fields.")