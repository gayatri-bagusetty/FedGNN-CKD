import streamlit as st
import pandas as pd
import os
from datetime import datetime,timedelta
import streamlit as st
from system_state import load_state

# --- IMPORT FROM YOUR FIXED DATABASE FILE ---
from database import get_total_users, get_db_connection
from local_model_update import show_local_model_update

def admin_dashboard():
    # --- 1. CSS (Matched to Doctor Dashboard) ---
    st.markdown("""
        <style>
        header { height: 3.5rem !important; background-color: transparent !important; }
        [data-testid="stDecoration"] { display: none; }
        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem !important; 
            margin-top: -30px !important; 
            padding-bottom: 2rem !important;
        }
        .welcome-box {
            background-color: #E8F5E9; 
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border: 1px solid #C8E6C9;
            margin-bottom: 30px;
            display: inline-block;
            width: auto;
        }
        .welcome-text {
            font-size: 2rem;
            font-weight: bold;
            color: #263238;
            margin: 0;
        }
        .shadow-container {
            background-color: white;
            padding: 30px;
            border-radius: 25px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            border: 1px solid #f0f2f6;
            height: 100%;
        }
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        .stButton > button {
            width: 100% !important;
            border-radius: 8px !important;
            border: 1px solid #d1d5db !important;
            padding: 10px !important;
            margin-bottom: 5px;
        }
        .stats-card {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .stats-label { font-size: 0.9rem; font-weight: 500; margin-bottom: 5px; }
        .stats-value { font-size: 1.8rem; font-weight: bold; }
        .card-red { background-color: #FEE2E2; color: #991B1B; }
        .card-green { background-color: #DCFCE7; color: #166534; }
        .card-teal { background-color: #F0FDFA; color: #115E59; }
        .card-blue { background-color: #DBEAFE; color: #1E40AF; }
        </style>
        """, unsafe_allow_html=True)

    db_admin_name = st.session_state.get('full_name', 'Admin')
    
    if "admin_page" not in st.session_state:
        st.session_state.admin_page = "Dashboard"

    # --- SIDEBAR (UNTOUCHED AS REQUESTED) ---
    with st.sidebar:
        st.title("🛡️ Admin Portal")
        st.write(f"Logged in as: **{db_admin_name}**") 
        st.markdown("---")
        if st.button("📊 Dashboard"): st.session_state.admin_page = "Dashboard"
        if st.button("🔄 Local Model Update"): st.session_state.admin_page = "Local Model Update"
        if st.button("👨‍⚕️ Staff Management"): st.session_state.admin_page = "Staff Management"
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    choice = st.session_state.admin_page

    # --- MAIN CONTENT ---
    if choice == "Dashboard":
        st.markdown("<br>", unsafe_allow_html=True)
        # Welcome Box (Matched to Doctor)
        st.markdown(f'<div class="welcome-box"><span class="welcome-text">Welcome back, {db_admin_name}! 👋</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        col_manual, col_stats = st.columns([1.8, 1.2])

        with col_manual:
            # Shadow Container (Matched to Doctor)
            st.markdown(f"""
            <div class="shadow-container">
                <h2 style='margin-top:0;'>📖 Admin System Manual</h2>
                <p><b>1. Federated Management:</b> Use 'Local Model Update' to trigger GNN training across nodes while maintaining data residency.</p>
                <p><b>2. Security & Privacy:</b> All updates utilize Differential Privacy (ε=2.0) to ensure hospital-specific data remains anonymous.</p>
                <p><b>3. User Auditing:</b> Monitor and manage clinical staff credentials under 'Manage Doctors' to ensure secure portal access.</p>
                <p><b>4. Global Synchronization:</b> Track the global model version and active participating hospital nodes in real-time.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats:
            st.markdown("<h3 style='text-align: center; color: Teal;'>Quick Network Stats</h3>", unsafe_allow_html=True)
            db_user_count = get_total_users()
            # Load federated system state
            state = load_state()
            fed_round = state.get("federated_round", 0)
            hospitals = state.get("participating_hospitals", [])

            
            # Logic for 6-month date calculation
            last_update = datetime(2025, 1, 15) # Example: date of last update
            next_update = last_update + timedelta(days=182) # ~6 months
             
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f'<div class="stats-card card-red"><div class="stats-label">Total Records</div><div class="stats-value">{db_user_count}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stats-card card-teal"><div class="stats-label">Next Update</div><div class="stats-value" style="font-size: 28px;">{next_update.strftime('%b %d, %Y')  }</div></div>', unsafe_allow_html=True)
            with m_col2:
                st.markdown(f'<div class="stats-card card-green"><div class="stats-label">Federated Round</div><div class="stats-value">Round {fed_round}</div></div>',unsafe_allow_html=True)
                st.markdown(f'<div class="stats-card card-blue"><div class="stats-label">Participating Hospitals</div><div class="stats-value">{len(hospitals)}</div></div>',unsafe_allow_html=True)

    elif choice == "Local Model Update":
        show_local_model_update()

    elif choice == "Staff Management":
        st.title("👨‍⚕️ Clinical Staff Management")
        with st.container(border=True):
            st.subheader("Register New User (Doctor/Admin)")
            with st.form("admin_reg_form", clear_on_submit=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    role_input = st.selectbox("Role", ["Doctor", "Admin"])
                    full_name = st.text_input("Full Name")
                with col_b:
                    user_id = st.text_input("User ID")
                    password = st.text_input("Password", type="password")
                with col_c:
                    dept = st.text_input("Department", value="Nephrology")
                    hosp = st.text_input("Hospital")

                if st.form_submit_button("Add User to Database", type="primary"):
                    conn = get_db_connection()
                    if conn and full_name and user_id:
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO users (role, user_id, password, full_name) VALUES (%s,%s,%s,%s)", 
                                       (role_input, user_id, password, full_name))
                        conn.commit()
                        conn.close()
                        st.success("User registered!")
                        st.rerun()

        st.markdown("---")
        st.subheader("📋 Registered Users Directory")
        conn = get_db_connection()
        if conn:
            df = pd.read_sql("SELECT user_id, full_name, role FROM users", conn)
            st.dataframe(df, use_container_width=True, hide_index=True)
            conn.close()