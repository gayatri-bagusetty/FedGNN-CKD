import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime
# --- IMPORT THE NEW MODULE ---
from local_model_update import show_local_model_update

# --- 1. Database Connection Logic ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1997",  # Your specific password
            database="clinical_db"
        )
        return connection
    except Error as e:
        st.error(f"Database Connection Error: {e}")
        return None

def verify_user(role, user_id, password):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True) 
        query = "SELECT * FROM users WHERE role = %s AND user_id = %s AND password = %s"
        cursor.execute(query, (role, user_id, password))
        user = cursor.fetchone() 
        conn.close()
        return user 
    return None

def admin_dashboard():
    # --- 2. Page Configuration & Enhanced CSS ---
    st.set_page_config(page_title="Admin Panel | NephroCare AI", layout="wide")

    st.markdown("""
        <style>
        header { 
            height: 0rem !important; 
            background-color: transparent !important; 
        }
        [data-testid="stDecoration"] { display: none; }
        [data-testid="stMainBlockContainer"] {
            padding-top: 0rem !important; 
            margin-top: 20px !important; 
            padding-bottom: 2rem !important;
        }
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        .stButton > button {
            width: 100% !important;
            border-radius: 8px !important;
            border: 1px solid #d1d5db !important;
            padding: 10px !important;
            margin-bottom: 5px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 14px;
            border: 1px solid #e0e4e8;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            margin-top: 0px !important; 
            margin-bottom: 20px;
        }
        .welcome-title {
            font-size: 32px !important;
            font-weight: bold !important;
            color: #1f2937;
            margin-top: 0px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- 3. Session State & Navigation ---
    db_admin_name = st.session_state.get('full_name', 'Admin')
    
    if "admin_page" not in st.session_state:
        st.session_state.admin_page = "Dashboard"

    with st.sidebar:
        st.title("🛡️ Admin Portal")
        st.write(f"Logged in as: **{db_admin_name}**") 
        st.markdown("---")
        
        if st.button("📊 Dashboard"):
            st.session_state.admin_page = "Dashboard"
        if st.button("🔄 Local Model Update"):
            st.session_state.admin_page = "Local Model Update"
        if st.button("👨‍⚕️ Manage Doctors"):
            st.session_state.admin_page = "Manage Doctors"
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    # --- 4. Main Panel Logic ---
    choice = st.session_state.admin_page

    # --- DASHBOARD PAGE ---
    if choice == "Dashboard":
        st.markdown(f"""
        <div class="card">
            <span class="welcome-title">Welcome back, {db_admin_name}! 👋</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("System Guidelines")
                st.write("This portal manages the Federated Learning pipeline. Monitor model performance, track local updates from participating hospitals, and manage clinical staff access.")
                st.markdown("""
                - **Model Updates:** Use 'Local Model Update' to trigger GNN training.
                - **Doctor Access:** Add or remove clinical staff in 'Manage Doctors'.
                - **Privacy:** Updates are noise-injected (ε=2.0) for anonymity.
                """)
        with col2:
            # Fetch real-time count from DB
            conn = get_db_connection()
            doc_count = 0
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'Doctor'")
                doc_count = cursor.fetchone()[0]
                conn.close()
            st.info(f"**Global Model Version:** v4.2.1\n\n**Registered Doctors:** {doc_count}")

    # --- LOCAL MODEL UPDATE PAGE ---
    elif choice == "Local Model Update":
        # CALL THE IMPORTED FUNCTION HERE
        show_local_model_update()

    # --- MANAGE DOCTORS PAGE ---
    elif choice == "Manage Doctors":
        st.title("👨‍⚕️ Clinical Staff Management")
        
        # --- REGISTRATION FORM SECTION (Database Insert) ---
        with st.container(border=True):
            st.subheader("Register New Doctor / User")
            with st.form("doctor_reg_form", clear_on_submit=True):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    role_input = st.selectbox("Role", ["Doctor", "Admin"])
                    doc_name = st.text_input("Doctor/Admin Name")
                
                with col_b:
                    user_id = st.text_input("User ID (Unique)")
                    password = st.text_input("Password", type="password")
                
                with col_c:
                    dept = st.text_input("Department", value="Nephrology")
                    hosp = st.text_input("Hospital")
                    pos = st.text_input("Position")

                submit_btn = st.form_submit_button("Add User Details", type="primary")
                
                if submit_btn:
                    if doc_name and user_id and password:
                        conn = get_db_connection()
                        if conn:
                            try:
                                cursor = conn.cursor()
                                # Query matches your DB structure: role, user_id, password, full_name
                                query = "INSERT INTO users (role, user_id, password, full_name) VALUES (%s, %s, %s, %s)"
                                cursor.execute(query, (role_input, user_id, password, doc_name))
                                conn.commit()
                                st.success(f"User {doc_name} registered successfully in the database!")
                                st.rerun()
                            except Error as e:
                                st.error(f"Database Error: {e}")
                            finally:
                                conn.close()
                    else:
                        st.error("Please fill in Name, User ID, and Password.")

        # --- TABLE SECTION (Database Select) ---
        st.markdown("---")
        st.subheader("📋 Registered Users Directory")
        
        conn = get_db_connection()
        if conn:
            # Fetch latest data for display
            query = "SELECT user_id AS 'Registration ID', full_name AS 'Doctor Name', role AS 'Role' FROM users"
            df = pd.read_sql(query, conn)
            st.dataframe(df, use_container_width=True, hide_index=True)
            conn.close()
        else:
            st.warning("Unable to retrieve directory. Check database connection.")