import streamlit as st
import pandas as pd
from datetime import datetime

def doctor_dashboard():
    # --- 1. Page Configuration ---
    st.set_page_config(page_title="Doctor Portal", layout="wide")

    # Get dynamic name from session state
    db_user_name = st.session_state.get('full_name', 'Doctor')

    # --- 2. Corrected CSS ---
    st.markdown("""
        <style>
        /* 1. Reduce the white space and Deploy header height */
        header { 
            height: 3.5rem !important; 
            background-color: transparent !important; 
        }
        
        [data-testid="stDecoration"] { display: none; }

        /* 2. Adjust main container to prevent content from cutting off */
        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem !important; 
            margin-top: -30px !important; 
            padding-bottom: 2rem !important;
        }

        /* 3. Welcome title specific styling */
        .welcome-title {
            font-size: 38px !important;
            font-weight: bold !important;
            margin-bottom: 5px !important;
            color: #1f2937;
            display: block;
        }

        /* 4. Sidebar - STOP PAGE UP/DOWN SCROLLING */
        [data-testid="stSidebar"] { 
            background-color: #f0f2f6; 
        }
        
        /* This prevents the left panel from having its own scrollbar */
        section[data-testid="stSidebar"] > div {
            overflow: hidden !important;
        }
        
        /* Make sidebar buttons full width */
        .stButton > button {
            width: 100% !important;
            border-radius: 8px !important;
            border: 1px solid #d1d5db !important;
            padding: 10px !important;
        }
        
        /* Result Box Styling */
        .result-card {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 24px;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- 3. Data Initialization ---
    # Modified column list to exclude Patient Name and include Sl.No/Time
    record_cols = ["Sl.No", "Time of Entry", "Age", "BP", "SG", "Albumin", "Sugar", "Creatinine", "Hemoglobin", "Result"]
    
    if 'patient_db' not in st.session_state:
        st.session_state.patient_db = pd.DataFrame(columns=record_cols)
    
    if 'admin_page' not in st.session_state:
        st.session_state.admin_page = "Dashboard"

    # --- 4. Sidebar Navigation ---
    with st.sidebar:
        st.title("🩺 NephroCare AI")
        st.write(f"Logged in as: **{db_user_name}**")
        st.markdown("---")

        if st.button("📊 Dashboard"):
            st.session_state.admin_page = "Dashboard"
        if st.button("🗂️ Patient Longitudinal Records"):
            st.session_state.admin_page = "Records" 
        if st.button("🔬 Patient Analysis"):
            st.session_state.admin_page = "Analysis"
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    # --- 5. Main Panel Logic ---
    choice = st.session_state.admin_page

    # --- DASHBOARD PAGE ---
    if choice == "Dashboard":
        # Welcome message ONLY on Dashboard
        st.markdown(f'<span class="welcome-title"><br>Welcome back, {db_user_name}! 👋</span>', unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.container(border=True):
                st.subheader("📖 System User Manual & Guidelines")
                st.markdown("""
                1. **Patient Analysis:** Navigate to this section to input clinical parameters for a new diagnosis.
                2. **Real-time Prediction:** Once data is entered, the system uses a Random Forest model to predict CKD (Chronic Kidney Disease) status.
                3. **XAI Insights:** Review the 'Explainable AI' section to understand *why* the model reached its conclusion.
                4. **Record Keeping:** All analyzed patients are automatically saved to the **Clinical History Vault**.
                5. **Data Export:** You can download the entire history as a CSV file.
                """)
    
        with col2:
            st.info(f"**Quick Stats**\n\nTotal Patients Analyzed: {len(st.session_state.patient_db)}")

    # --- PATIENT ANALYSIS PAGE (NO CHANGES MADE TO LOGIC/INPUTS) ---
    elif choice == "Analysis":
        st.subheader("🔬 Clinical Diagnostic Analysis")
        with st.form("ckd_form"):
            col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Basic Info")
            name = st.text_input("Patient Full Name")
            age = st.number_input("Age", 1, 120, 45)
            bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
            htn = st.selectbox("Hypertension", ["No", "Yes"])

        with col2:
            st.markdown("##### Lab Results")
            alb = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
            sug = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])
            sc = st.number_input("Serum Creatinine", 0.0, 15.0, 1.2)
            hemo = st.number_input("Hemoglobin (gms)", 3.0, 18.0, 12.0)
            dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])

        with col3:
            st.markdown("##### Clinical Status")
            pcv = st.number_input("Packed Cell Volume (%)", 10, 60, 40)
            rbc = st.number_input("RBC Count (m/uL)", 2.0, 8.0, 4.5)
            cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
            pe = st.selectbox("Pedal Edema", ["No", "Yes"])
            ane = st.selectbox("Anemia", ["No", "Yes"])

        submitted = st.form_submit_button("Run Diagnostic Analysis", type="primary")

    if submitted:
        if not name:
            st.error("Please enter the Patient Name before proceeding.")
        else:
            # Map Binary Categorical Fields
            bin_map = {"No": 0, "Yes": 1}
            
            # Prepare Features for GNN (Total of 14 features for the Global Model)
            input_data = [
                age, bp, sg, alb, sug, sc, hemo, pcv, rbc,
                bin_map[htn], bin_map[dm], bin_map[cad], bin_map[pe], bin_map[ane]
            ]

            # Logic to store the data in the session state database
            new_row = {
                "Sl.No": len(st.session_state.patient_db) + 1,
                "Time of Entry": datetime.now().strftime("%H:%M:%S"), 
                "Age": age, 
                "BP": bp, 
                "SG": sg, 
                "Albumin": alb, 
                "Sugar": sug, 
                "Creatinine": sc, 
                "Hemoglobin": hemo,
                "PCV": pcv,
                "RBC": rbc,
                "HTN": htn,
                "DM": dm,
                "CAD": cad,
                "PE": pe,
                "ANE": ane
            }
            
            # Convert to DataFrame and update the record table
            st.session_state.patient_db = pd.concat([st.session_state.patient_db, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"Data for {name} has been successfully recorded in the system.")

    # --- RECORDS PAGE ---
    elif choice == "Records":
        st.subheader("🗂️ Patient Longitudinal Records")
        
        # Show table space
        st.dataframe(st.session_state.patient_db, use_container_width=True, hide_index=True)
        
        # If no data, show message inside/below table area
        if st.session_state.patient_db.empty:
            st.info("No records found yet.")