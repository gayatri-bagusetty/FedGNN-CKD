import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

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
    record_cols = ["Sl.No", "Time of Entry", "Patient Name", "Age", "BP", "Creatinine", "Hemoglobin", "Result"]
    
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
        st.markdown(f'<span class="welcome-title"><br>Welcome back, {db_user_name}! 👋</span>', unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.container(border=True):
                st.subheader("📖 System User Manual & Guidelines")
                st.markdown("""
                1. **Patient Analysis:** Navigate to this section to input clinical parameters for a new diagnosis.
                2. **Real-time Prediction:** Once data is entered, the system uses a Graph Neural Network (GNN) model to predict CKD status.
                3. **XAI Insights:** Review the 'Explainable AI' section to understand *why* the model reached its conclusion.
                4. **Record Keeping:** All analyzed patients are automatically saved to the **Clinical History Vault**.
                5. **Data Export:** You can view the entire history in the Records section.
                """)
    
        with col2:
            st.info(f"**Quick Stats**\n\nTotal Patients Analyzed: {len(st.session_state.patient_db)}")

    # --- ANALYSIS PAGE ---
    elif choice == "Analysis":
        st.subheader("🔬 Clinical Diagnostic Analysis")
        
        with st.form("ckd_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("##### 👤 Identification")
                name = st.text_input("Patient Full Name")
                age = st.number_input("Age", 1, 120, 45)
                bp = st.number_input("Blood Pressure", 50, 200, 80)
                sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
                al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
                su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
                
            with col2:
                st.markdown("##### 🧪 Microscopy/Lab I")
                rbc = st.selectbox("RBC", ["normal", "abnormal"])
                pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
                pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
                ba = st.selectbox("Bacteria", ["notpresent", "present"])
                bgr = st.number_input("Blood Glucose Random", 20, 500, 120)
                bu = st.number_input("Blood Urea", 1, 400, 40)

            with col3:
                st.markdown("##### 🧪 Lab Results II")
                sc = st.number_input("Serum Creatinine", 0.0, 15.0, 1.2)
                sod = st.number_input("Sodium", 100, 170, 135)
                pot = st.number_input("Potassium", 2.0, 8.0, 4.5)
                hemo = st.number_input("Hemoglobin", 3.0, 18.0, 12.0)
                pcv = st.number_input("Packed Cell Volume", 10, 60, 40)
                wbcc = st.number_input("WBC Count", 2000, 20000, 8000)

            with col4:
                st.markdown("##### 🏥 Clinical Assessment")
                rbcc = st.number_input("RBC Count (m/uL)", 2.0, 8.0, 4.5)
                htn = st.selectbox("Hypertension", ["No", "Yes"])
                dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
                cad = st.selectbox("CAD", ["No", "Yes"])
                appet = st.selectbox("Appetite", ["good", "poor"])
                pe = st.selectbox("Pedal Edema", ["No", "Yes"])
                ane = st.selectbox("Anemia", ["No", "Yes"])

            submitted = st.form_submit_button("Run Diagnostic Analysis", type="primary")

        if submitted:
            if not name:
                st.error("Please enter the Patient Name.")
            else:
                try:
                    # 1. Setup Paths
                    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    model_path = os.path.join(root, "data", "models", "global_model.pth")
                    scaler_path = os.path.join(root, "data", "processed", "scaler.pkl")

                    # 2. Initialize and Pass Data to Clinical Engine
                    from clinical_engine import ClinicalInferenceEngine
                    engine = ClinicalInferenceEngine(model_path, scaler_path)

                    raw_patient_data = {
                        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 
                        'pc': pc, 'pcc': pcc, 'ba': ba, 'bgr': bgr, 'bu': bu, 'sc': sc, 
                        'sod': sod, 'pot': pot, 'hemo': hemo, 'pcv': pcv, 'wbcc': wbcc, 
                        'rbcc': rbcc, 'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 
                        'pe': pe, 'ane': ane
                    }

                    with st.spinner("Processing through GNN-XAI Engine..."):
                        results = engine.run_diagnosis(raw_patient_data)
                    
                    # 3. UI Display of Results from Engine
                    is_ckd = (results["prediction"] == "CKD")
                    res_col, xai_col = st.columns([1,1])

                    with res_col:
                        st.markdown("### 🧪 Diagnosis")
                        st.metric("Status", results["prediction"])
                        st.metric("Confidence", f"{results['probability']:.2%}")

                    with xai_col:
                        st.markdown("### 🧠 Explanation")
                        st.success(results["report"])
                        for feat, score in results["top_features"]:
                            st.progress(float(score))
                            st.caption(f"{feat.upper()} : {score:.4f}")

                except Exception as e:
                    st.error(f"System Error: {str(e)}")

    # --- RECORDS PAGE ---
    elif choice == "Records":
        st.subheader("🗂️ Patient Longitudinal Records")
        st.dataframe(st.session_state.patient_db, use_container_width=True, hide_index=True)
        if st.session_state.patient_db.empty: 
            st.info("No records found yet.")