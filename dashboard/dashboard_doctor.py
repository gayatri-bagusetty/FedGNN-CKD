import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys
from database import save_patient_data, fetch_all_patients
def doctor_dashboard():
    # --- 1. Page Configuration ---
    st.set_page_config(page_title="Doctor Portal", layout="wide")

    db_user_name = st.session_state.get('full_name', 'Doctor')

    # --- 2. CSS ---
    st.markdown("""
        <style>
        header { height: 3.5rem !important; background-color: transparent !important; }
        [data-testid="stDecoration"] { display: none; }
        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem !important; 
            margin-top: -30px !important; 
            padding-bottom: 2rem !important;
        }
        .welcome-title {
            font-size: 38px !important;
            font-weight: bold !important;
            margin-bottom: 5px !important;
            color: #1f2937;
            display: block;
        }
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        section[data-testid="stSidebar"] > div { overflow: hidden !important; }
        .stButton > button {
            width: 100% !important;
            border-radius: 8px !important;
            border: 1px solid #d1d5db !important;
            padding: 10px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- 3. Data Initialization ---
    record_cols = ["Sl.No", "Time of Entry", "Patient Name", "Age", "BP", "Creatinine", "Hemoglobin", "Result"]
    if 'patient_db' not in st.session_state:
        st.session_state.patient_db = pd.DataFrame(columns=record_cols)
    if 'admin_page' not in st.session_state:
        st.session_state.admin_page = "Dashboard"

    # --- 4. Sidebar ---
    with st.sidebar:
        st.title("🩺 NephroCare AI")
        st.write(f"Logged in as: **{db_user_name}**")
        st.markdown("---")
        if st.button("📊 Dashboard"): st.session_state.admin_page = "Dashboard"
        if st.button("🗂️ Patient Longitudinal Records"): st.session_state.admin_page = "Records" 
        if st.button("🔬 Patient Analysis"): st.session_state.admin_page = "Analysis"
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    choice = st.session_state.admin_page
    
    if choice == "Dashboard":
        st.markdown(f'<span class="welcome-title"><br>Welcome back, {db_user_name}! 👋</span>', unsafe_allow_html=True)
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.container(border=True):
                st.subheader("📖 System User Manual & Guidelines")
                st.markdown("1. Patient Analysis\n2. Real-time Prediction\n3. XAI Insights\n4. Record Keeping")
        with col2:
            st.info(f"**Quick Stats**\n\nTotal Patients Analyzed: {len(st.session_state.patient_db)}")

    elif choice == "Analysis":
        st.subheader("🔬 Clinical Diagnostic Analysis")
        with st.form("ckd_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                age = st.number_input("Age", 1, 120, 45)
                bp = st.number_input("Blood Pressure", 50, 200, 80)
                sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
                al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
                su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
                rbc = st.selectbox("RBC", ["normal", "abnormal"])
            with col2:
                pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
                pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
                ba = st.selectbox("Bacteria", ["notpresent", "present"])
                bgr = st.number_input("Blood Glucose Random", 20, 500, 120)
                bu = st.number_input("Blood Urea", 1, 400, 40)
                sc = st.number_input("Serum Creatinine", 0.0, 15.0, 1.2)
            with col3:
                sod = st.number_input("Sodium", 100, 170, 135)
                pot = st.number_input("Potassium", 2.0, 8.0, 4.5)
                hemo = st.number_input("Hemoglobin", 3.0, 18.0, 12.0)
                pcv = st.number_input("Packed Cell Volume", 10, 60, 40)
                wbcc = st.number_input("WBC Count", 2000, 20000, 8000)
                rbcc = st.number_input("RBC Count (m/uL)", 2.0, 8.0, 4.5)
            with col4:
                htn = st.selectbox("Hypertension", ["No", "Yes"])
                dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
                cad = st.selectbox("CAD", ["No", "Yes"])
                appet = st.selectbox("Appetite", ["good", "poor"])
                pe = st.selectbox("Pedal Edema", ["No", "Yes"])
                ane = st.selectbox("Anemia", ["No", "Yes"])
            submitted = st.form_submit_button("Run Diagnostic Analysis", type="primary")

        if submitted:
                try:
                    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    model_path = os.path.join(root, "data", "models", "global_model.pth")
                    scaler_path = os.path.join(root, "data", "processed", "scaler.pkl")

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
                
                    st.markdown("---")
                    res_col, xai_col = st.columns([1, 1.2])

                    with res_col:
                        st.markdown("### Diagnosis")
                        border_color = "#700e18" if results["probability"] >= 0.50 else "#13772b"
                        st.markdown(f"""
                            <div style="box-shadow: 0 4px 10px rgba(0,0,0,0.1); padding: 25px; border-radius: 10px; border-left: 10px solid {border_color}; background-color: white;">
                            <p style="margin:0; font-size: 14px; color: #666;">Current Prediction</p>
                            <h2 style="margin:0; color: {border_color};">{results['prediction']}</h2>
                            <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                            <p style="margin:0; font-size: 14px; color: #666;">Model Confidence</p>
                            <h3 style="margin:0;">{results['probability']:.2%}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        save_patient_data(raw_patient_data, results["prediction"])
                        st.markdown(f"""
                            <div style="
                                background-color: #d4edda;  /* light green */
                                color: #155724;             /* dark green text */
                                padding: 15px 20px;
                                border-radius: 8px;
                                border: 1px solid #c3e6cb;
                                font-weight: 500;
                                font-size: 16px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            ">
                            Patient record saved to database successfully!
                            </div>
                        """, unsafe_allow_html=True)


                    with xai_col:
                        st.markdown("### 🧠 Explanation & Insights")
                        
                        # Set colors based on prediction
                        is_high_risk = results["probability"] >= 0.50
                        accent_color = "#dc3545" if is_high_risk else "#28a745"
                        bg_color = "#fff5f5" if is_high_risk else "#f8fff9"

                        # --- THE SHADOW BOX ---
                        # result['report'] is now the dictionary from inference_xai.py
                        st.markdown(f"""
                            <div style="box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
                                        padding: 25px; 
                                        border-radius: 12px; 
                                        border-left: 10px solid {accent_color}; 
                                        background-color: {bg_color}; 
                                        margin-bottom: 25px;">
                                <h4 style="margin-top:0; color: {accent_color};">CLINICAL ANALYSIS REPORT</h4>
                                <p style="font-size: 1.1em; color: #333;"><strong>Status:</strong> {results['prediction']}</p>
                                <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 15px 0;">
                                <p style="font-size: 0.95em; color: #444; line-height: 1.6;">
                                    <strong>EXPLANATION:</strong><br>
                                    {results['report']['explanation']}
                                </p>
                                <p style="font-size: 0.95em; color: #444;">
                                    <strong>PRIMARY DRIVERS:</strong> {results['report']['primary_drivers']}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                        # --- BIOMARKER INFLUENCE BOX ---
                        st.markdown(f"""
                            <div style="box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
                                        padding: 25px; 
                                        border-radius: 12px; 
                                        background-color: {bg_color}; 
                                        border-left: 1px solid #f0f2f6;">
                                <h4 style="margin-top:0; color: #333;">📊 Biomarker Influence</h4>
                                <p style="font-size: 0.85em; color: #666; margin-bottom: 20px;">Impact of features on the model outcome:</p>
                        """, unsafe_allow_html=True)
                        
                        for feat, score in results["top_features"]:
                            st.write(f"**{feat.upper()}**")
                            st.progress(min(max(float(score), 0.0), 1.0))
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"System Error: {str(e)}")

    elif choice == "Records":
        st.subheader("🗂️ Patient Longitudinal Records")

        # Column order
        final_cols = [
            "Sl.No", "TimeOfEntry",
            "age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu",
            "sc","sod","pot","hemo","pcv","wbcc","rbcc","htn","dm","cad",
            "appet","pe","ane",
            "Result"
        ]

        # Fetch patient data directly from DB
        patients = fetch_all_patients()

        if patients:
            # Convert to DataFrame
            df = pd.DataFrame(patients)

            # Add Sl.No
            df.insert(0, "Sl.No", range(1, len(df) + 1))

            # Reorder columns (only those that exist)
            df = df[[c for c in final_cols if c in df.columns]]

            # Format numeric columns
            int_cols = ["age", "bp", "al", "su", "bgr", "bu", "pcv", "wbcc", "rbcc"]
            float_cols = ["sg", "sc", "sod", "pot", "hemo"]

            for c in int_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").round(0)

            for c in float_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

            # Row coloring based on Result
            def highlight_row(row):
                if row.get("Result") == "CKD":
                    return ["background-color: #f8d7da"] * len(row)  # reddish
                elif row.get("Result") == "Non-CKD":
                    return ["background-color: #d4edda"] * len(row)  # greenish
                else:
                    return [""] * len(row)

            # Display styled table
            st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)

            # CSV download
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Records as CSV",
                data=csv_data,
                file_name="patient_records.csv",
                mime="text/csv"
            )
        else:
            st.info("No patient data available.")
