import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys
from database import save_patient_data, fetch_all_patients,get_total_patients
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
        st.markdown("""
        <style>
        /* Main Container Styling */
        .welcome-box {
            background-color: #E8F5E9; /* Light green tint from image */
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border: 1px solid #C8E6C9;
            margin-bottom: 30px;
            display: inline-block;
            width: auto;
        }
        
        .shadow-container {
            background-color: white;
            padding: 30px;
            border-radius: 25px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            border: 1px solid #f0f2f6;
            height: 100%;
        }

        .welcome-text {
            font-size: 2rem;
            font-weight: bold;
            color: #263238;
            margin: 0;
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
        .stats-delta { font-size: 1rem; color: #d32f2f; }
        
        /* Individual Card Colors */
        .card-red { background-color: #FEE2E2; color: #991B1B; }
        .card-green { background-color: #DCFCE7; color: #166534; }
        .card-teal { background-color: #F0FDFA; color: #115E59; }
        .card-blue { background-color: #DBEAFE; color: #1E40AF; }
        </style>
        """, unsafe_allow_html=True)
        
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        # 1. Welcome Message in a Box (Top Row)
        st.markdown(f'<div class="welcome-box"><span class="welcome-text">Welcome back, {db_user_name}! 👋</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 2. Main Content (Two Columns)
        col_manual, col_stats = st.columns([1.8, 1.2])

        with col_manual:
            # System Manual in a Shadow Box
            st.markdown(f"""
            <div class="shadow-container">
                <h2 style='margin-top:0;'>📖 System Manual: How to Use</h2>
                <p><b>1. Patient Analysis:</b> Upload laboratory results (Creatinine, eGFR, etc.) to receive immediate risk scores and renal health assessments.</p>
                <p><b>2. Real-time Prediction:</b> Utilize our longitudinal engine to forecast potential disease progression and kidney function decline over time.</p>
                <p><b>3. XAI Insights:</b> Access Explainable AI modules to understand the specific clinical features (like blood pressure or age) driving the model's decisions.</p>
                <p><b>4. Record Keeping:</b> Securely manage and review historical patient data to track treatment efficacy and clinical history.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats:
            # Quick Stats Header and Grid (Matches Image 2 Style)
            st.markdown("<h3 style='text-align: center; color: Teal;'>Quick Stats</h3>", unsafe_allow_html=True)
        
            # Grid Layout for Stats
            m_col1, m_col2 = st.columns(2)
            db_patient_count = get_total_patients()
            with m_col1:
                st.markdown(f"""
                <div class="stats-card card-red">
                    <div class="stats-label">Total Analyzed</div>
                    <div class="stats-value">{db_patient_count}</div>
                </div>
                <div class="stats-card card-teal">
                    <div class="stats-label">Last AI Confidence</div>
                    <div class="stats-value">94.7%</div>
                </div>
            """, unsafe_allow_html=True)
            
            with m_col2:
                st.markdown("""
                <div class="stats-card card-green">
                    <div class="stats-label">High Risk Patients</div>
                    <div class="stats-value">156 <span class="stats-delta">↑</span></div>
                </div>
                <div class="stats-card card-blue">
                    <div class="stats-label">No. of Patients</div>
                    <div class="stats-value">20</div>
                </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
    elif choice == "Analysis":
        st.subheader("🔬 Clinical Diagnostic Analysis")
        with st.form("ckd_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                age = st.number_input("Age", 1, 120, 45)
                bp = st.number_input("Blood Pressure", 50, 200, 80)
                sg = st.number_input("Specific Gravity", min_value=1.001, max_value=1.035, value=1.020,step=0.005,format="%.3f")
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

                    with st.spinner("Processing through FedGNN-XAI Engine..."):
                        results = engine.run_diagnosis(raw_patient_data)
                
                    st.markdown("---")
                    res_col, xai_col = st.columns([1, 1.2])

                    with res_col:
                        st.markdown("### Diagnosis")
                        border_color = "#D81939" if results["probability"] >= 0.50 else "#18B78A"
                        st.markdown(f"""
                            <div style="box-shadow: 0 4px 10px rgba(0,0,0,0.1); padding: 60px; border-radius: 30px; border-block: 10px solid #5DBF9B; background-color: white;">
                            <p style="margin:0; font-size: 14px; color: #666;">Current Prediction</p>
                            <h2 style="margin:0; color: {border_color};">{results['prediction']}</h2>
                            <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                            <p style="margin:0; font-size: 14px; color: #666;">Model Confidence</p>
                            <h3 style="margin:0;">{results['probability']:.2%}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"       ")
                        save_patient_data(raw_patient_data, results["prediction"])
                        st.markdown(f"""
                            <div style="
                                background-color: #d4edda;  /* light green */
                                color: black;             /* dark green text */
                                padding: 15px 20px;
                                border-radius: 28px;
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

                        # --- THE SHADOW BOX ---
                        # result['report'] is now the dictionary from inference_xai.py
                        st.markdown(f"""
                            <div style="box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
                                        padding: 25px; 
                                        border-radius: 30px; 
                                        border-block: 10px solid #5DBF9B; 
                                        background-color: white; 
                                        margin-bottom: 25px;">
                                <h4 style="margin-top:0; color: Black;">CLINICAL ANALYSIS REPORT</h4>
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
                                        background-color: white; 
                                        border-left: 1px solid #f0f2f6;">
                                <h4 style="margin-top:0; color: #333;">📊 Biomarker Influence</h4>
                                <p style="font-size: 0.85em; color: #666; margin-bottom: 20px;">Impact of features on the model outcome:</p>
                        """, unsafe_allow_html=True)
                        
                        for feat, score in results["top_features"]:
                            # Your logic: Normalize and scale to 100
                            clean_score = min(max(float(score), 0.0), 1.0) * 100
                            st.markdown("""
                                <style>
                                    .biomarker-container {
                                        background-color: #ffffff;
                                        padding: 15px;
                                        border-radius: 12px;
                                        border: 1px solid #e0e0e0;
                                        margin-bottom: 10px;
                                    }
                                    .biomarker-label {
                                        font-weight: bold;
                                        color: #333;
                                        display: flex;
                                        justify-content: space-between;
                                        margin-bottom: 5px;
                                    }
                                    .biomarker-bar-bg {
                                        background-color: #f0f2f6;
                                        border-radius: 10px;
                                        width: 100%;
                                        height: 8px;
                                    }
                                    .biomarker-bar-fill {
                                        background: linear-gradient(90deg, #50C0A4, #55C0A1);
                                        height: 8px;
                                        border-radius: 10px;
                                    }
                                    .biomarker-score {
                                        color: #666;
                                        font-size: 0.85rem;
                                    }
                                </style>
                            """, unsafe_allow_html=True)
                            # Render the custom biomarker bar
                            st.markdown(f"""
                                <div class="biomarker-container">
                                    <div class="biomarker-label">
                                    <span>{feat}</span>
                                    <span class="biomarker-score">{clean_score:.1f}% Impact</span>
                                    </div>
                                    <div class="biomarker-bar-bg">
                                    <div class="biomarker-bar-fill" style="width: {clean_score}%;"></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
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
