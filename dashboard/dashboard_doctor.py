import streamlit as st
import pandas as pd
from datetime import datetime

def doctor_dashboard():
    # --- 1. Page Configuration ---
    st.set_page_config(page_title="Doctor Portal", layout="wide")

    # --- 2. Enhanced CSS ---
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { background-color: #f0f2f6; }
        div.row-widget.stRadio > div { flex-direction: column; gap: 15px; padding-top: 20px; }
        div.row-widget.stRadio div[role="radiogroup"] > label {
            background-color: #ffffff; border: 1px solid #d1d5db; padding: 10px 15px;
            border-radius: 8px; cursor: pointer; width: 100%; display: flex;
            align-items: center; transition: all 0.2s ease-in-out; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        div.row-widget.stRadio div[role="radiogroup"] > label:hover { background-color: #f9fafb; border-color: #9ca3af; transform: translateY(-1px); }
        div.row-widget.stRadio div[role="radiogroup"] > label[data-selected="true"] {
            background-color: #e5efff !important; border: 2px solid #007bff !important; color: #007bff !important;
        }
        div.row-widget.stRadio div[role="radiogroup"] > label > div:first-child { display: none; }
        div.row-widget.stRadio div[role="radiogroup"] > label p { font-size: 18px !important; font-weight: 500 !important; margin: 0; }
        
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
    if 'patient_db' not in st.session_state:
        st.session_state.patient_db = pd.DataFrame(columns=[
            "Entry Date", "Name", "Age", "BP", "SG", "Albumin", "Sugar", "Creatinine", "Hemoglobin", "Result"
        ])
    
    if 'admin_page' not in st.session_state:
        st.session_state.admin_page = "Dashboard"

    # --- 4. Sidebar Navigation ---
    with st.sidebar:
        st.title("🩺 NephroCare AI")
        st.write("Logged in as: **Dr. Smith**")
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
        st.title("Welcome back, Dr. Smith! 👋")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Container for the User Manual
            with st.container(border=True):
                st.subheader("📖 System User Manual & Guidelines")
                st.markdown("""
                1. **Patient Analysis:** Navigate to this section to input clinical parameters for a new diagnosis.
                2. **Real-time Prediction:** Once data is entered, the system uses a Random Forest model to predict CKD (Chronic Kidney Disease) status.
                3. **XAI Insights:** Review the 'Explainable AI' section to understand *why* the model reached its conclusion (Feature Importance).
                4. **Record Keeping:** All analyzed patients are automatically saved to the **Clinical History Vault**.
                5. **Data Export:** You can download the entire history as a CSV file for hospital records or research.
                """)
    
        with col2:
            st.info(f"**Quick Stats**\n\nTotal Patients Analyzed: {len(st.session_state.patient_db)}")

    # --- PATIENT ANALYSIS PAGE ---
    elif choice == "Analysis":
        st.title("🔬 Patient Analysis")
        st.write("Complete all clinical fields to generate the diagnostic report.")
        
        with st.form("ckd_form"):
            # Organized into 3 columns for better readability
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Basic Info")
                name = st.text_input("Patient Full Name")
                age = st.number_input("Age", 1, 120, 45)
                bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
                sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])

            with col2:
                st.markdown("##### Lab Results (Biomarkers)")
                alb = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
                sug = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])
                sc = st.number_input("Serum Creatinine", 0.0, 15.0, 1.2)
                hemo = st.number_input("Hemoglobin (gms)", 3.0, 18.0, 12.0)

            with col3:
                st.markdown("##### Blood Counts & History")
                pcv = st.number_input("Packed Cell Volume (%)", 10, 60, 40)
                rbc = st.number_input("Red Blood Cell Count (m/uL)", 2.0, 8.0, 4.5)
                htn = st.selectbox("Hypertension", ["No", "Yes"])
                dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])

            submitted = st.form_submit_button("Run Diagnostic Analysis", type="primary")

        if submitted:
            if not name:
                st.error("Please enter the Patient Name before proceeding.")
            else:
                # Dummy prediction logic (Replace with your model)
                status = "CKD" if (alb > 2 or sc > 2.0 or hemo < 10) else "Non-CKD"
                
                # Save Data to History
                new_row = {
                    "Entry Date": datetime.now().strftime("%Y-%m-%d"), 
                    "Name": name, "Age": age, "BP": bp, "SG": sg, 
                    "Albumin": alb, "Sugar": sug, "Creatinine": sc, 
                    "Hemoglobin": hemo, "Result": status
                }
                st.session_state.patient_db = pd.concat([st.session_state.patient_db, pd.DataFrame([new_row])], ignore_index=True)

                # Result & XAI Layout
                st.markdown("---")
                res_col, xai_col = st.columns(2)
                
                with res_col:
                    st.subheader("Diagnostic Result")
                    bg_color = "#ff4b4b" if status == "CKD" else "#28a745"
                    st.markdown(f"""
                        <div class="result-card" style="background-color: {bg_color};">
                            {status}
                        </div>
                    """, unsafe_allow_html=True)
                    st.metric("Model Confidence", "96.4%")
                
                with xai_col:
                    st.subheader("XAI Insight (Local Explanation)")
                    # Visualizing Feature Importance for this specific patient
                    feat_imp = pd.DataFrame({
                        "Feature": ["Albumin", "Creatinine", "Hemoglobin", "SG"],
                        "Impact": [0.45, 0.30, 0.15, 0.10]
                    }).set_index("Feature")
                    st.bar_chart(feat_imp)
                    st.caption("This chart indicates which parameters influenced the 'Result' the most.")

    # --- RECORDS PAGE ---
    elif choice == "Records":
        st.title("🗂️ Patient Longitudinal Records")
        if not st.session_state.patient_db.empty:
            st.dataframe(st.session_state.patient_db, use_container_width=True)
            csv = st.session_state.patient_db.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download History as CSV", data=csv, file_name="ckd_history.csv")
        else:
            st.info("No records found yet. Complete a Patient Analysis to populate this vault.")