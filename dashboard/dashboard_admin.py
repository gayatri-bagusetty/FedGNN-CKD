import streamlit as st

def admin_dashboard():

    st.set_page_config(layout="wide")

    # ---------------- SESSION STATE ----------------
    if "admin_page" not in st.session_state:
        st.session_state.admin_page = "Dashboard"

    admin_name = st.session_state.get("username", "Admin")

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.title("🛠 Admin Panel")

        if st.button("📊 Dashboard"):
            st.session_state.admin_page = "Dashboard"

        if st.button("🔄 Local Model Update"):
            st.session_state.admin_page = "Local Model Update"

        if st.button("👨‍⚕️ Manage Doctors"):
            st.session_state.admin_page = "Manage Doctors"

        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    # ---------------- CSS ----------------
    st.markdown("""
    <style>
    .main { background-color: #F4F6FA; }

    .card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .metric {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .metric-icon { font-size: 28px; }

    .step {
        padding: 12px;
        border-radius: 30px;
        color: white;
        text-align: center;
        font-weight: 600;
    }

    .step1 { background: linear-gradient(90deg, #2DBEAA, #48D6C9); }
    .step2 { background: linear-gradient(90deg, #4C7EF3, #6FA8FF); }
    .step3, .step4, .step5 {
        background: linear-gradient(90deg, #A0AEC0, #CBD5E0);
    }
    .step6 { background: linear-gradient(90deg, #F6AD55, #ED8936); }
    .pending { background: linear-gradient(90deg, #F6AD55, #ECC94B); }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================
    # ================= DASHBOARD PAGE ========================
    # =========================================================
    if st.session_state.admin_page == "Dashboard":

        st.markdown(f"""
        <div class="card">
            <h2>Welcome, {admin_name} 👋</h2>
            <p>
                This admin dashboard allows you to monitor federated learning
                performance, manage model updates, and control doctor access
                across hospitals while maintaining patient privacy using
                Differential Privacy and Federated Graph Neural Networks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # ================ LOCAL MODEL UPDATE =====================
    # =========================================================
    elif st.session_state.admin_page == "Local Model Update":

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="card metric">
                <div class="metric-icon">✅</div>
                <div>
                    <b>Accuracy</b><br>
                    Noised Accuracy <b>98.7%</b><br>
                    ε = 2.0
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card metric">
                <div class="metric-icon">🕒</div>
                <div>
                    <b>Last Update</b><br>
                    2 mins ago
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card metric">
                <div class="metric-icon">⏱️</div>
                <div>
                    <b>Total Time</b><br>
                    15 secs
                </div>
            </div>
            """, unsafe_allow_html=True)

        step_cols = st.columns(7)
        steps = [
            ("Step 1", "step1"),
            ("Step 2", "step2"),
            ("Step 3", "step3"),
            ("Step 4", "step4"),
            ("Step 5", "step5"),
            ("Step 6", "step6"),
            ("Pending", "pending")
        ]

        for col, (label, style) in zip(step_cols, steps):
            with col:
                st.markdown(f"<div class='step {style}'>{label}</div>", unsafe_allow_html=True)

    # =========================================================
    # ================= MANAGE DOCTORS ========================
    # =========================================================
    elif st.session_state.admin_page == "Manage Doctors":

        st.markdown("""
        <div class="card">
            <h3>Add New Doctor</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.form("add_doctor_form"):
            col1, col2 = st.columns(2)

            with col1:
                doctor_name = st.text_input("Doctor Name")
                doctor_id = st.text_input("Doctor ID")
                department = st.text_input("Department")

            with col2:
                branch = st.text_input("Branch / Hospital")
                position = st.selectbox(
                    "Position",
                    ["Junior Doctor", "Senior Doctor", "Consultant", "Specialist"]
                )

            submitted = st.form_submit_button("Add Doctor")

            if submitted:
                st.success(f"Doctor {doctor_name} added successfully ✅")