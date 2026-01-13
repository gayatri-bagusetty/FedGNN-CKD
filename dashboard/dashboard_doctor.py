import streamlit as st

def doctor_dashboard():
    st.title("🩺 Doctor Dashboard")

    st.success("Welcome Doctor 👋")

    st.subheader("Doctor Actions")
    st.write("• Enter patient data")
    st.write("• Predict CKD")
    st.write("• View patient history")

    st.markdown("---")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()