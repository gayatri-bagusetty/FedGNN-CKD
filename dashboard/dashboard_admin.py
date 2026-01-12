import streamlit as st

def render_admin_dashboard():
    st.title("🛠️ Admin Dashboard")
    st.divider()

    st.subheader("System Overview")
    st.success("Federated learning system running normally")

    st.markdown("""
    **Admin Capabilities**
    - Monitor global model lifecycle
    - Trigger local model updates
    - Ensure privacy compliance
    """)