import streamlit as st

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None


USERS = {
    "doctor": {"username": "doctor", "password": "doctor123"},
    "admin": {"username": "admin", "password": "admin123"},
}


def login():
    placeholder = st.empty()

    with placeholder.form(key="login_form_unique"):
        st.subheader("🔐 Dashboard Login")

        role = st.selectbox("Login as", ["Doctor", "Admin"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        submit = st.form_submit_button("Login")

    if submit:
        role_key = role.lower()

        if (
            username == USERS[role_key]["username"]
            and password == USERS[role_key]["password"]
        ):
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.username = username

            placeholder.empty()
            st.success(f"{role} login successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")