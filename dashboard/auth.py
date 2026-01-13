import streamlit as st

def login():
    st.set_page_config(page_title="Clinical Login", page_icon="🔐", layout="wide")

    # # ---------- SESSION ----------
    # if "logged_in" not in st.session_state:
    #     st.session_state.logged_in = False
    # if "role" not in st.session_state:
    #     st.session_state.role = "Doctor"

    USERS = {
        "Doctor": {"ID": "41568", "password": "doctor123"},
        "Admin": {"ID": "56987", "password": "admin123"},
    }

    # ---------- CSS ----------
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        height: 100%;
        overflow: hidden !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        overflow: hidden !important;
    }

    .stApp {
        background-color: #eef6fb;
    }

    .square-box {
        width: 300px;
        padding: 18px;
        background-color: #eef6fb;
        border-radius: 10px;
        border: 1px solid #e5eef7;
        margin: auto;
        margin-top: 60px;
        text-align: center;
    }


    .square-box .avatar {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: #1f6feb;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 32px;
        margin: auto;
        margin-bottom: 14px;
    }

    .title {
        font-size: 22px;
        font-weight: 600;
        color: #1f6feb;
        margin-bottom: 10px;
        text-align: center
    }

    .subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 18px;
        text-align: center
    }

    .stTextInput > div > div > input {
        text-align: left;
    }

    .stButton > button {
        width: 100%;
        height: 42px;
        background-color: #1f6feb;
        color: white;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- CENTER ----------
    _, center, _ = st.columns([1, 1.3, 1])

    with center:
        # REAL container (owns widgets)
        with st.container():
            # Visual square wrapper
            st.markdown("<div class='square-box'>" \
            "<div class='avatar'>👤</div>", unsafe_allow_html=True)
            st.markdown("<div class='title'>Choose Account Type</div>", unsafe_allow_html=True)

            role = st.radio(
                "",
                ["Doctor", "Admin"],
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state.role = role

            st.markdown(
                f"<div class='subtitle'>Hello {role.lower()}! Please login</div>",
                unsafe_allow_html=True
            )

            email = st.text_input("ID")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                user = USERS.get(role)
                if user and email == user["ID"] and password == user["password"]:
                    st.success("Login successful")
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            # Close visual wrapper
            st.markdown("</div>", unsafe_allow_html=True)