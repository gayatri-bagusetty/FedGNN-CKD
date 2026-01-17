import streamlit as st
from database import verify_user # Importing the new database function

def login():
    st.set_page_config(page_title="Clinical Login", page_icon="🔐", layout="wide")

    # ---------- SESSION ----------
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = "Doctor"

    # ---------- CSS (MOVED UP + MINIMIZED HEADER) ----------
    st.markdown("""
    <style>
    /* 1. Shrink the header/deploy area significantly */
    header {
        height: 2rem !important;
        background-color: transparent !important;
    }
    
    /* 2. Remove the top colorful decoration line */
    [data-testid="stDecoration"] {
        display: none;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        height: 100%;
        overflow: hidden !important;
    }

    /* 3. Remove all padding from the top of the main container */
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
        padding: 15px 25px;
        background-color: #eef6fb;
        border-radius: 10px;
        border: 1px solid #e5eef7;
        margin: auto;
        
        /* 4. NEGATIVE MARGIN: Pulls the box up into the header space */
        margin-top: -20px; 
        
        text-align: center;
        z-index: 999;
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
        margin-bottom: 10px;
    }

    .title {
        font-size: 22px;
        font-weight: 600;
        color: #1f6feb;
        margin-bottom: 5px;
        text-align: center
    }

    .subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 15px;
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
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- CENTER ----------
    _, center, _ = st.columns([1, 1.3, 1])

    with center:
        # Extra spacer to fine-tune the "Lift" 
        # (Remove this if it's still too low)
        st.write("") 

        with st.container():
            st.markdown("<div class='square-box'>"
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

            user_id = st.text_input("ID")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                user_data = verify_user(role, user_id, password)
                if user_data:
                    st.session_state.logged_in = True
                    # Pulling the name we just added to MySQL
                    st.session_state.full_name = user_data.get('full_name', 'Doctor')
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.markdown("</div>", unsafe_allow_html=True)