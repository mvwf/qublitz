import streamlit as st
import requests
from PIL import Image
import os

API_BASE_URL = "http://127.0.0.1:8001"  # Your backend URL

def api_login():
    st.title("Qublitz Virtual Qubit Lab - Login")
    api_key = st.text_input("Enter API Key", type="password", key='api_key_input')
    login_clicked = st.button("Login")
    
    if login_clicked:
        try:
            r = requests.get(f"{API_BASE_URL}/me", headers={"X-API-Key": api_key}, timeout=5)
            r.raise_for_status()
            user_data = r.json()
            # Save in session_state
            st.session_state['api_key'] = api_key
            st.session_state['user_data'] = user_data
            st.success(f"Welcome {user_data['user']}!")
        except Exception as e:
            st.error(f"Login failed: {e}")

def main_app():
    user_data = st.session_state.get('user_data', {})
    if not user_data:
        st.error("User data not found. Please log in again.")
        return

    st.title("Qublitz Virtual Qubit Lab")

    # Load logo safely
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "images", "logo.png")
    try:
        logo = Image.open(logo_path)
        st.sidebar.image(logo, use_column_width=True)
    except Exception:
        st.sidebar.write("Logo not found")

    st.header('This app simulates the dynamics of a driven qubit (two-level system)')
    
    omega_q = user_data.get('omega_q')
    omega_rabi = user_data.get('omega_rabi')
    T1 = user_data.get('T1')
    T2 = user_data.get('T2')

    st.sidebar.write(f"Logged in as: **{user_data.get('user')}**")

    if st.sidebar.button("Logout"):
        # clear session_state keys on logout
        for key in ['api_key', 'user_data']:
            if key in st.session_state:
                del st.session_state[key]

    sim_mode = st.selectbox("Select Simulation Mode", ["Time Domain", "Frequency Domain"])

    # Your simulation UI and logic comes below (omitted for brevity)
    st.write(f"Using parameters: ω_q={omega_q}, Ω_rabi={omega_rabi}, T1={T1}, T2={T2}")
    st.write(f"Simulation Mode: {sim_mode}")

def run():
    # Show login form if not logged in, else main app
    if 'user_data' not in st.session_state:
        api_login()
    else:
        main_app()

if __name__ == "__main__":
    run()