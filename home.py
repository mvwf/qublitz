import streamlit as st
from PIL import Image
def main():
    # Display main image and logo side by side
    main_img = Image.open("images/qublitz.png")
    logo_img = Image.open("images/logo.png")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(main_img, width=345)
    with col2:
        st.image(logo_img, width=263)
    st.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)
    st.write("Welcome to Qublitz!")
    
    st.page_link("home.py", label="Home", icon="🏠")
    st.page_link("pages/Sonify.py", label="Sonify Images", icon="🎵")
    st.page_link("pages/Qubit_Simulator.py", label="Qubit Simulator", icon="🎮")
    st.page_link("pages/Custom_Qubit_Query.py", label="Custom Qubit Query (Local/VPN)", icon="🧪")
    st.page_link("pages/IQ_mixer.py", label="IQ Mixing", icon="〰️")
    st.page_link("pages/EP_TPD_exploration.py", label="Exceptional Point and Transmission Peak Degeneracy Exploration", icon="🎯")
    st.page_link("pages/Laser_Heating_Calculator.py", label="Laser Heating Calculator", icon="⚡")
    st.page_link("pages/Dilution_Refrigerator_Noise_Explorer.py", label="Dilution Refrigerator Noise Explorer", icon="❄️")

if __name__ == "__main__":
    main()
