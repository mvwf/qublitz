import streamlit as st
from PIL import Image
def main():
    # Display main image and logo side by side
    main_img = Image.open("images/qublitz.png")
    logo_img = Image.open("images/logo.png")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(main_img, width=400)
    with col2:
        st.image(logo_img, width=263)
    st.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)
    st.write("Welcome to Qublitz!")
    st.page_link("home.py", label="Home", icon="🏠")
    st.page_link("pages/sonify.py", label="Sonify Images", icon="🎵")
    st.page_link("pages/qubit_simulator.py", label="Qubit Simulator", icon="🎮")
    st.page_link("pages/custom_qubit_query.py", label="Custom Qubit Query", icon="🔬")
    st.page_link("pages/IQ_mixer.py", label="IQ Mixing", icon="〰️")
    st.page_link("pages/EP_TPD_exploration.py", label="Exceptional Point and Transmission Peak Degeneracy Exploration", icon="🎯")
if __name__ == "__main__":
    main()
