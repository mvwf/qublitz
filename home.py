import streamlit as st
from PIL import Image
def main():
    logo = Image.open("images/logo.png") 
    st.sidebar.image(logo)
    st.title ("Qublitz")
    st.write("Welcome to Qublitz!")
    st.page_link("home.py", label="Home", icon="🏠")
    st.page_link("pages/sonify.py", label="Sonify Images", icon="🎵")
    st.page_link("pages/qubit_simulator.py", label="Qubit Simulator", icon="🎮")
    st.page_link("pages/custom_qubit_query.py", label="Custom Qubit Query", icon="🔬")
    st.page_link("pages/IQ_mixer.py", label="IQ Mixing", icon="〰️")
    st.page_link("pages/EP_TPD_exploration.py", label="Exceptional Point and Transmission Peak Degeneracy Exploration", icon="🎯")
if __name__ == "__main__":
    main()
