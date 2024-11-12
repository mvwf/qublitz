import streamlit as st
def main():
    st.title ("Qublitz")
    st.write("Welcome to Qublitz! A quantum computing game where you can learn about quantum computing and quantum gates.")
    st.page_link("home.py", label="Home", icon="🏠")
    st.page_link("pages/1_free_play.py", label="Free Play", icon="1️⃣")
    st.page_link("pages/2_gates_challenge.py", label="Gates Challenge", icon="2️⃣")
    st.page_link("pages/3_step_pulse_challenge.py", label="Step Pulse Challenge", icon="3️⃣")
    st.page_link("pages/4_custom_qubit_query.py", label="Custom Qubit Query", icon="4️⃣")
    st.page_link("pages/5_neutral_atoms.py", label=" Neutral Atoms", icon="🔬")
    st.page_link("pages/6_IQ_demo.py", label="IQ Mixing", icon="〰️")
    st.page_link("pages/7_qubit_measurement.py", label = "Qubit Measurement", icon="📏")
if __name__ == "__main__":
    main()