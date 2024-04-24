import streamlit as st
st.title ("Qublitz")
st.write("Welcome to Qublitz! A quantum computing game where you can learn about quantum computing and quantum gates.")
st.page_link("home.py", label="Home", icon="🏠")
st.page_link("pages/1_free_play.py", label="Free Play Mode", icon="1️⃣")
st.page_link("pages/2_challenge_mode.py", label="Challenge Mode", icon="2️⃣")
st.page_link("pages/3_neutral_atoms.py", label=" Neutral Atoms", icon="🔬")