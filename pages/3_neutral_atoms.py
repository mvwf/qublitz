import streamlit as st

# Initialize session state for current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Define page content in functions
def home_page():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")
    if st.button("Go to Data Page"):
        st.session_state.current_page = 'Data'

def data_page():
    st.title("Data Page")
    st.write("Here's some interesting data.")
    if st.button("Return Home"):
        st.session_state.current_page = 'Home'

# Page navigation logic
pages = {
    'Home': home_page,
    'Data': data_page
}

# Execute the function corresponding to the current page
pages[st.session_state.current_page]()