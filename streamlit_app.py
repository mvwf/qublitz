
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from glob import glob
from PIL import Image
import requests as req
import io
import datetime
import streamlit_authenticator as stauth
import streamlit as st
from datetime import date
# from slack_sdk import WebClient
# from slack_sdk.errors import SlackApiError
# from slack_bolt import App
import ssl
from datetime import timedelta
from ratelimit import limits, sleep_and_retry

ssl._create_default_https_context = ssl._create_unverified_context

# skidmore-datahub.us.reclaim.cloud
@sleep_and_retry
@limits(calls=1, period=timedelta(seconds=60).total_seconds())
def get_auth_header(email, password):
    """
    Authorizes a user through the SEGC using an input email and password string
    param: email
    param: password
    """

    url_signin = st.secrets["url_signin"]
    signin_response = req.post(url_signin, data={'email': email, 'password': password})
    return {'Authorization': "Bearer {}".format(signin_response.json()['accessToken'])}

@sleep_and_retry
@limits(calls=1, period=timedelta(seconds=60).total_seconds())
@st.cache_data
def load_data(resource_name, auth_header):
    '''
    Function to load the data, given a ResourceName, appropiately encoded based on the developed API.
    '''
    query_response = req.post(st.secrets["query_response_url"],
                        data={
                            "resourceName": f"{resource_name}",
                            "repository": "FITZLAB_CASSINI",
                            "SQL_Select_Fields": "*",
                            "SQL_Where_Fields": "",
                            "SQL_Where_Statement": ""
                        },
                        headers=auth_header)

    galaxy_data = pd.read_csv(io.StringIO(query_response.content.decode('utf-8')))
    return galaxy_data

def get_time_series(ResourceName, auth_header, phys_quantity, channel):
    '''
    Function to get the figure object for the desired physical quantity (T, R, P) for the desired (available) channel.
    '''

    data = load_data(ResourceName, auth_header)
    fig = px.line(data, x='time', y=f"{phys_quantity}",
            labels={
                        "time": "Hour",
                        f"temp": f"temp [K]",
                    },
            title=f"{phys_quantity} CH{channel}")
    return fig

def get_formatted_date(date_obj):
    '''
    Get the date object in a useful format to get a correcly formatted ResourceName, as designed in the API itself.
    '''
    year = str(date_obj.year)[2:]
    month = str(date_obj.month).zfill(2)
    day = str(date_obj.day).zfill(2)
    return f"{year}-{month}-{day}"

                    
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():

    auth_header = get_auth_header(st.secrets["auth_header_email"], st.secrets["auth_header_password"])

    fitzlab_logo = Image.open("images/fitz_lab_logo.png")
    saturn_image = Image.open("images/saturn.jpeg")

    st.sidebar.image(fitzlab_logo, use_column_width=True)
    st.sidebar.image(saturn_image, use_column_width=True)

    st.title('🪐 Cassini Remote Monitor')
    st.header('BlueFors Dilution Fridge @ FitzLab')

    st.sidebar.markdown('')

    def main():
        # opt_dict = {'': '', 'Temperature': 'T', 'Resistance': 'R'}
        # units_dict = {'Temperature': '[K]', 'Resistance': '[Ohms]'}

        current_date = st.radio('Do you want to monitor the current date?', ["Yes", "No, I want to check a previous date"], index=0)

        if current_date == 'Yes':
            try:
                today = date.today()
                formatted_date = get_formatted_date(today)
                phys_quant_selected = st.selectbox("Select the temperature channel you want to monitor",
                                                ["", '1', '2', '5'])
                
                if phys_quant_selected:
                    ResourceName = f'CH{phys_quant_selected} T {formatted_date}.log'
                    ## temp for temperature. Once I know how to access P, and R I will update accordingly.
                    fig = get_time_series(ResourceName, auth_header, 'temp', phys_quant_selected)
                    st.plotly_chart(fig)
            except:
                st.error("😕 Data not available. Please go and check the connections in the fridge!!!")

        else:
            try:
                history = st.date_input(
                        "Please select the date you want to monitor",
                        datetime.date(2023, 4, 14))
                
                formatted_date = get_formatted_date(history)

                phys_quant_selected = st.selectbox("Select the temperature channel you want to monitor for the selected date",
                                                ["", '1', '2', '5'])
                
                if phys_quant_selected:
                    ResourceName = f'CH{phys_quant_selected} T {formatted_date}.log'
                    ## temp for temperature. Once I know how to access P, and R I will update accordingly.
                    fig = get_time_series(ResourceName, auth_header, 'temp', phys_quant_selected)
                    st.plotly_chart(fig)
            except:
                st.error("😕 Date not available. Data is missing!")

        button = st.button('Send Alert (This can be automated)')
        if button:
            try:
                # Initialize the Slack app and client
                slack_app = App(token=st.secrets["bot_token"])
                slack_client = WebClient(token=st.secrets["web_client_token"])

                # Define the channel to send the message to
                channel = "#try_cassini_bot"

                # Send the message
                try:
                    response = slack_client.chat_postMessage(
                        channel=channel,
                        text="Hello from the Streamlit app!"
                    )
                    st.success("Message sent!")
                except SlackApiError as e:
                    st.error("😕 Error sending message: {}".format(e))
            except:
                st.error("😕 Error sending message!!")
        else:
            pass

    if __name__ == "__main__":
        main()