'''
Authors:
    M. Weiner

Release Date: 
    V 1.0: 10/12/2024

'''
######################
###### IQ Mixer ######
# Author: Max Weiner #
### Date: 8/5/2024 ###
######################
# Modified by: Mattias Fitzpatrick
# Date: 07/27/2025


from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmam, Options
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
from matplotlib.colors import Normalize
from qutip import *
from scipy.fft import fftshift, fft, ifft
from matplotlib.widgets import Slider
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from PIL import Image

def main():
    st.title("IQ Mixing Demo")
    qublitz_logo = Image.open("images/qublitz.png")
    st.sidebar.image(qublitz_logo)
    logo = Image.open("images/logo.png") 
    st.sidebar.image(logo) # display logo on the side 
    
    st.sidebar.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)
    st.write("This app simulates the output of an IQ mixer given input and local oscillator signals. Adjust the sliders to see how the output changes.")
    col1, col2 = st.columns(2)
    with col1:
        A = st.slider('Input Amplitude [arb.]', 0.2, 20.0, 1.0, 0.1)
        phi = st.slider('Input Phase (°)', 0.0, 180.0, 1.0, 0.1)
        omega = st.slider('Input Frequency (GHz)', 1.0, 20.0, 1.0, 0.1)
    with col2:
        Al = st.slider('LO Amplitude [arb.]', 0.2, 20.0, 1.0, 0.1)
        phil = st.slider('LO Phase (°)', 0.0, 180.0, 1.0, 0.1)
        omegal = st.slider('LO Frequency (GHz)', 1.0, 20.0, 1.0, 0.1)

    nsteps = 100000
    t = np.linspace(0, 20 * 1 / min(omegal, omega), nsteps)
    d = max(t) / nsteps
    f = np.linspace(-1 / (2 * d), 1 / (2 * d), nsteps)

    ### Calculate Q from Mag, Phase, Frequency ###
    Q = A * np.sin(2 * np.pi * omega * t + np.deg2rad(phi))
    I = A * np.cos(2 * np.pi * omega * t + np.deg2rad(phi))
    fftinQ = np.abs(fftshift(fft(Q)))
    fftinI = np.abs(fftshift(fft(I)))

    Ql = Al * np.sin(2 * np.pi * omegal * t + np.deg2rad(phil))
    Il = Al * np.cos(2 * np.pi * omegal * t + np.deg2rad(phil))
    fftlocQ = np.abs(fftshift(fft(Ql)))
    fftlocI = np.abs(fftshift(fft(Il)))

    #################### Compute Input and Output of IQ Mixer ##################
    Qout = Q * Ql
    fftQout = np.abs(fftshift(fft(Qout)))
    Iout = I * Il
    fftIout = np.abs(fftshift(fft(Iout)))

    ############################## Separate I and Q Plots with Titles #######################################
    # 3 rows × 2 cols: each subplot for I or Q signal
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Input FFT I", "Input FFT Q",
            "LO FFT I", "LO FFT Q",
            "Output FFT I", "Output FFT Q"
        ),
        horizontal_spacing=0.15,
        vertical_spacing=0.2
    )

    # Input FFT I
    fig.add_trace(go.Scatter(x=f, y=fftinI, mode='lines', line=dict(color='black')), row=1, col=1)
    fig.update_xaxes(title_text='Frequency [GHz]', range=[0, omega + 5], row=1, col=1)
    fig.update_yaxes(title_text='Amplitude [arb.]', row=1, col=1)

    # Input FFT Q
    fig.add_trace(go.Scatter(x=f, y=fftinQ, mode='lines', line=dict(color='red')), row=1, col=2)
    fig.update_xaxes(title_text='Frequency [GHz]', range=[0, omega + 5], row=1, col=2)
    fig.update_yaxes(title_text='Amplitude [arb.]', row=1, col=2)

    # LO FFT I
    fig.add_trace(go.Scatter(x=f, y=fftlocI, mode='lines', line=dict(color='black')), row=2, col=1)
    fig.update_xaxes(title_text='Frequency [GHz]', range=[0, omegal + 5], row=2, col=1)
    fig.update_yaxes(title_text='Amplitude [arb.]', row=2, col=1)

    # LO FFT Q
    fig.add_trace(go.Scatter(x=f, y=fftlocQ, mode='lines', line=dict(color='red')), row=2, col=2)
    fig.update_xaxes(title_text='Frequency [GHz]', range=[0, omegal + 5], row=2, col=2)
    fig.update_yaxes(title_text='Amplitude [arb.]', row=2, col=2)

    # Output FFT I
    fig.add_trace(go.Scatter(x=f, y=fftIout, mode='lines', line=dict(color='black')), row=3, col=1)
    fig.update_xaxes(title_text='Frequency [GHz]', range=[0, omegal + omega + 5], row=3, col=1)
    fig.update_yaxes(title_text='Amplitude [arb.]', row=3, col=1)

    # Output FFT Q
    fig.add_trace(go.Scatter(x=f, y=fftQout, mode='lines', line=dict(color='red')), row=3, col=2)
    fig.update_xaxes(title_text='Frequency [GHz]', range=[0, omegal + omega + 5], row=3, col=2)
    fig.update_yaxes(title_text='Amplitude [arb.]', row=3, col=2)

    fig.update_layout(height=900, width=900, title_text="IQ Mixer Frequency Domain Signals (Separate I and Q)")

    ### Time domain plots for Output I and Q signals (can be a second figure or below) ###
    st.write("### Output Time Domain Signals - I and Q")

    fig_time = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("Output I(t)", "Output Q(t)"),
        horizontal_spacing=0.2
    )
    fig_time.add_trace(go.Scatter(x=t, y=Iout, mode='lines', line=dict(color='blue')), row=1, col=1)
    fig_time.add_trace(go.Scatter(x=t, y=Qout, mode='lines', line=dict(color='green')), row=1, col=2)

    fig_time.update_xaxes(title_text='Time [ns]', range=[0, 2], row=1, col=1)
    fig_time.update_yaxes(title_text='Amplitude [arb.]', row=1, col=1)
    fig_time.update_xaxes(title_text='Time [ns]', range=[0, 2], row=1, col=2)
    fig_time.update_yaxes(title_text='Amplitude [arb.]', row=1, col=2)

    fig_time.update_layout(height=400, width=900)

    # Display the plots
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_time, use_container_width=True)

if __name__ == "__main__":
    main()