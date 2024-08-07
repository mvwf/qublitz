######################
###### IQ Mixer ######
# Author: Max Weiner #
### Date: 8/5/2024 ###
######################
# Draft 1

from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmam, Options
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
from matplotlib.colors import Normalize
from qutip import *
from scipy.fft import fftshift,fft,ifft
from matplotlib.widgets import Slider 
import plotly.graph_objects as go
import streamlit as st
# Updates everytime the slider values change

# initialize sliders
def main():
    st.title("IQ Mixing Demo")

    A = st.slider('Input Amplitude', 0.2, 20.0, 1.0, 0.1)
    phi = st.slider('Input Phase (°)', 0.0, 180.0, 1.0, 0.1)
    omega = st.slider('Input Frequency (GHz)', 1.0, 20.0, 1.0, 0.1)

    Al = st.slider('LO Amplitude', 0.2, 20.0,1.0,0.1)
    phil = st.slider('LO Phase (°)', 0.0, 180.0, 1.0,0.1)
    omegal = st.slider('LO Frequency (GHz)', 1.0, 20.0,1.0,0.1)

    nsteps = 1000
    t = np.linspace(0,20*1/min(omegal,omega),nsteps)
    d = max(t)/nsteps
    f = np.linspace(-1/(2*d),1/(2*d),nsteps)

    ### Calculate Q from Mag, Phase, Frequency ###

    Q = A*np.sin(2*np.pi*omega*t + phi)
    I = A*np.cos(2*np.pi*omega*t + phi)
    fftinQ = np.abs(fftshift(fft(Q)))
    fftinI = np.abs(fftshift(fft(I)))

    Ql = Al*np.sin(2*np.pi*omegal*t + phil)
    Il = Al*np.cos(2*np.pi*omegal*t + phil)
    fftlocQ = np.abs(fftshift(fft(Ql)))
    fftlocI = np.abs(fftshift(fft(Il)))

    #################### Compute Input and Output of IQ Mixer ##################

    Qout=Q*Ql
    fftQout=np.abs(fftshift(fft(Qout)))
    Iout=I*Il
    fftIout=np.abs(fftshift(fft(Iout)))

    ############################## Plots #######################################

    # Input Frequency
    fig_infft = go.Figure(data=[
        go.Scatter(x=f, y=fftinI, mode = 'lines', name = "I",marker = dict(color = 'black')),
            go.Scatter(
                x=f,  # ⟨σ_x⟩ values
                y=fftinQ,
                mode = 'lines',
                name = "Q",  # ⟨σ_y⟩ values
                marker = dict(color = 'red')
            )
        ])
    fig_infft.update_xaxes(range=[0, omega+5])
    fig_infft.update_layout(title='Input FFT',xaxis_title='Frequency [GHz]',yaxis_title='Amplitude')
    st.plotly_chart(fig_infft)
     
    ### LO FFT ###
    fig_locfft = go.Figure(data=[
    go.Scatter(x=f, y=fftlocI, mode = 'lines', name = "I", marker = dict(color = 'black')),
    go.Scatter(x=f, y=fftlocQ, mode = 'lines', name = "Q",marker = dict(color = 'red'))
    ])
    fig_locfft.update_xaxes(range=[0, omegal+5])
    fig_locfft.update_layout(title='Local Oscillator FFT',xaxis_title='Frequency [GHz]',yaxis_title='Amplitude')
    st.plotly_chart(fig_locfft)
    ##############

    ### Time-Domain Signal ###
    fig_Iout = go.Figure(data=[
    go.Scatter(x=t, y=Iout, mode = 'lines', marker = dict(color = 'blue'))
    ])
    fig_Iout.update_layout(title='Output I',xaxis_title='Time [ns]',yaxis_title='Amplitude')
    st.plotly_chart(fig_Iout)

    fig_Qout = go.Figure(data=[
    go.Scatter(x=t, y=Qout, mode = 'lines', marker = dict(color = 'blue'))
    ])
    fig_Qout.update_layout(title='Output Q',xaxis_title='Time [ns]',yaxis_title='Amplitude')
    st.plotly_chart(fig_Qout)
    ##############

    ## Output FFT ##
    fig_ffti = go.Figure(data=[
    go.Scatter(x=f, y=fftIout, mode = 'lines', marker = dict(color = 'black'))
    ])

    fig_ffti.update_xaxes(range=[0, omegal+omega+5])
    fig_ffti.update_layout(title='Output FFT I',xaxis_title='Frequency [GHz]',yaxis_title='Amplitude')
    st.plotly_chart(fig_ffti)

    fig_fftq = go.Figure(data=[
    go.Scatter(x=f, y=fftQout, mode = 'lines', marker = dict(color = 'black'))
    ])
    fig_fftq.update_xaxes(range=[0, omegal+omega+5])
    fig_fftq.update_layout(title='Output FFT Q',xaxis_title='Frequency [GHz]',yaxis_title='Amplitude')
    st.plotly_chart(fig_fftq)

    ########################

if __name__ == "__main__":
    main()