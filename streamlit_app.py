'''
Authors:
    M.F. Fitzpatrick

Release Date: 
    V 1.0: 12/04/2023

'''
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from quantum_simulator import run_quantum_simulation
from bokeh.plotting import figure
from bokeh.models import FreehandDrawTool

def add_gaussian(pulse_vector, amplitude, sigma, center, n_steps, t_final):
    """
    Adds a Gaussian pulse to the pulse vector.
    """
    tlist = np.linspace(0, t_final, n_steps)
    gaussian = amplitude * np.exp(-((tlist - center) ** 2) / (2 * sigma ** 2))
    return np.clip(pulse_vector + gaussian, -1, 1)  # Ensuring values are within [-1, 1]

def add_square(pulse_vector, amplitude, start, stop, n_steps, t_final):
    tlist = np.linspace(0, t_final, n_steps)
    square_pulse = np.where((tlist >= start) & (tlist <= stop), amplitude, 0)
    updated_pulse_vector = np.clip(pulse_vector + square_pulse, -1, 1)
    return updated_pulse_vector


def main():

    st.title('Qubit Pulse Simulator: A Virtual Lab for ENGS 53')
    fitzlab_logo = Image.open("images/fitz_lab_logo.png")

    st.sidebar.image(fitzlab_logo, use_column_width=True)

    st.header('This app simulates the dynamics of a driven qubit (two-level system)')
    st.header('Simulation Parameters')
    omega_z = 1.0
    omega_d = 1.0 
    st.header('')
    # User inputs for simulation parameters
    omega_rabi = st.number_input('ω_rabi (Rabi rate)', 0.0, value=0.028 * 2, step=0.001)
    # t_final = st.number_input('t_final (length of trace in ns)', 0, value=200, step=1)
    t_final = int(st.number_input('t_final [ns]', 0, value=200, step=1))
    # num_shots = st.number_input('num_shots (number of shots)', 1, value=500, step=1)
    T1 = st.number_input(r'$T_1$ ($\mu$s)', 0.0, value= 50000.0, step=10.0)
    T2 = st.number_input(r'$T_2$ ($\mu$s)', 0.0, value=10000.0, step=10.0)
    
    num_shots = 5000 ## currently this not important.
    # st.title('Qubit sPulse Simulator')
    st.header('Pulse Parameters')
    pulse_method = st.selectbox("Choose Pulse Input Method", ["Pre-defined Pulse", "Upload Pulses", "Draw Pulses"], key='pulse_input_type')
    # Input for detuning
    detuning = st.number_input('Detuning (MHz)', value=0.0, step=0.1, key='detuning')

    n_steps = 4 * t_final
    tlist = np.linspace(0, t_final, n_steps)

    # Initialize or retrieve sigma_x_vec and sigma_y_vec
    if 'sigma_x_vec' not in st.session_state:
        st.session_state.sigma_x_vec = np.zeros(n_steps)
    if 'sigma_y_vec' not in st.session_state:
        st.session_state.sigma_y_vec = np.zeros(n_steps)

    if pulse_method == "Pre-defined Pulse":
        pulse_type = st.selectbox("Choose Pulse Type", ["Gaussian", "Square"], key='pulse_type')
        target_channel = st.selectbox("Choose Target Channel", ["sigma_x", "sigma_y"], key='target_channel')

        if pulse_type == 'Gaussian':
            amp = st.number_input('Amplitude', 0.0, 1.0, 0.4, key='gaussian_amp')
            sigma = st.number_input('Sigma', 1, 100, 9, key='gaussian_sigma')
            center = st.number_input('Center Position', 0, t_final, t_final // 2, key='gaussian_center')
            if st.button('Add Gaussian Pulse', key='gaussian_button'):
                pulse_vector = st.session_state.sigma_x_vec if target_channel == "sigma_x" else st.session_state.sigma_y_vec
                updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                if target_channel == "sigma_x":
                    st.session_state.sigma_x_vec = updated_pulse_vector
                else:
                    st.session_state.sigma_y_vec = updated_pulse_vector

        elif pulse_type == 'Square':
            amp = st.number_input('Amplitude', 0.0, 1.0, 0.5, key='square_amp')
            start = st.number_input('Start Time (ns)', min_value=0, max_value=t_final, value=10, step=1, key='square_start')
            stop = st.number_input('Stop Time (ns)', min_value=start, max_value=t_final, value=max(30, start), step=1, key='square_stop')
            if st.button('Add Square Pulse', key='square_button'):
                pulse_vector = st.session_state.sigma_x_vec if target_channel == "sigma_x" else st.session_state.sigma_y_vec
                updated_pulse_vector = add_square(pulse_vector, amp, start, stop, n_steps, t_final)
                if target_channel == "sigma_x":
                    st.session_state.sigma_x_vec = updated_pulse_vector
                else:
                    st.session_state.sigma_y_vec = updated_pulse_vector


    elif pulse_method == "Upload Pulses":
        uploaded_file = st.file_uploader("Upload your pulse file", type=['csv', 'json'])
        
    else:
         # # Create two Bokeh plots for σx and σy
        p_sigma_x = figure(x_range=(0, t_final), y_range=(-1, 1), width=400, height=400, title='σ_x')
        p_sigma_y = figure(x_range=(0, t_final), y_range=(-1, 1), width=400, height=400, title='σ_y')

        # Initialize the traces as flat lines
        xs_x = [[0, t_final]]
        ys_x = [[0, 0]]
        xs_y = [[0, t_final]]
        ys_y = [[0, 0]]

        # Create FreehandDrawTool for each plot
        renderer_x = p_sigma_x.multi_line(xs_x, ys_x, line_width=1, alpha=0.4, color='red')
        renderer_y = p_sigma_y.multi_line(xs_y, ys_y, line_width=1, alpha=0.4, color='blue')

        draw_tool_x = FreehandDrawTool(renderers=[renderer_x], num_objects=99999)
        draw_tool_y = FreehandDrawTool(renderers=[renderer_y], num_objects=99999)

        p_sigma_x.add_tools(draw_tool_x)
        p_sigma_x.toolbar.active_drag = draw_tool_x

        p_sigma_y.add_tools(draw_tool_y)
        p_sigma_y.toolbar.active_drag = draw_tool_y

        # Display both Bokeh plots
        col1, col2 = st.columns(2)
        col1.bokeh_chart(p_sigma_x)
        col2.bokeh_chart(p_sigma_y)
        # Clear Pulses Button
    if st.button('Clear Pulses'):
        st.session_state.sigma_x_vec = np.zeros(n_steps)
        st.session_state.sigma_y_vec = np.zeros(n_steps)

    # Create Plotly figures for σx and σy
    # Adjust Plotly figures to show the entire trace
    fig_sigma = go.Figure()
    fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_x_vec, mode='lines', name='⟨σ_x⟩'))
    fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_y_vec, mode='lines', name='⟨σ_y⟩'))
    fig_sigma.update_layout(
        xaxis_title='Time [ns]',
        yaxis_title='Amplitude',
        title='σ_x and σ_y Traces',
        xaxis=dict(range=[0, t_final]),  # Adjust the x-axis to show the entire trace
        yaxis=dict(range=[-1, 1])         # Adjust the y-axis range if necessary
    )
    st.plotly_chart(fig_sigma)
    # Button to update traces and run simulation


    # Button to run simulation
    if st.button('Run Simulation'):
        params = (omega_z, omega_rabi, t_final, n_steps, omega_d, st.session_state.sigma_x_vec, st.session_state.sigma_y_vec, num_shots, T1*1e3, T2*1e3)
        exp_values = run_quantum_simulation(*params)

        # Time array for transformation
        time_array = np.linspace(0, t_final, n_steps)  # Convert time to microseconds

        # Demodulate the expectation values
        exp_x_rotating = exp_values[0] * np.cos(2 * np.pi * (omega_d + detuning) * time_array) + exp_values[1] * np.sin(2 * np.pi * (omega_d + detuning) * time_array)
        exp_y_rotating = exp_values[1] * np.cos(2 * np.pi * (omega_d + detuning) * time_array) - exp_values[0] * np.sin(2 * np.pi * (omega_d + detuning) * time_array)

        # Plot results in rotating frame
        fig_results_rotating = go.Figure()
        fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_x_rotating, mode='lines', name='⟨σ_x⟩ Rotating Frame'))
        fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_y_rotating, mode='lines', name='⟨σ_y⟩ Rotating Frame'))
        fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_values[2], mode='lines', name='⟨σ_z⟩'))
        fig_results_rotating.update_layout(
            xaxis_title='Time',
            yaxis_title='Expectation Values in Rotating Frame',
            title='Quantum Simulation Results in Rotating Frame'
        )
        st.plotly_chart(fig_results_rotating)

if __name__ == "__main__":
    main()
