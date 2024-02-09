'''
Authors:
    M.F. Fitzpatrick

Release Date: 
    V 1.0: 12/16/2023

'''
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from quantum_simulator import run_quantum_simulation, run_frequency_sweep  
from bokeh.plotting import figure
from bokeh.models import FreehandDrawTool
import matplotlib.pyplot as plt
import plotly.express as px

def add_gaussian(pulse_vector, amplitude, sigma, center, n_steps, t_final):
    """
    Adds a Gaussian pulse to the pulse vector.
    """
    tlist = np.linspace(0, t_final, n_steps)
    gaussian = amplitude * np.exp(-((tlist - center) ** 2) / (2 * sigma ** 2))
    return np.clip(pulse_vector + gaussian, -1, 1)  # Ensuring values are within [-1, 1]

def add_square(pulse_vector, amplitude, start, stop, n_steps, t_final):
    """
    Adds a Square pulse to the pulse vector.
    """
    tlist = np.linspace(0, t_final, n_steps)
    square_pulse = np.where((tlist >= start) & (tlist <= stop), amplitude, 0)

    return np.clip(pulse_vector + square_pulse, -1, 1)


def main():

    # if running locally, run source fitzlab/cassini-fitzlab/venv_st/bin/activate
    st.title('Qu-Blitz Virtual Qubit Lab')
    fitzlab_logo = Image.open("images/fitz_lab_logo.png")

    st.sidebar.image(fitzlab_logo, use_column_width=True)

    st.header('This app simulates the dynamics of a driven qubit (two-level system)')
    st.header('Simulation Parameters')

    # Additional UI for frequency sweep
    sim_mode = st.selectbox("Select Simulation Mode", ["Single Frequency","Frequency Sweep"])
    omega_q = st.number_input(r'$\omega_q$ [GHz]', 0.00, value=5.00, step=0.01, key='qubit_freq',format="%.2f") # need to address this later
    
    if sim_mode == "Frequency Sweep":
        start_freq = st.number_input("Start Frequency [GHz]", value=4.8, step=0.1)
        stop_freq = st.number_input("Stop Frequency [GHz]", value=5.2, step=0.1)
        num_points = st.number_input("Number of Points", value=20, step=1)
        t_final = 10  # Ensure this is defined
        n_steps = 100*int(t_final)
        omega_rabi = st.number_input('Rabi Rate $\Omega_0=\Omega_{0,x}=\Omega_{0,y}$ [MHz]', 0.0, value=50.0, step=1.0, key='rabi')
        T1 = st.number_input(r'$T_1$ [ns]', 0, value=100, step=10, key='T1_input')
        T2 = st.number_input(r'$T_2$ [ns]', 0, value=200, step=10, key='T2_input')
        num_shots = st.number_input('shots', 1, value=256, step=1)
        constant_I_amp = 1.


        if st.button('Run Frequency Sweep'):
            results = run_frequency_sweep(start_freq, stop_freq, num_points, t_final, n_steps, omega_q, omega_rabi, T1, T2, num_shots)
    
            # Plotting heatmaps
            for sigma, data in results['expectation_values'].items():
                fig = go.Figure(data=go.Heatmap(
                    z=data, 
                    x=results['frequencies'], 
                    y=results['time_list'],
                    colorscale='Viridis'
                ))
                fig.update_layout(
                    title=f'Time-resolved ⟨{sigma}⟩ vs. Drive Frequency',
                    xaxis_title='Drive Frequency (GHz)',
                    yaxis_title='Time (ns)'
                )
                st.plotly_chart(fig)
            
            # Plotting the probability of being in state |1⟩
            fig_prob_1 = go.Figure(data=go.Heatmap(
                    z=results['prob_1_time_series'],
                    x=results['frequencies'],
                        y=results['time_list'],
                        colorscale='Viridis'
                    ))
            st.plotly_chart(fig_prob_1)
    else:
        # User inputs for simulation parameters
        omega_d = st.number_input(r'$\omega_d$ [GHz]', 0.00, value=6.00, step=0.01, key='drive_freq',format="%.2f") # need to address this later
        detuning = (omega_d - omega_q)*1e3
        t_final = int(st.number_input('t_final [ns]', 0, value=100, step=1, key='t_final'))
        T1 = st.number_input(r'$T_1$ [ns]', 0, value=100, step=10, key='T1_input')
        T2 = st.number_input(r'$T_2$ [ns]', 0, value=200, step=10, key='T2_input')
        # Enforce T2 <= 2*T1 constraint
        while T2 > 2 * T1:
            st.warning(r"T2 $\leq$ 2*T1")
            T2 = st.number_input(r'$T_2$ [$\mu$s]', 0.0, step=1.0)
        num_shots = st.number_input('shots', 1, value=256, step=1)
        # st.title('Qubit sPulse Simulator')
        st.header('Pulse Parameters')
        omega_rabi = st.number_input('Rabi Rate $\Omega_0=\Omega_{0,x}=\Omega_{0,y}$ [MHz]', 0.0, value=50.0, step=1.0, key='rabi')
        pulse_method = st.selectbox("Choose Pulse Input Method", ["Pre-defined Pulse", "Upload Pulses", "Draw Pulses"], key='pulse_input_type')
        # Input for detuning
        n_steps = 10 * t_final
        tlist = np.linspace(0, t_final, n_steps)
    # User inputs for simulation parameters
    omega_q = st.number_input(r'$\omega_q$ [GHz]', 0.000, value=6.000, step=0.001, key='qubit_freq',format="%.3f") # need to address this later
    omega_d = st.number_input(r'$\omega_d$ [GHz]', 0.000, value=6.000, step=0.001, key='drive_freq',format="%.3f") # need to address this later
    detuning = st.number_input(r'Plot $\Delta$ [MHz]', 0.000, value=0.0, step=1.0, key='detuning') # need to address this later
   
    t_final = int(st.number_input('t_final [ns]', 0, value=200, step=1, key='t_final'))
    T1 = st.number_input(r'$T_1$ [$\mu$s]', 0.0, value=5.0, step=1.0, key='T1_input')
    T2 = st.number_input(r'$T_2$ [$\mu$s]', 0.0, value=7.0, step=1.0, key='T2_input')
    # Enforce T2 <= 2*T1 constraint
    while T2 > 2 * T1:
        st.warning(r"T2 $\leq$ 2*T1")
        T2 = st.number_input(r'$T_2$ [$\mu$s]', 0.0, step=1.0)
    num_shots = st.number_input('shots', 1, value=256, step=1)
    # st.title('Qubit sPulse Simulator')
    st.header('Pulse Parameters')
    omega_rabi = st.number_input('Rabi Rate $\Omega$ [MHz]', 0.0, value=50.0, step=1.0, key='rabi')
    pulse_method = st.selectbox("Choose Pulse Input Method", ["Pre-defined Pulse", "Upload Pulses", "Draw Pulses"], key='pulse_input_type')
    # Input for detuning
    
    n_steps = 10 * t_final
    tlist = np.linspace(0, t_final, n_steps)

        # Initialize or retrieve sigma_x_vec and sigma_y_vec
        if 'sigma_x_vec' not in st.session_state:
            st.session_state.sigma_x_vec = 0*tlist
        if 'sigma_y_vec' not in st.session_state:
            st.session_state.sigma_y_vec = 0*tlist

        # check to see if you can execute both x and y drives.

        if pulse_method == "Pre-defined Pulse":
            pulse_type = st.selectbox("Choose Pulse Type", [ "Square", "Gaussian", "H", "X", "Y"], key='pulse_type')
            
    # check to see if you can execute both x and y drives.

    if pulse_method == "Pre-defined Pulse":
        pulse_type = st.selectbox("Choose Pulse Type", [ "Square", "Gaussian", "H", "X", "Y"], key='pulse_type')
        

            if pulse_type == 'Gaussian':
                target_channel = st.selectbox("Choose Target Channel", ["σ_x", "σ_y"], key='gaussian_target_channel')
                amp = st.number_input('Amplitude', -1.0, 1.0, 0.4, key='gaussian_amp')
                sigma = st.number_input('Sigma', 0.0, 100.0, 9.0, key='gaussian_sigma')
                center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
                if st.button('Add Gaussian Pulse', key='gaussian_button'):
                    pulse_vector = st.session_state.sigma_x_vec if target_channel == "σ_x" else st.session_state.sigma_y_vec
                    updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                    if target_channel == "σ_x":
                        st.session_state.sigma_x_vec = updated_pulse_vector
                    else:
                        st.session_state.sigma_y_vec = updated_pulse_vector
        if pulse_type == 'Gaussian':
            target_channel = st.selectbox("Choose Target Channel", ["σ_x", "σ_y"], key='gaussian_target_channel')
            amp = st.number_input('Amplitude', -1.0, 1.0, 0.4, key='gaussian_amp')
            sigma = st.number_input('Sigma', 0.0, 100.0, 9.0, key='gaussian_sigma')
            center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
            if st.button('Add Gaussian Pulse', key='gaussian_button'):
                pulse_vector = st.session_state.sigma_x_vec if target_channel == "σ_x" else st.session_state.sigma_y_vec
                updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                if target_channel == "σ_x":
                    st.session_state.sigma_x_vec = updated_pulse_vector
                else:
                    st.session_state.sigma_y_vec = updated_pulse_vector


            elif pulse_type == 'Square':
                target_channel = st.selectbox("Choose Target Channel", ["σ_x", "σ_y"], key='square_target_channel')
                amp = st.number_input('Amplitude', -1.0, 1.0, 1.0, key='square_amp')
                start = st.number_input('Start Time (ns)', min_value=0, max_value=t_final, value=0, step=1, key='square_start')
                
                stop = st.number_input('Stop Time (ns)', min_value=start, max_value=t_final, value=10, step=1, key='square_stop')
                
                while stop < start:
                    st.warning(r"stop time must be after start time")
                    stop = st.number_input('Stop Time (ns)', min_value=start, max_value=t_final, value=10, step=1, key='square_stop')

                # Enforce T2 <= 2*T1 constraint
                while stop < start:
                    st.warning(r"Cannot have stop before start!")
                    stop = st.number_input('Stop Time (ns)', min_value=start, max_value=t_final, value=10, step=1)
        elif pulse_type == 'Square':
            target_channel = st.selectbox("Choose Target Channel", ["σ_x", "σ_y"], key='square_target_channel')
            amp = st.number_input('Amplitude', -1.0, 1.0, 1.0, key='square_amp')
            start = st.number_input('Start Time (ns)', min_value=0, max_value=t_final, value=0, step=1, key='square_start')
            stop = st.number_input('Stop Time (ns)', min_value=start, max_value=t_final, value=10, step=1, key='square_stop')
            # Enforce T2 <= 2*T1 constraint
            while stop < start:
                st.warning(r"Cannot have stop before start!")
                stop = st.number_input('Stop Time (ns)', min_value=start, max_value=t_final, value=10, step=1)

                if st.button('Add Square Pulse', key='square_button'):
                    pulse_vector = st.session_state.sigma_x_vec if target_channel == "σ_x" else st.session_state.sigma_y_vec
                    updated_pulse_vector = add_square(pulse_vector, amp, start, stop, n_steps, t_final)
                    if target_channel == "σ_x":
                        st.session_state.sigma_x_vec = updated_pulse_vector
                    else:
                        st.session_state.sigma_y_vec = updated_pulse_vector


            elif pulse_type == 'H':
                
                amp = 0.2
                sigma = 9
                center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
                if st.button('Add H Gate', key='H_button'):
                    pulse_vector = st.session_state.sigma_x_vec
                    updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                    st.session_state.sigma_x_vec = updated_pulse_vector
                    
            elif pulse_type == 'X':
                
                amp = 0.4
                sigma = 9
                center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
                if st.button('Add X Gate', key='X_button'):
                    pulse_vector = st.session_state.sigma_x_vec
                    updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                    st.session_state.sigma_x_vec = updated_pulse_vector
        elif pulse_type == 'H':
            
            amp = 0.2
            sigma = 9
            center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
            if st.button('Add H Gate', key='H_button'):
                pulse_vector = st.session_state.sigma_x_vec
                updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                st.session_state.sigma_x_vec = updated_pulse_vector
                
        elif pulse_type == 'X':
            
            amp = 0.4
            sigma = 9
            center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
            if st.button('Add X Gate', key='X_button'):
                pulse_vector = st.session_state.sigma_x_vec
                updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                st.session_state.sigma_x_vec = updated_pulse_vector

            elif pulse_type == 'Y':
                
                amp = 0.4
                sigma = 9
                center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
                if st.button('Add Y Gate', key='Y_button'):
                    pulse_vector = st.session_state.sigma_y_vec
                    updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
                    st.session_state.sigma_y_vec = updated_pulse_vector
        elif pulse_type == 'Y':
            
            amp = 0.4
            sigma = 9
            center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
            if st.button('Add Y Gate', key='Y_button'):
                pulse_vector = st.session_state.sigma_y_vec
                updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
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
        fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_x_vec, mode='lines', name='$\Omega_x$'))
        fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_y_vec, mode='lines', name='$\Omega_y$'))
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
            params = (omega_q, omega_rabi*1e-3, t_final, n_steps, omega_d, st.session_state.sigma_x_vec, st.session_state.sigma_y_vec, num_shots, T1, T2)
            exp_values, probabilities, sampled_probabilities  = run_quantum_simulation(*params)
    # Button to run simulation
    if st.button('Run Simulation'):
        params = (omega_q, omega_rabi*1e-3, t_final, n_steps, omega_d, st.session_state.sigma_x_vec, st.session_state.sigma_y_vec, num_shots, T1*1e3, T2*1e3)
        exp_values, probabilities, sampled_probabilities  = run_quantum_simulation(*params)

            # Time array for transformation
            time_array = np.linspace(0, t_final, n_steps)  # Convert time to microseconds

            # Demodulate the expectation values
            exp_x_rotating = exp_values[0] * np.cos(2 * np.pi * (omega_d + detuning*1e-3) * time_array) + exp_values[1] * np.sin(2 * np.pi * (omega_d + detuning*1e-3) * time_array)
            exp_y_rotating = exp_values[1] * np.cos(2 * np.pi * (omega_d +detuning*1e-3) * time_array) - exp_values[0] * np.sin(2 * np.pi * (omega_d + detuning*1e-3) * time_array)
        # Demodulate the expectation values
        exp_x_rotating = exp_values[0] * np.cos(2 * np.pi * (omega_d + detuning*1e-3) * time_array) + exp_values[1] * np.sin(2 * np.pi * (omega_d + detuning*1e-3) * time_array)
        exp_y_rotating = exp_values[1] * np.cos(2 * np.pi * (omega_d +detuning*1e-3) * time_array) - exp_values[0] * np.sin(2 * np.pi * (omega_d + detuning*1e-3) * time_array)

            # Plot results in rotating frame
            fig_results_rotating = go.Figure()
            fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_x_rotating, mode='lines', name='⟨σ_x⟩'))
            fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_y_rotating, mode='lines', name='⟨σ_y⟩'))
            fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_values[2], mode='lines', name='⟨σ_z⟩'))
            fig_results_rotating.update_layout(
                xaxis_title='Time [ns]',
                yaxis_title='Expectation Values',
                title=f'Quantum Simulation Results in Rotating Frame of the drive frequency (ω_d={omega_d} GHz), with detuning (Δ={detuning} MHz)'
            )
            st.plotly_chart(fig_results_rotating)
        # Plot results in rotating frame
        fig_results_rotating = go.Figure()
        fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_x_rotating, mode='lines', name='⟨σ_x⟩'))
        fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_y_rotating, mode='lines', name='⟨σ_y⟩'))
        fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_values[2], mode='lines', name='⟨σ_z⟩'))
        fig_results_rotating.update_layout(
            xaxis_title='Time [ns]',
            yaxis_title='Expectation Values',
            title=f'Quantum Simulation Results in Rotating Frame (Δ={detuning} MHz)'
        )
        st.plotly_chart(fig_results_rotating)

            # Plot results in rotating frame
            fig_sampled_results = go.Figure()
            fig_sampled_results.add_trace(go.Scatter(x=tlist, y=sampled_probabilities, mode='lines'))
            fig_sampled_results.update_layout(
                xaxis_title='Time [ns]',
                yaxis_title='Measured Probability of |0⟩',
                title=f'Measurement Results (shots={num_shots})'
            )
            st.plotly_chart(fig_sampled_results)

            # Create a list of time values from 0 to t_final
            time_values = np.linspace(0, t_final, n_steps)

            # Creating a mesh for the unit sphere
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_sphere = np.cos(u) * np.sin(v)
            y_sphere = np.sin(u) * np.sin(v)
            z_sphere = np.cos(v)

            # Color map for the time evolution
            colors = time_values

            fig_bloch = go.Figure(data=[
                go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3),
                go.Scatter3d(
                    x=exp_x_rotating,  # ⟨σ_x⟩ values
                    y=exp_y_rotating,  # ⟨σ_y⟩ values
                    z=exp_values[2],  # ⟨σ_z⟩ values
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=colors,
                        opacity=0.8,
                        colorscale='inferno',
                        colorbar=dict(title='Time', tickvals=[0, t_final], ticktext=[0, t_final])
                    )  # Use time-based colors
                )
            ])

            fig_bloch.update_layout(
                title='State Vector on the Bloch Sphere',
                scene=dict(
                    xaxis_title='⟨σ_x⟩',
                    yaxis_title='⟨σ_y⟩',
                    zaxis_title='⟨σ_z⟩',
                    xaxis=dict(range=[-1, 1]),
                    yaxis=dict(range=[-1, 1]),
                    zaxis=dict(range=[-1, 1]),
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )

            st.plotly_chart(fig_bloch)


if __name__ == "__main__":
    main()
