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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import base64
from io import BytesIO
from plotly.subplots import make_subplots
import json
import os
from ratelimit import limits, sleep_and_retry
from datetime import timedelta

# functions

# function: converts data to CSV (for download)
@sleep_and_retry # rate limiting decorator
@limits(calls=1, period=timedelta(seconds=60).total_seconds()) # rate limit of 1 call per minute

def to_csv(index, data):
    df = pd.DataFrame(data, index=index) # Dataframe from data
    return df.to_csv().encode('utf-8') 

# function: adds a gaussian pulse to the pulse vector
@sleep_and_retry 
@limits(calls=1, period=timedelta(seconds=60).total_seconds())
def add_gaussian(pulse_vector, amplitude, sigma, center, n_steps, t_final):
    """
    Adds a Gaussian pulse to the pulse vector.
    """
    tlist = np.linspace(0, t_final, n_steps) 
    gaussian = amplitude * np.exp(-((tlist - center) ** 2) / (2 * sigma ** 2))
    return np.clip(pulse_vector + gaussian, -1, 1)  # Ensuring values are within [-1, 1]

# function: adds a square pulse to the pulse vector
@sleep_and_retry
@limits(calls=1, period=timedelta(seconds=60).total_seconds())
@st.cache_data
def add_square(pulse_vector, amplitude, start, stop, n_steps, t_final):
    """
    Adds a Square pulse to the pulse vector.
    """
    tlist = np.linspace(0, t_final, n_steps)
    square_pulse = np.where((tlist >= start) & (tlist <= stop), amplitude, 0)
    # Extend pulse_vector to match the length of square_pulse
    if len(pulse_vector) < len(square_pulse):
        pulse_vector = np.pad(pulse_vector, (0, len(square_pulse) - len(pulse_vector)), 'constant')
    return np.clip(pulse_vector + square_pulse, -1, 1)

# main function
@sleep_and_retry
@limits(calls=1, period=timedelta(seconds=60).total_seconds())
@st.cache_data(experimental_allow_widgets=True)
def main():

    # if running locally, run source fitzlab/cassini-fitzlab/venv_st/bin/activate
    st.title('Qublitz Virtual Qubit Lab') # site title
    logo = Image.open("images/logo.png") 
    st.sidebar.image(logo, use_column_width=True) # display logo on the side 

    st.header('This app simulates the dynamics of a driven qubit (two-level system)')
    st.subheader(r'$\hat{H/\hbar} = \frac{\omega_q}{2}\hat{\sigma}_z + \frac{\Omega(t)}{2}\hat{\sigma}_x\cos(\omega_d t) + \frac{\Omega(t)}{2}\hat{\sigma}_y\cos(\omega_d t)$') # Hamiltonian 
    st.latex(r'''\text{Where } |1\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \text{ and } |0\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}''') # basis vectors

    st.header('Simulation Parameters')
    

    # add a dropdown to select "Free Play" or "Custom Qubit Query"
    # if "Free Play" is selected, the user can input their own parameters
    # if "Custom Qubit Query" is selected, the user can select a user ID from a list of pre-defined parameters
    

    user_selection = st.selectbox("Select User Mode", ["Free Play", "Custom Qubit Query"], key='user_selection')
    
    if user_selection == "Custom Qubit Query":
        # Convert st.secrets to a dictionary to easily list all user IDs
        secrets_dict = dict(st.secrets)
        user_ids = list(secrets_dict.keys())

        if user_ids:
            # Select a user ID from the list
            user_id = st.selectbox("Select User ID", user_ids)
            
            # Access the parameters directly via st.secrets since each user ID is at the top level
            user_parameters = st.secrets[user_id]
            omega_q = user_parameters["omega_q"]
            omega_rabi = user_parameters["Rabi_rate"]
            T1 = user_parameters["T1"]
            T2 = 2 * T1
        else:
            st.warning("No user parameters found. Please configure the user parameters first.")
            # Instructions to configure them or stop execution
            st.stop()

    if user_selection == "Free Play":
        omega_q = st.number_input(r'$\omega_q/2\pi$ [GHz]', 0.000, value=5.000, step=0.001, key='qubit_freq',format="%.3f") # need to address this later
    
    sim_mode = st.selectbox("Select Simulation Mode", ["Time Domain", "Frequency Domain"], key='sim_mode')

    if sim_mode == "Frequency Domain":
        start_freq = st.number_input(r"Start $\omega_d/2\pi$ [GHz]", value=4.8, step=0.1, key='start_freq_frequency_domain')
        stop_freq = st.number_input(r"Stop $\omega_d/2\pi$ [GHz]", value=5.2, step=0.1, key='stop_freq_frequency_domain')
        num_points = st.number_input("Number of Frequencies", value=11, min_value=1, max_value=81,  step=1, key='num_points_frequency_domain')
        
        # t_final = int(st.number_input(r'Duration $\Delta t$ [ns]', 0, value=25, step=1, key='t_final'))
        t_final = 25
        n_steps = 25*int(t_final)

        if user_selection == "Free Play":
            omega_rabi = st.number_input('Rabi Rate $\Omega_0/2\pi$ [MHz]', 0.0, value=100.0, step=1.0, key='rabi_frequency_domain')
            T1 = st.number_input(r'$T_1$ [ns]', 0, value=1000, step=10, key='T1_input_frequency_domain')
            T2 = st.number_input(r'$T_2$ [ns]', 0, value=2000, step=10, key='T2_input_frequency_domain')
            # Enforce T2 <= 2*T1 constraint
            while T2 > 2 * T1:
                st.warning(r"T2 $\leq$ 2*T1")
                T2 = st.number_input(r'$T_2$ [$\mu$s]', 0.0, step=1.0)

        num_shots = st.number_input('shots', 1, value=128, step=1, key='num_shots_frequency_domain')
        
        if st.button('Run Frequency Sweep'):
            results = run_frequency_sweep(start_freq, stop_freq, num_points, t_final, n_steps, omega_q, omega_rabi*1e-3, T1, T2, num_shots)
            prob_1_data = np.array(results['prob_1_time_series'])
            frequencies = results['frequencies']
            time_list = results['time_list']
            prob_1_data_transposed = prob_1_data.T

            # Calculate the maximum and average probabilities over time
            max_prob_1_over_time = np.max(prob_1_data_transposed, axis=0)
            avg_prob_1_over_time = np.mean(prob_1_data_transposed, axis=0)

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.02, 
                    subplot_titles=('Time-resolved Probability of State |1⟩', 'Max Probability of |1⟩', 'Avg Probability of |1⟩'),
                    row_heights=[0.6, 0.2, 0.2])

            # Heatmap for the probability of state |1⟩
            fig.add_trace(
                go.Heatmap(x=frequencies, y=time_list, z=prob_1_data_transposed, coloraxis="coloraxis"),
                row=1, col=1
            )

            # Line plot for maximum probability of state |1⟩ over time
            fig.add_trace(
                go.Scatter(x=frequencies, y=max_prob_1_over_time, mode='lines', showlegend=False),
                row=2, col=1
            )

            # Line plot for average probability of state |1⟩ over time
            # don't add to legend
            fig.add_trace(
                go.Scatter(x=frequencies, y=avg_prob_1_over_time, mode='lines', showlegend=False),
                row=3, col=1
            )

            # Update layout
            fig.update_layout(height=800, width=600, title_text="Frequency Domain Simulation Results",
                            coloraxis=dict(colorscale='Viridis'),  # Adjust colorscale as needed
                            xaxis_title="Frequency [GHz]",
                            yaxis_title="Time [ns]",
                            xaxis3_title="Frequency [GHz]",
                            yaxis2_title="Max Prob of |1⟩",
                            yaxis3_title="Avg Prob of |1⟩")

            # Adjust axes properties if needed
            fig.update_xaxes(title_text="Drive Frequency [GHz]", row=3, col=1)
            fig.update_yaxes(title_text="Time [ns]", row=1, col=1)

            # Display the figure in Streamlit
            st.plotly_chart(fig)

            if 'avg_prob_1_over_time' in locals():
                # Convert simulation results to CSV
                csv = to_csv(range(len(frequencies)), data = {'frequencies [GHz]':frequencies, 
                    'avg_prob_1_data': avg_prob_1_over_time, 
                    'max_prob_1_data': max_prob_1_over_time
                })
                
                # Generate download button for the CSV file
                st.download_button(
                    label="Download Simulation Data as CSV",
                    data=csv,
                    file_name="simulation_data.csv",
                    mime="text/csv",
                )

    else: # Time Domain
        # User inputs for simulation parameters

        omega_d = st.number_input(r'$\omega_d/2\pi$ [GHz]', 0.000, value=5.000, step=0.001, key='drive_freq',format="%.3f") # need to address this later
        detuning = (omega_d - omega_q)*1e3
        t_final = int(st.number_input(r'Duration $\Delta t$ [ns]', value=200.0, min_value=0.0, max_value=1000.0, step=1.0, key='t_final_time_domain'))
        n_steps = 20*int(t_final)
        
        if user_selection == "Free Play":
            omega_rabi = st.number_input('Rabi Rate $\Omega_0/2\pi$ [MHz]', 0.0, value=50.0, step=1.0, key='rabi_time_domain')
            T1 = st.number_input(r'$T_1$ [ns]', 0, value=100, step=10, key='T1_input_time_domain')
            T2 = st.number_input(r'$T_2$ [ns]', 0, value=200, step=10, key='T2_input_time_domain')
            # Enforce T2 <= 2*T1 constraint
            while T2 > 2 * T1:
                st.warning(r"T2 $\leq$ 2*T1")
                T2 = st.number_input(r'$T_2$ [$\mu$s]', 0.0, step=1.0)
        
        num_shots = st.number_input('shots', value=256,  min_value=1, max_value=4096,  step=1, key='num_shots_time_domain')
    
        st.header('Pulse Parameters')
        target_channel = st.selectbox('Choose Target Channel', ['σ_x', 'σ_y'], key='square_target_channel') # user selects target channel
        n_steps = 25*int(t_final) # set the number of steps based on user inputs times 25
        tlist = np.linspace(0, t_final, n_steps) # set time domain list

        if 'sigma_x_vec' not in st.session_state: # if no x vector has been initialized yet
            st.session_state.sigma_x_vec = tlist*0 # set the sigma x vector to 0 with the dimensions of the time domain
        if 'sigma_y_vec' not in st.session_state: # if no y vector has been initialized yet
            st.session_state.sigma_y_vec = tlist*0
        
        amp = st.slider('Amplitude', -1.0, 1.0, 1.0, key='square_amp') # user selects amplitude
        start = st.slider('Start Time (ns)', 0.0, float(t_final-1.0), step=1.0, key='square_start')
        stop = st.slider('Stop Time (ns)', min_value=start, max_value=float(t_final), step=1.0, key='square_stop')

        if st.button('Add Square Pulse', key='square_button'):
            if target_channel == 'σ_x': # if the target channel is sigma x
                pulse_vector = st.session_state.sigma_x_vec # set the pulse vector to the sigma x vector
                updated_pulse_vector = add_square(pulse_vector, amp, start, stop, n_steps, t_final) # add a square pulse to the sigma x vector
                st.session_state.sigma_x_vec = updated_pulse_vector # update the sigma x vector
                st.session_state.sigma_y_vec = np.pad(st.session_state.sigma_y_vec, (0, len(updated_pulse_vector) - len(st.session_state.sigma_y_vec)), 'constant', constant_values=st.session_state.sigma_y_vec[-1]) # pad the sigma y vector with the last value of the sigma y vector
            else: # for target channel sigma y
                pulse_vector = st.session_state.sigma_y_vec
                updated_pulse_vector = add_square(pulse_vector, amp, start, stop, n_steps, t_final) # add a square pulse to the sigma y vector
                st.session_state.sigma_y_vec = updated_pulse_vector
                st.session_state.sigma_x_vec = np.pad(st.session_state.sigma_x_vec, (0, len(updated_pulse_vector) - len(st.session_state.sigma_x_vec)), 'constant', constant_values=st.session_state.sigma_x_vec[-1])

        # Clear pulses button
        if st.button('Clear Pulse', key='clear_button'):
            st.session_state.sigma_x_vec = tlist*0
            st.session_state.sigma_y_vec = tlist*0
            pulse_vector = None
            updated_pulse_vector = None
            

        # Create Plotly figures for σx and σy
        # Adjust Plotly figures to show the entire trace
        fig_sigma = go.Figure()
        plot_lw = 3 # plot linewidth
        fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_x_vec, mode='lines', name='Ω_x(t)',line=dict(width=plot_lw)))
        fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_y_vec, mode='lines', name=rf'Ω_y(t)',line=dict(width=plot_lw)))
        fig_sigma.update_layout(
            xaxis_title='Time [ns]',
            yaxis_title='Amplitude',
            title=r'Time-Dependent Amplitudes of Ω_0 (Ω_x(t) and Ω_x(t))',
            xaxis=dict(range=[0, t_final]),  # Adjust the x-axis to show the entire trace
            yaxis=dict(range=[-1.05, 1.05])         # Adjust the y-axis range if necessary
        )
        st.plotly_chart(fig_sigma)

        # Button to run simulation
        if st.button('Run Simulation'):
            
            params = (omega_q, omega_rabi*1e-3, t_final, n_steps, omega_d, st.session_state.sigma_x_vec, st.session_state.sigma_y_vec, num_shots, T1, T2)

            try:
                exp_values, __, sampled_probabilities = run_quantum_simulation(*params)
            except Exception as e:
                st.warning(f"An error occurred. Please refresh the page and try again: {e}")

            # Time array for transformation
            time_array = np.linspace(0, t_final, n_steps)  # Convert time to microseconds

            # Demodulate the expectation values
            exp_y_rotating = -(exp_values[0] * np.cos(2 * np.pi * omega_d * time_array) + exp_values[1] * np.sin(2 * np.pi * omega_d * time_array))
            exp_x_rotating = exp_values[0] * np.sin(2 * np.pi * omega_d * time_array) - exp_values[1] * np.cos(2 * np.pi * omega_d * time_array)
            
            # Plot results in rotating frame
            fig_results_rotating = go.Figure()
            fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_x_rotating, mode='lines', name=r'⟨σ_x⟩',line=dict(width=plot_lw)))
            fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_y_rotating, mode='lines', name=r'⟨σ_y⟩',line=dict(width=plot_lw)))
            fig_results_rotating.add_trace(go.Scatter(x=tlist, y=exp_values[2], mode='lines', name=r'⟨σ_z⟩',line=dict(width=plot_lw)))
            fig_results_rotating.update_layout(
                yaxis=dict(range=[-1.05, 1.05]),
                xaxis_title='Time [ns]',
                yaxis_title='Expectation Values',
                title=f'Quantum Simulation Results in Rotating Frame of the drive frequency (ω_d={omega_d} GHz), with detuning (Δ={detuning} MHz)'
            )
            st.plotly_chart(fig_results_rotating)

            # Plot results in rotating frame
            fig_sampled_results = go.Figure()
            fig_sampled_results.add_trace(go.Scatter(x=tlist, y=sampled_probabilities, mode='lines',line=dict(width=plot_lw)))
            fig_sampled_results.update_layout(
                xaxis_title='Time [ns]',
                yaxis=dict(range=[-0.05, 1.05]),
                yaxis_title='Measured Probability of |1⟩',
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
                        size=6,
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
                    xaxis_title=dict(text="σ_x"),
                    yaxis_title=dict(text="σ_y"),
                    zaxis_title=r'σ_z',
                    xaxis=dict(range=[-1, 1]),
                    yaxis=dict(range=[-1, 1]),
                    zaxis=dict(range=[-1, 1]),
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            
            fig_bloch.add_trace(go.Scatter3d(
                x=[0, 0], 
                y=[0, 0], 
                z=[1.0, -1.0], 
                mode='text',
                text=["|1⟩", "|0⟩"],
                textposition=["top center","bottom center"],
                textfont=dict(
                    color=["white", "white"],
                    size=20
                )
            ))

            # Display the plot
            st.plotly_chart(fig_bloch)
    # Assuming 'exp_values', 'time_values', 'sampled_probabilities' are your data variables
    if 'exp_values' in locals():
        # Convert simulation results to CSV
        csv = to_csv(range(len(time_values)), data = {
            'Time [ns]': time_values,
            'Exp X': exp_x_rotating, 
            'Exp Y': exp_y_rotating, 
            'Exp Z': exp_values[2], 
            'Sampled Probabilities': sampled_probabilities})
        
        # Generate download button for the CSV file
        st.download_button(
            label="Download Simulation Data as CSV",
            data=csv,
            file_name="simulation_data.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
