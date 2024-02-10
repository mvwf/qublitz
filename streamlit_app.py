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

# Function to convert data to CSV (for download)
def to_csv(index, data):
    df = pd.DataFrame(data, index=index)
    return df.to_csv().encode('utf-8')


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
    st.title('Qublitz Virtual Qubit Lab')
    logo = Image.open("images/logo.png")
    st.sidebar.image(logo, use_column_width=True)

    st.header('This app simulates the dynamics of a driven qubit (two-level system)')
    st.subheader(r'$\hat{H/\hbar} = \frac{\omega_q}{2}\hat{\sigma}_z + \frac{\Omega(t)}{2}\hat{\sigma}_x\cos(\omega_d t) + \frac{\Omega(t)}{2}\hat{\sigma}_y\cos(\omega_d t)$')
    st.latex(r'''\text{Where } |0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \text{ and } |1\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}''')

    st.header('Simulation Parameters')
    # make a subheader for the Hamiltonian
    
    # Additional UI for frequency sweep
    sim_mode = st.selectbox("Select Simulation Mode", ["Time Domain", "Frequency Domain"], key='sim_mode')
    omega_q = st.number_input(r'$\omega_q/2\pi$ [GHz]', 0.000, value=5.000, step=0.001, key='qubit_freq',format="%.3f") # need to address this later
    
    if sim_mode == "Frequency Domain":
        start_freq = st.number_input(r"Start $\omega_d/2\pi$ [GHz]", value=4.8, step=0.1, key='start_freq_frequency_domain')
        stop_freq = st.number_input(r"Stop $\omega_d/2\pi$ [GHz]", value=5.2, step=0.1, key='stop_freq_frequency_domain')
        num_points = st.number_input("Number of Frequencies", value=11, step=1, key='num_points_frequency_domain')
        t_final = int(st.number_input(r'Duration $\Delta t$ [ns]', 0, value=50, step=1, key='t_final'))
        n_steps = 20*int(t_final)
        omega_rabi = st.number_input('Rabi Rate $\Omega_0/2\pi$ [MHz]', 0.0, value=100.0, step=1.0, key='rabi_frequency_domain')
        T1 = st.number_input(r'$T_1$ [ns]', 0, value=1000, step=10, key='T1_input_frequency_domain')
        T2 = st.number_input(r'$T_2$ [ns]', 0, value=2000, step=10, key='T2_input_frequency_domain')
        # Enforce T2 <= 2*T1 constraint
        while T2 > 2 * T1:
            st.warning(r"T2 $\leq$ 2*T1")
            T2 = st.number_input(r'$T_2$ [$\mu$s]', 0.0, step=1.0)
        num_shots = st.number_input('shots', 1, value=256, step=1, key='num_shots_frequency_domain')
        
        if st.button('Run Frequency Sweep'):
            results = run_frequency_sweep(start_freq, stop_freq, num_points, t_final, n_steps, omega_q, omega_rabi*1e-3, T1, T2, num_shots)
            prob_1_data = np.array(results['prob_1_time_series'])
            frequencies = results['frequencies']
            time_list = results['time_list']
            prob_1_data_transposed = prob_1_data.T

            # Calculate the maximum and average probabilities over time
            max_prob_1_over_time = np.max(prob_1_data_transposed, axis=0)
            avg_prob_1_over_time = np.mean(prob_1_data_transposed, axis=0)

            # Create a figure with two subplots, sharing the x-axis
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

            # Heatmap
            im = axes[0].imshow(
                prob_1_data_transposed,
                aspect='auto',
                extent=[min(frequencies), max(frequencies), min(time_list), max(time_list)],
                origin='lower',
                cmap='viridis'
            )
            axes[0].set_ylabel('Time (ns)', fontsize=14)
            axes[0].set_title('Time-resolved Probability of State |1⟩ vs. Drive Frequency', fontsize=16)

            # Colorbar
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Prob of State |1⟩', size=14)

            # Max probabilities plot
            axes[1].plot(frequencies, max_prob_1_over_time, color='blue')
            axes[1].set_ylabel('Max Prob of |1⟩', fontsize=14)
            axes[1].grid(True)

            # Average probabilities plot
            axes[2].plot(frequencies, avg_prob_1_over_time, color='green')
            axes[2].set_ylabel('Avg Prob of |1⟩', fontsize=14)
            axes[2].set_xlabel('Drive Frequency [GHz]', fontsize=14)
            axes[2].grid(True)

            # Ensure the x-axes are aligned
            for ax in axes[1:]:
                ax.set_xlim(axes[0].get_xlim())

            # Set tick parameters for all axes
            for ax in axes:
                ax.tick_params(labelsize=12)

            # Adjust layout for tight fit and to show x-axis labels at the bottom
            plt.tight_layout()

            # Display the plots in Streamlit
            st.pyplot(fig)

            if 'avg_prob_1_over_time' in locals():
                # Convert simulation results to CSV
                csv = to_csv(range(len(time_list)), data = {'frequencies [GHz]':frequencies, 
                    'prob_1_data': avg_prob_1_over_time, 
                    'times [ns]': time_list, 
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
        t_final = int(st.number_input(r'Duration $\Delta t$ [ns]', 0, value=100, step=1, key='t_final_time_domain'))
        n_steps = 20*int(t_final)
                # Input for detuning
        omega_rabi = st.number_input('Rabi Rate $\Omega_0/2\pi$ [MHz]', 0.0, value=50.0, step=1.0, key='rabi_time_domain')
        T1 = st.number_input(r'$T_1$ [ns]', 0, value=100, step=10, key='T1_input_time_domain')
        T2 = st.number_input(r'$T_2$ [ns]', 0, value=200, step=10, key='T2_input_time_domain')
        # Enforce T2 <= 2*T1 constraint
        while T2 > 2 * T1:
            st.warning(r"T2 $\leq$ 2*T1")
            T2 = st.number_input(r'$T_2$ [$\mu$s]', 0.0, step=1.0)
        num_shots = st.number_input('shots', 1, value=256, step=1, key='num_shots_time_domain')
        
        # st.title('Qubit sPulse Simulator')
        st.header('Pulse Parameters')
        # pulse_method = st.selectbox("Choose Pulse Input Method", ["Pre-defined Pulse", "Upload Pulses", "Draw Pulses"], key='pulse_input_type')

        n_steps = 10 * t_final
        tlist = np.linspace(0, t_final, n_steps)
        plot_lw = 3
        # Initialize or retrieve sigma_x_vec and sigma_y_vec
        if 'sigma_x_vec' not in st.session_state:
            st.session_state.sigma_x_vec = 0*tlist
        if 'sigma_y_vec' not in st.session_state:
            st.session_state.sigma_y_vec = 0*tlist

        # pulse_type = st.selectbox("Choose Pulse Type", [ "Square", "Gaussian"], key='pulse_type')
            
        # if pulse_type == 'Gaussian':
        #     target_channel = st.selectbox("Choose Target Channel", ["σ_x", "σ_y"], key='gaussian_target_channel')
        #     amp = st.number_input('Amplitude', -1.0, 1.0, 0.4, key='gaussian_amp')
        #     sigma = st.number_input('Sigma', 0.0, 100.0, 9.0, key='gaussian_sigma')
        #     center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
        #     if st.button('Add Gaussian Pulse', key='gaussian_button'):
        #         pulse_vector = st.session_state.sigma_x_vec if target_channel == "σ_x" else st.session_state.sigma_y_vec
        #         updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
        #         if target_channel == "σ_x":
        #             st.session_state.sigma_x_vec = updated_pulse_vector
        #         else:
        #             st.session_state.sigma_y_vec = updated_pulse_vector

        # elif pulse_type == 'Square':
            
        target_channel = st.selectbox("Choose Target Channel", ["σ_x", "σ_y"], key='square_target_channel')
        amp = st.number_input('Amplitude', -1.0, 1.0, 1.0, key='square_amp')
        start = st.number_input('Start Time (ns)', min_value=0, max_value=t_final, value=0, step=1, key='square_start')
        stop = st.number_input('Stop Time (ns)', min_value=start, max_value=t_final, value=100, step=10, key='square_stop')
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


            # elif pulse_type == 'H':
            #     amp = 0.2
            #     sigma = 9
            #     center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
            #     if st.button('Add H Gate', key='H_button'):
            #         pulse_vector = st.session_state.sigma_x_vec
            #         updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
            #         st.session_state.sigma_x_vec = updated_pulse_vector
                    
            # elif pulse_type == 'X':
                
            #     amp = 0.4
            #     sigma = 9
            #     center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
            #     if st.button('Add X Gate', key='X_button'):
            #         pulse_vector = st.session_state.sigma_x_vec
            #         updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
            #         st.session_state.sigma_x_vec = updated_pulse_vector

            # elif pulse_type == 'Y':
                
            #     amp = 0.4
            #     sigma = 9
            #     center = st.number_input('Center Position', 0, t_final, 50, key='gaussian_center')
            #     if st.button('Add Y Gate', key='Y_button'):
            #         pulse_vector = st.session_state.sigma_y_vec
            #         updated_pulse_vector = add_gaussian(pulse_vector, amp, sigma, center, n_steps, t_final)
            #         st.session_state.sigma_y_vec = updated_pulse_vector

            # elif pulse_method == "Upload Pulses":
            #     uploaded_file = st.file_uploader("Upload your pulse file", type=['csv', 'json'])
                
            # else:
            #     # # Create two Bokeh plots for σx and σy
            #     p_sigma_x = figure(x_range=(0, t_final), y_range=(-1, 1), width=400, height=400, title=r'σ_x')
            #     p_sigma_y = figure(x_range=(0, t_final), y_range=(-1, 1), width=400, height=400, title=r'σ_y')

            #     # Initialize the traces as flat lines
            #     xs_x = [[0, t_final]]
            #     ys_x = [[0, 0]]
            #     xs_y = [[0, t_final]]
            #     ys_y = [[0, 0]]

            #     # Create FreehandDrawTool for each plot
            #     renderer_x = p_sigma_x.multi_line(xs_x, ys_x, line_width=1, alpha=0.4, color='red')
            #     renderer_y = p_sigma_y.multi_line(xs_y, ys_y, line_width=1, alpha=0.4, color='blue')

            #     draw_tool_x = FreehandDrawTool(renderers=[renderer_x], num_objects=99999)
            #     draw_tool_y = FreehandDrawTool(renderers=[renderer_y], num_objects=99999)

            #     p_sigma_x.add_tools(draw_tool_x)
            #     p_sigma_x.toolbar.active_drag = draw_tool_x

            #     p_sigma_y.add_tools(draw_tool_y)
            #     p_sigma_y.toolbar.active_drag = draw_tool_y

            #     # Display both Bokeh plots
            #     col1, col2 = st.columns(2)
            #     col1.bokeh_chart(p_sigma_x)
            #     col2.bokeh_chart(p_sigma_y)

        # Clear Pulses Button
        if st.button('Clear Pulses'):
            st.session_state.sigma_x_vec = np.zeros(n_steps)
            st.session_state.sigma_y_vec = np.zeros(n_steps)

        # Create Plotly figures for σx and σy
        # Adjust Plotly figures to show the entire trace
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_x_vec, mode='lines', name='Ω_x(t)',line=dict(width=plot_lw)))
        fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_y_vec, mode='lines', name=rf'Ω_y(t)',line=dict(width=plot_lw)))
        fig_sigma.update_layout(
            xaxis_title='Time [ns]',
            yaxis_title='Amplitude',
            title=r'Time-Dependent Amplitudes of Ω_0 (Ω_x(t) and Ω_x(t))',
            xaxis=dict(range=[0, t_final]),  # Adjust the x-axis to show the entire trace
            yaxis=dict(range=[-1, 1])         # Adjust the y-axis range if necessary
        )
        st.plotly_chart(fig_sigma)

        # Button to run simulation
        if st.button('Run Simulation'):
            
            params = (omega_q, omega_rabi*1e-3, t_final, n_steps, omega_d, st.session_state.sigma_x_vec, st.session_state.sigma_y_vec, num_shots, T1, T2)
            exp_values, __, sampled_probabilities  = run_quantum_simulation(*params)

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
                text=["|0⟩", "|1⟩"],
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
