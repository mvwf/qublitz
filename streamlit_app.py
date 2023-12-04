import streamlit as st
import numpy as np
import plotly.express as px
from quantum_simulator import run_quantum_simulation, add_gaussian
import plotly.figure_factory as ff


def main():
    # ... Existing Streamlit code ...'

    # Define your simulation parameters
    omega_z = 5.0  # qubit frequency
    omega_d = 5.0  # drive frequency
    omega_rabi = 0.028 * 2  # rabi rate (same for X and Y)
    t_final = 120  # length of trace
    n_steps = 400  # number of steps

    num_shots = 5000  # number of shots
    control_noise_amp = 0.0  # noise added to control channels
    T1 = 50  # T1 time in ns
    T2 = 2 * T1  # T2 time in ns
    # T2 = 20
    assert T2 <= 2 * T1  # enforce physically relevant parameters

    user_vector_I = np.ones(n_steps)
    user_vector_Q = np.zeros(n_steps)

    simulation_results = run_quantum_simulation(omega_z, omega_rabi, t_final, n_steps, omega_d, user_vector_I, user_vector_Q, num_shots, T1, T2)

    # Create a time vector for plotting
    tlist = np.linspace(0, t_final, n_steps)

    # Create a Plotly figure for the simulation results
    fig_simulation = px.line(x=tlist, y=simulation_results, labels={"x": "Time", "y": "Probability"},
                             title="Quantum Simulation Results")

    # Display the generated plots
    st.plotly_chart(fig_simulation)


    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ['Group 1', 'Group 2', 'Group 3']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
            hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)
