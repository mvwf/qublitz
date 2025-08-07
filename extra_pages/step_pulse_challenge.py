import streamlit as st 
import numpy as np 
import qutip as qt
import plotly.graph_objects as go
from quantum_simulator import run_quantum_simulation



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

def main():
    st.title('Step Pulse Challenge') 

    # Below is starter code for Unitary Hack

    # Bloch Sphere - Creating a meshgrid for the Bloch Sphere 
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)

    # Bloch Sphere - Color map for the time evolution
    t_final = 200 # NEEDS FIX to dynamically on the set start and stop times the user selects
    n_steps = 25*int(t_final) # set the number of steps based on user inputs times 25
    time_values = np.linspace(0, t_final, n_steps)
    colors = time_values

    fig_bloch = go.Figure(data=[
        go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3),
        go.Scatter3d(
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
    
    # Bloch Sphere - this is the default layout
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

    # Bloch Sphere - this sets the starting point of the Bloch Sphere
               
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

    st.header('Qubit Properties')
    # Setting parameters for the qubit, these do not need to be changed
    omega_q = 5
    omega_d = omega_q
    T1 = 100000000000 # Default relaxation time constant
    T2 = 200000000000 # Default dephasing time constant from 0 to 1000.0 ns
    detuning = 0

    #User input for qubit properties
    omega_rabi = st.slider('Rabi Rate: 'r'$\omega_0/2\pi$ (MHz)', 0.000, value=50.000, step=1.0, key='rabi_time_domain', format="%.3f") # User selects a Rabi rate from 0 to 50.000 MHz

    num_shots = 256

    st.header('Square Pulse Generator')
    target_channel = st.selectbox('Choose Target Channel', ['σ_x', 'σ_y'], key='square_target_channel') # user selects target channel
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
        else:
            pulse_vector = st.session_state.sigma_y_vec
            updated_pulse_vector = add_square(pulse_vector, amp, start, stop, n_steps, t_final) # add a square pulse to the sigma y vector
            st.session_state.sigma_y_vec = updated_pulse_vector
            st.session_state.sigma_x_vec = np.pad(st.session_state.sigma_x_vec, (0, len(updated_pulse_vector) - len(st.session_state.sigma_x_vec)), 'constant', constant_values=st.session_state.sigma_x_vec[-1])
        
    if st.button('Clear Pulse', key='clear_button'):
        st.session_state.sigma_x_vec = tlist*0
        st.session_state.sigma_y_vec = tlist*0

    # Display the Pulse   
    fig_sigma = go.Figure()
    plot_lw = 3 # plot linewidth
    fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_x_vec, mode='lines', name='σ_x', line=dict(width=plot_lw)))
    fig_sigma.add_trace(go.Scatter(x=tlist, y=st.session_state.sigma_y_vec, mode='lines', name='σ_y', line=dict(width=plot_lw)))
    fig_sigma.update_layout(title='Time-Dependent Amplotudes of Ω_0 (Ω_x(t) and Ω_x(t)) ', xaxis_title='Time (ns)', yaxis_title='Amplitude', xaxis=dict(range=[0, t_final]), yaxis=dict(range=[-1.05, 1.05]))
        
    st.plotly_chart(fig_sigma)
        


    # Actual Simulation Part 
    if st.button('Run Simulation'): # button to run simulation
        params = (omega_q, omega_rabi*1e-3, t_final, n_steps, omega_d, st.session_state.sigma_x_vec, st.session_state.sigma_y_vec, num_shots, T1, T2) # set the parameters for the simulation

        try:
            exp_values, __, sampled_probabilities = run_quantum_simulation(*params) # run the quantum simulation
        except Exception as e:
            st.warning(f'An error occured, refresh the page and try again')

        time_array = np.linspace(0, t_final, n_steps) # set the time array
        exp_y_rotating = -(exp_values[0] * np.cos(2*np.pi*omega_d*time_array) + exp_values[1] * np.sin(2*np.pi*omega_d*time_array)) # set the y rotating expectation values
        exp_x_rotating = exp_values[0] * np.sin(2*np.pi*omega_d*time_array) - exp_values[1] * np.cos(2*np.pi*omega_d*time_array)

        # NEEDS FIX, plotting the bloch sphere should update the bloch sphere at the top of the page not generate a new one below
        time_values = np.linspace(0, t_final, n_steps) # create list of time values 
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j] # create a meshgrid of u and v values
        x_sphere = np.cos(u) * np.sin(v) # set x values of the sphere
        y_sphere = np.sin(u) * np.sin(v) # set y values of the sphere
        z_sphere = np.cos(v)

        colors = time_values # set colors to time values

        fig_bloch = go.Figure(data=[
            go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3),
            go.Scatter3d(
                x = exp_x_rotating,
                y = exp_y_rotating,
                z = exp_values[2],
                mode = 'markers',
                marker = dict (size=6, color=colors, opacity=0.8, colorscale='Viridis', colorbar=dict(title='Time', tickvals=[0, t_final], ticktext=[0, t_final]))
            )])
            
        fig_bloch.update_layout(title='Bloch Sphere Trajectory', scene=dict(xaxis_title='σ_x', yaxis_title='σ_y', zaxis_title='σ_z',
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1])),
            margin=dict(l=0, r=0, b=0, t=0))
            
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
            
        st.plotly_chart(fig_bloch)

if __name__ == "__main__":
    main()