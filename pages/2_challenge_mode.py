import streamlit as st 
import numpy as np 
import qutip as qt
import plotly.graph_objects as go

## Function: Takes in a state vector, applies a Pauli-X gate to the state vector, and returns the new state vector
def x_gate(state_vector):
    # Define the Pauli-X gate
    X = qt.sigmax()
    
    # Apply the gate to the state vector
    new_state_vector = X * state_vector
    
    return new_state_vector

st.title("Challenge Mode")
st.write("Try to do an X gate.")


# Setting Default Parameters
omega_q = 5.000 # Default qubit frequency is 5.000 GHz
omega_d = 5.000 # Default drive frequency is 5.000
omega_rabi = 50.000 # Default Rabi frequency is 50.000 M
t_final = 200 # Default total time for each simulation is 200 ns
T1 = 100000000000 # Default relaxation time constant
T2 = 200000000000 # Default dephasing time constant from 0 to 1000.0 ns
detuning = (omega_d - omega_q)*1e3 # set the detuning based on user inputs 
n_steps = 20*int(t_final) # set the number of steps based on user inputs times 20

# plotting the bloch sphere 
time_values = np.linspace(0, t_final, n_steps) # create list of time values 
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j] # create a meshgrid of u and v values
x_sphere = np.cos(u) * np.sin(v) # set x values of the sphere
y_sphere = np.sin(u) * np.sin(v) # set y values of the sphere
z_sphere = np.cos(v)

colors = time_values # set colors to time values

fig_bloch = go.Figure(data=[
    go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3)])
            
fig_bloch.update_layout(title='Bloch Sphere Trajectory', scene=dict(xaxis_title='σ_x', yaxis_title='σ_y', zaxis_title='σ_z',
    xaxis=dict(range=[-1, 1]),
    yaxis=dict(range=[-1, 1]),
    zaxis=dict(range=[-1, 1])),
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
            
st.plotly_chart(fig_bloch)