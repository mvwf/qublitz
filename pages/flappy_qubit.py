'''
Authors:
    M.F. Fitzpatrick

Release Date: 
    V 1.0: 08/09/2025

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
from PIL import Image
from qutip import sigmax, sigmay, sigmaz
import time
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
def run_main_logic():
    # Start button and auto-advance logic
    if 'fq_running' not in st.session_state:
        st.session_state.fq_running = False
    start_col, stop_col = st.columns([1,1])
    with start_col:
        if st.button('Start Game'):
            st.session_state.fq_running = True
            st.session_state.last_step_time = time.time() - 1.0  # So first auto-advance is in 1 sec
    with stop_col:
        if st.button('Pause'):
            st.session_state.fq_running = False
    # Only auto-advance if running
    auto_advance = False
    if st.session_state.fq_running:
        if 'last_step_time' not in st.session_state:
            st.session_state.last_step_time = time.time() - 1.0
        if time.time() - st.session_state.last_step_time > 1.0:
            auto_advance = True
            st.session_state.last_step_time = time.time()
    # Ensure fq_traj is initialized before use
    if 'fq_traj' not in st.session_state:
        st.session_state.fq_traj = []

    st.title('Qublitz Virtual Qubit Lab') # site title
    qublitz_logo = Image.open("images/qublitz.png")
    st.sidebar.image(qublitz_logo)
    logo = Image.open("images/logo.png") 
    st.sidebar.image(logo) # display logo on the side 
    st.sidebar.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)
    
    st.header('This app simulates the dynamics of a driven qubit (two-level system)')
    st.subheader(r'$\hat{H/\hbar} = \frac{\omega_q}{2}\hat{\sigma}_z + \frac{\Omega(t)}{2}\hat{\sigma}_x\cos(\omega_d t) + \frac{\Omega(t)}{2}\hat{\sigma}_y\cos(\omega_d t)$') # Hamiltonian 
    st.latex(r'''\text{Where } |1\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \text{ and } |0\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}''') # basis vectors

    st.header('Simulation Parameters')

    user_selection = "Video Game"  # Default to Video Game mode

    # Video Game mode parameters
    GAME_LEVELS = [
        {"name": "Level 1: Pi Pulse", "target": (0, 0, 1), "desc": "Design a pulse to move the qubit from |0⟩ (bottom) to |1⟩ (top) of the Bloch sphere."},
        {"name": "Level 2: Hadamard Gate", "target": (1, 0, 0), "desc": "Design a pulse to move the qubit to the Hadamard state (⟨σ_x⟩ ≈ 1, ⟨σ_z⟩ ≈ 0)."},
    ]

    
    # Level selection (for now, just Level 1)
    level_idx = st.session_state.get('game_level', 0)
    level = GAME_LEVELS[level_idx]
    st.success(f"{level['name']}")
    st.info(level['desc'])
    # Difficulty selection: Easy (infinite T1), Medium/Hard (finite T1)
    difficulty = st.selectbox("Select Difficulty", ["Easy (No Dissipation)", "Medium (T1=500)", "Hard (T1=200)"])
    if difficulty == "Easy (No Dissipation)":
        T1 = float('inf')
        T2 = float('inf')
        pipe_interval_default = 18
        gap_halfwidth_default = 0.4
    elif difficulty == "Medium (T1=500)":
        T1 = 500.0
        T2 = 500.0
        pipe_interval_default = 15
        gap_halfwidth_default = 0.3
    else:
        T1 = 200.0
        T2 = 200.0
        pipe_interval_default = 12
        gap_halfwidth_default = 0.2
    # Ensure pipe parameters are always initialized
    st.session_state.fq_pipe_interval = pipe_interval_default
    st.session_state.fq_gap_halfwidth = gap_halfwidth_default
    # Initialize ground state button
    t_final = 100.0
    n_steps = 25*int(t_final)
    num_shots = 256
    if st.button('Initi    import timealize Ground State'):
        tlist = np.linspace(0, t_final, n_steps)
        st.session_state.sigma_x_vec = tlist*0
        st.session_state.sigma_y_vec = tlist*0
        st.session_state['initialized'] = True
        st.success("Qubit initialized to ground state |0⟩.")
    st.info(f"Qubit frequency for this level: ω_q/2π = {st.session_state.get('omega_q_game', np.random.uniform(3.0, 5.0)):.3f} GHz")
    omega_q = st.session_state.get('omega_q_game', np.random.uniform(3.0, 5.0))
    omega_d = omega_q
    omega_rabi = st.number_input(r'Rabi Rate $\Omega_0/2\pi$ [MHz]', 0.0, value=50.0, step=1.0, key='rabi_time_domain_game')
    # --- Real-time Flappy Qubit Game Loop ---
    st.header('Flappy Qubit Controls')
    # Game state: current step, quantum state, trajectory, expectation values, timer
    from qutip import Qobj
    if 'fq_step' not in st.session_state:
        st.session_state.fq_step = 0
    if 'fq_traj' not in st.session_state:
        st.session_state.fq_traj = []
    if 'fq_state' not in st.session_state:
        # Initialize to ground state density matrix
        st.session_state.fq_state = Qobj([[0,0],[0,1]]) # |0><0| in QuTiP convention
    if 'fq_exp' not in st.session_state:
        st.session_state.fq_exp = []
    if 'fq_time' not in st.session_state:
        st.session_state.fq_time = time.time()
    if 'fq_game_over' not in st.session_state:
        st.session_state.fq_game_over = False
    if 'fq_pulse_hist' not in st.session_state:
        st.session_state.fq_pulse_hist = []
    if 'fq_score' not in st.session_state:
        st.session_state.fq_score = 0
    if 'fq_pipes' not in st.session_state:
        st.session_state.fq_pipes = []  # each: {step, gap_center}



    col_restart, col_spacer = st.columns([1,3])
    with col_restart:
        if st.button('Restart Game'):
            st.session_state.fq_step = 0
            st.session_state.fq_traj = []
            st.session_state.fq_exp = []
            st.session_state.fq_pulse_hist = []
            st.session_state.fq_score = 0
            st.session_state.fq_pipes = []
            st.session_state.fq_game_over = False
            st.session_state.fq_time = time.time()
            st.session_state.last_step_time = time.time()
            st.session_state.fq_state = Qobj([[0,0],[0,1]])
            st.rerun()
    # Controls
    colx1, colx2, coly1, coly2 = st.columns(4)
    with colx1:
        up_pressed = st.button('↑ σ_x+', key='up_x')
    with colx2:
        down_pressed = st.button('↓ σ_x-', key='down_x')
    with coly1:
        w_pressed = st.button('↑ σ_y+', key='w_y')
    with coly2:
        s_pressed = st.button('↓ σ_y-', key='s_y')
    # Move Bloch sphere plot below control buttons
    # Bloch sphere visualization of trajectory
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    traj = np.array(st.session_state.fq_traj) if st.session_state.fq_traj else np.array([[0,0,-1]])
    fig_bloch = go.Figure(data=[
        go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3),
        go.Scatter3d(
            x=traj[:,0],
            y=traj[:,1],
            z=traj[:,2],
            mode='lines+markers',
            marker=dict(size=6, color=np.arange(len(traj)), colorscale='inferno', opacity=0.8, colorbar=dict(title='Step'))
        )
    ])
    fig_bloch.update_layout(
        title='Bloch Sphere Trajectory',
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
        textfont=dict(color=["white", "white"], size=20)
    ))
    # Add wireframe box to visualize bounds
    box_lines = []
    bounds = [-1, 1]
    for x in bounds:
        for y in bounds:
            box_lines.append(go.Scatter3d(x=[x, x], y=[y, y], z=[-1, 1], mode='lines', line=dict(color='gray', width=2, dash='dot'), showlegend=False))
    for x in bounds:
        for z in bounds:
            box_lines.append(go.Scatter3d(x=[x, x], y=[-1, 1], z=[z, z], mode='lines', line=dict(color='gray', width=2, dash='dot'), showlegend=False))
    for y in bounds:
        for z in bounds:
            box_lines.append(go.Scatter3d(x=[-1, 1], y=[y, y], z=[z, z], mode='lines', line=dict(color='gray', width=2, dash='dot'), showlegend=False))
    for trace in box_lines:
        fig_bloch.add_trace(trace)
    st.plotly_chart(fig_bloch)
    # Advance if any pulse button pressed or auto-advance
    step_triggered = auto_advance or up_pressed or down_pressed or w_pressed or s_pressed

    max_time = 100.0  # seconds for the race
    elapsed = time.time() - st.session_state.fq_time
    st.progress(min(elapsed / max_time, 1.0), text=f"Time: {elapsed:.1f}s / {max_time}s")
    if elapsed > max_time:
        st.session_state.fq_game_over = True
        st.error('⏰ Time is up!')
    # Control logic
    dt = t_final / 50  # 20 steps per game
    omega_x = 0.0
    omega_y = 0.0
    if up_pressed:
        omega_x = 1.0
    elif down_pressed:
        omega_x = -1.0
    if w_pressed:
        omega_y = 1.0
    elif s_pressed:
        omega_y = -1.0
    # Track global simulation time for correct drive phase
    if 'fq_global_time' not in st.session_state:
        st.session_state.fq_global_time = 0.0
    # Only update if step button pressed or auto-advance, and game not over
    if step_triggered and not st.session_state.fq_game_over:
        from quantum_simulator import evolve_density_matrix_step
        rho = st.session_state.fq_state
        # Evolve for one step, pass global time offset
        rho_new = evolve_density_matrix_step(
            rho, omega_q, omega_rabi, omega_d, omega_x, omega_y, dt, T1, T2, t0=st.session_state.fq_global_time)
        st.session_state.fq_state = rho_new
        # Compute expectation values
        exp_x = (rho_new * sigmax()).tr().real
        exp_y = (rho_new * sigmay()).tr().real
        exp_z = (rho_new * sigmaz()).tr().real
        st.session_state.fq_exp.append([exp_x, exp_y, exp_z])
        st.session_state.fq_step += 1
        st.session_state.fq_traj.append([exp_x, exp_y, exp_z])
        st.session_state.fq_pulse_hist.append([omega_x, omega_y])
        st.session_state.fq_global_time += dt
        # Flappy Bird mechanics: pipes/GAP checks at scheduled steps
        # Create a pipe at this step if due
        if 'fq_next_pipe_step' not in st.session_state:
            st.session_state.fq_next_pipe_step = 10
        if st.session_state.fq_step >= st.session_state.fq_next_pipe_step:
            gap_center = float(np.random.uniform(-0.7, 0.7))
            st.session_state.fq_pipes.append({
                'step': st.session_state.fq_step,
                'gap_center': gap_center,
            })
            # Check pass/fail
            if abs(exp_z - gap_center) <= st.session_state.fq_gap_halfwidth:
                st.session_state.fq_score += 1
            else:
                st.session_state.fq_game_over = True
            # schedule next pipe
            st.session_state.fq_next_pipe_step += st.session_state.fq_pipe_interval
        # Always rerun after a step to keep auto-advance working
        st.rerun()
    # Show current state
    st.write(f'Step: {st.session_state.fq_step}')
    st.write(f'Current Bloch vector: {st.session_state.fq_traj[-1] if st.session_state.fq_traj else [0,0,-1]}')
    st.metric(label='Score', value=st.session_state.get('fq_score', 0))
    # Plot running expectation values (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩) with reduced height
    fq_exp_arr = np.array(st.session_state.fq_exp) if st.session_state.fq_exp else np.zeros((1,3))
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(x=np.arange(len(fq_exp_arr)), y=fq_exp_arr[:,0], mode='lines+markers', name='⟨σ_x⟩'))
    fig_exp.add_trace(go.Scatter(x=np.arange(len(fq_exp_arr)), y=fq_exp_arr[:,1], mode='lines+markers', name='⟨σ_y⟩'))
    fig_exp.add_trace(go.Scatter(x=np.arange(len(fq_exp_arr)), y=fq_exp_arr[:,2], mode='lines+markers', name='⟨σ_z⟩'))
    fig_exp.update_layout(title='Expectation Values', xaxis_title='Step', yaxis_title='Value', yaxis=dict(range=[-1.05,1.05]), height=250)
    st.plotly_chart(fig_exp)
    # Plot pulse history (Ω_x and Ω_y on same plot) with reduced height
    fq_pulse_arr = np.array(st.session_state.fq_pulse_hist) if st.session_state.fq_pulse_hist else np.zeros((1,2))
    fig_pulse = go.Figure()
    fig_pulse.add_trace(go.Scatter(x=np.arange(len(fq_pulse_arr)), y=fq_pulse_arr[:,0], mode='lines+markers', name='Ω_x'))
    fig_pulse.add_trace(go.Scatter(x=np.arange(len(fq_pulse_arr)), y=fq_pulse_arr[:,1], mode='lines+markers', name='Ω_y'))
    fig_pulse.update_layout(title='Pulse History', xaxis_title='Step', yaxis_title='Amplitude', yaxis=dict(range=[-1.05,1.05]), height=250)
    st.plotly_chart(fig_pulse)
    # Removed Flappy Track plot

if __name__ == "__main__":
    run_main_logic()