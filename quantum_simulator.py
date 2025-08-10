# Stepwise evolution function for density matrix
def evolve_density_matrix_step(rho, omega_q, omega_rabi, omega_d, amp_x, amp_y, dt, T1, T2, t0=0.0):
    """
    Evolve a density matrix for a single time step dt with given drive amplitudes.
    Args:
        rho: qutip.Qobj, initial density matrix
        omega_q: qubit frequency (GHz)
        omega_rabi: Rabi frequency (MHz)
        omega_d: drive frequency (GHz)
        amp_x: amplitude for sigma_x
        amp_y: amplitude for sigma_y
        dt: time step (ns)
        T1: relaxation time (ns)
        T2: dephasing time (ns)
        t0: global time offset (ns)
    Returns:
        qutip.Qobj, evolved density matrix
    """
    # Hamiltonian for this step
    H0 = 2 * np.pi * omega_q * sigmaz() / 2
    H1 = 2 * np.pi * omega_rabi * sigmax() / 2
    H2 = 2 * np.pi * omega_rabi * sigmay() / 2
    # Use time-dependent amplitudes for this step
    H = [H0,
         [H1, lambda t, args: amp_x * np.cos(args['w'] * (t + t0))],
         [H2, lambda t, args: amp_y * np.cos(args['w'] * (t + t0))]]
    # Collapse operators
    c_ops = []
    if T1 > 0 and np.isfinite(T1):
        rate_1 = 1.0 / T1
        c_ops.append(np.sqrt(rate_1) * sigmam())
    if T2 > 0 and np.isfinite(T2):
        # Ensure non-negative pure dephasing rate
        rate_2 = max(0.0, 1.0 / T2 - (0.5 / T1 if (T1 > 0 and np.isfinite(T1)) else 0.0))
        if rate_2 > 0:
            c_ops.append(np.sqrt(rate_2) * (sigmaz() * -1))
    # Time array for this step: a few points to ease integration
    n_step_points = 5
    tlist = np.linspace(0, dt, n_step_points)
    # Solver options: increase allowed substeps and tighten tolerances
    opts = Options(nsteps=10000, atol=1e-8, rtol=1e-6)
    # Mesolve for this step
    result = mesolve(H, rho, tlist, c_ops, [], args={'w': 2 * np.pi * omega_d}, options=opts)
    # Return final density matrix
    return result.states[-1]
# quantum_simulator.py
import numpy as np
from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmam, Options
def run_frequency_sweep(start_freq, stop_freq, num_points, t_final, n_steps, omega_q, omega_rabi, T1, T2, num_shots):
    """
    Performs a frequency sweep by running the quantum simulation across a range of drive frequencies.

    Parameters:
    - start_freq: Starting frequency of the sweep in GHz.
    - stop_freq: Stopping frequency of the sweep in GHz.
    - num_points: Number of points in the frequency sweep.
    - t_final: Total time for each simulation in ns.
    - n_steps: Number of time steps in each simulation.
    - omega_rabi: Rabi frequency in MHz.
    - T1: Relaxation time constant in ns.
    - T2: Dephasing time constant in ns.
    - num_shots: Number of measurements for each simulation.
    - constant_I: Constant in-phase control signal amplitude for all frequencies.

    Returns:
    A dictionary containing the sweep frequencies, final probabilities of being in state |0>, and final expectation values.
    """
    frequencies = np.linspace(start_freq, stop_freq, num_points)
    time_list = np.linspace(0, t_final, n_steps)
    expectation_values = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
    prob_1_time_series = []  # List to store probability of being in |1> state time series for each frequency

    # Constant in-phase control signal amplitude
    user_vector_I = np.ones(n_steps)
    user_vector_Q = np.zeros(n_steps)
    
    for omega_d in frequencies:
        exp_values, probabilities, _ = run_quantum_simulation(omega_q, omega_rabi, t_final, n_steps, omega_d, user_vector_I, user_vector_Q, num_shots, T1, T2)
        
        # Store full time-resolved expectation values for each sigma
        expectation_values['sigma_x'].append(exp_values[0])
        expectation_values['sigma_y'].append(exp_values[1])
        expectation_values['sigma_z'].append(exp_values[2])
        
        # Calculate and store time-resolved probability of being in state |1>
        prob_1 = [1 - p for p in probabilities]
        prob_1_time_series.append(prob_1)

    return {
        'frequencies': frequencies,
        'time_list': time_list,
        'expectation_values': expectation_values,
        'prob_1_time_series': prob_1_time_series,
    }

# Define your run_quantum_simulation function
def run_quantum_simulation(omega_q, omega_rabi, t_final, n_steps, omega_d, user_vector_I, user_vector_Q, num_shots, T1, T2):
    # Support partial pulse evolution: if user_vector_I or user_vector_Q is shorter than n_steps, pad with zeros
    user_vector_I = np.array(user_vector_I)
    user_vector_Q = np.array(user_vector_Q)
    if len(user_vector_I) < n_steps:
        user_vector_I = np.pad(user_vector_I, (0, n_steps - len(user_vector_I)), 'constant')
    if len(user_vector_Q) < n_steps:
        user_vector_Q = np.pad(user_vector_Q, (0, n_steps - len(user_vector_Q)), 'constant')
    tlist = np.linspace(0, t_final, n_steps)

    # Hamiltonian and other setup as before... fac
    # setting hbar to 1

    H0 =  2*np.pi*omega_q * sigmaz() / 2
    H1 =  2*np.pi*omega_rabi * sigmax() / 2
    H2 =  2*np.pi*omega_rabi * sigmay()  / 2

    H = [H0, [H1, lambda t, args: user_vector_I[min(int(t / t_final * n_steps), n_steps - 1)] * np.cos(args['w'] * t)], 
         [H2, lambda t, args: user_vector_Q[min(int(t / t_final * n_steps), n_steps - 1)] * np.cos(args['w'] * t)]]
    
    psi0 = basis(2, 1)

    # Collapse operators for T1 and T2
    c_ops = []
    if T1 > 0:
        rate_1 = 1.0 / T1
        c_ops.append(np.sqrt(rate_1) * sigmam())  # Relaxation. Note, we are defining the |1> = [0,1] to be the ground state

    if T2 > 0:
        # Dephasing rate; clamp to non-negative and handle infinite T1
        rate_2 = 1.0 / T2 - (1.0 / (2 * T1) if (T1 > 0 and np.isfinite(T1)) else 0.0)
        rate_2 = max(0.0, rate_2)
        if rate_2 > 0:
            c_ops.append(np.sqrt(rate_2) * (sigmaz() * -1))  # Dephasing

    # Set QuTiP Options to increase nsteps
    options = Options(nsteps=20000, atol=1e-8, rtol=1e-6)
    options.store_states = True

    # Mesolve with collapse operators
    result = mesolve(H, psi0, tlist, c_ops, [sigmax(), sigmay(), sigmaz()], args={'w': 2 * np.pi * omega_d}, options=options)

    probabilities = []
    sampled_probabilities = []

    for state in result.states:
        prob_0 = np.abs(state[0, 0])**2 # Probability of being in state |0>
        prob_1 = 1 - prob_0
        probabilities.append(prob_1)
        # Sample the distribution
        samples = np.random.choice([0, 1], size=num_shots, p=[prob_0, prob_1])
        sampled_prob_1 = np.sum(samples == 0) / num_shots
        sampled_probabilities.append(sampled_prob_1)

    return result.expect, probabilities, sampled_probabilities
