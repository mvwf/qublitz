# quantum_simulator.py
import numpy as np
from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmap, sigmam, Options
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
    tlist = np.linspace(0, t_final, n_steps)

    # Hamiltonian and other setup as before... fac
    # setting hbar to 1

    H0 =  2*np.pi*omega_q * sigmaz() / 2
    H1 =  2*np.pi*omega_rabi * sigmax() / 2
    H2 =  2*np.pi*omega_rabi * sigmay()  / 2

    H = [H0, [H1, lambda t, args: user_vector_I[min(int(t / t_final * n_steps), n_steps - 1)] * np.cos(args['w'] * t)], 
         [H2, lambda t, args: user_vector_Q[min(int(t / t_final * n_steps), n_steps - 1)] * np.cos(args['w'] * t)]]
    
    psi0 = basis(2, 0)

    # Collapse operators for T1 and T2
    c_ops = []
    if T1 > 0:
        rate_1 = 1.0 / T1
        c_ops.append(np.sqrt(rate_1) * sigmap())  # Relaxation. Note, we are defining the |0> = [1,0] to be the ground state

    if T2 > 0:
        rate_2 = 1.0 / T2 - 1.0 / (2 * T1)  # Dephasing rate is T2* - 1/(2*T1)
        c_ops.append(np.sqrt(rate_2) * -sigmaz())  # Dephasing


    # Set QuTiP Options to increase nsteps
    options = Options(nsteps=5000)
    options.store_states = True

    # Mesolve with collapse operators
    result = mesolve(H, psi0, tlist, c_ops, [sigmax(), sigmay(), sigmaz()], args={'w': 2 * np.pi * omega_d}, options=options)

    probabilities = []
    sampled_probabilities = []

    for state in result.states:
        # print(state)
        prob_0 = np.abs(state[0, 0])**2 # Probability of being in state |0>
        # prob_1 = state[1, 1].real  # Probability of being in state |1>
        probabilities.append(prob_0)

        # Sample the distribution
        samples = np.random.choice([0, 1], size=num_shots, p=[prob_0, 1 - prob_0])
        sampled_prob_0 = np.sum(samples == 0) / num_shots
        sampled_probabilities.append(sampled_prob_0)

    # print(len(probabilities))
    return result.expect, probabilities, sampled_probabilities
