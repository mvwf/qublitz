# quantum_simulator.py
import numpy as np
from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmam, Options

def run_frequency_sweep(start_freq, stop_freq, num_points, t_final, n_steps, omega_q, omega_rabi, T1, T2, num_shots):
    frequencies = np.linspace(start_freq, stop_freq, num_points)
    time_list = np.linspace(0, t_final, n_steps)
    expectation_values = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
    prob_1_time_series = []

    user_vector_I = np.ones(n_steps)
    user_vector_Q = np.zeros(n_steps)
    
    for omega_d in frequencies:
        exp_values, probabilities, _ = run_quantum_simulation(
            omega_q, omega_rabi, t_final, n_steps, omega_d, user_vector_I, user_vector_Q, num_shots, T1, T2
        )
        expectation_values['sigma_x'].append(exp_values[0])
        expectation_values['sigma_y'].append(exp_values[1])
        expectation_values['sigma_z'].append(exp_values[2])
        prob_1 = [1 - p for p in probabilities]
        prob_1_time_series.append(prob_1)

    return {
        'frequencies': frequencies,
        'time_list': time_list,
        'expectation_values': expectation_values,
        'prob_1_time_series': prob_1_time_series,
    }

def run_quantum_simulation(omega_q, omega_rabi, t_final, n_steps, omega_d, user_vector_I, user_vector_Q, num_shots, T1, T2):
    tlist = np.linspace(0, t_final, n_steps)

    H0 =  2 * np.pi * omega_q * sigmaz() / 2
    # FACTOR OF 2 FIX: Removed the incorrect '/ 2' from the drive terms. 
    # A lab frame amplitude of Omega leads to a rotating frame Rabi rate of exactly Omega.
    H1 =  2 * np.pi * omega_rabi * sigmax() 
    H2 =  2 * np.pi * omega_rabi * sigmay() 

    H = [H0, [H1, lambda t, args: user_vector_I[min(int(t / t_final * n_steps), n_steps - 1)] * np.cos(args['w'] * t)], 
             [H2, lambda t, args: user_vector_Q[min(int(t / t_final * n_steps), n_steps - 1)] * np.cos(args['w'] * t)]]
    
    psi0 = basis(2, 1) # Defining |1> (ground state)

    # HARDCODE T2 in terms of T1 (purely T1-limited)
    if T1 > 0:
        T2 = 2.0 * T1 

    c_ops = []
    if T1 > 0:
        rate_1 = 1.0 / T1
        c_ops.append(np.sqrt(rate_1) * sigmam())

    if T2 > 0:
        # FACTOR OF 2 FIX: Pure dephasing operator coefficient must be divided by 2
        gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
        if gamma_phi > 0:
            c_ops.append(np.sqrt(gamma_phi / 2.0) * sigmaz())

    # OPTION BUG FIX: Explicitly instantiate the Options object for QuTiP backwards-compatibility
    try:
        options = Options()
        options.store_states = True
        options.nsteps = 10000
    except Exception:
        options = {"store_states": True, "nsteps": 10000}

    result = mesolve(H, psi0, tlist, c_ops, [sigmax(), sigmay(), sigmaz()], args={'w': 2 * np.pi * omega_d}, options=options)

    probabilities = []
    sampled_probabilities = []

    for state in result.states:
        prob_0 = np.abs(state[0, 0])**2
        prob_1 = 1 - prob_0
        probabilities.append(prob_1)
        samples = np.random.choice([0, 1], size=num_shots, p=[prob_0, prob_1])
        sampled_prob_1 = np.sum(samples == 0) / num_shots
        sampled_probabilities.append(sampled_prob_1)

    return result.expect, probabilities, sampled_probabilities