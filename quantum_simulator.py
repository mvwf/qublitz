import numpy as np
from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmam, Options


def _state_excited_population(state):
    """Return excited-state population P(|1>) in the app convention.

    For this simulator:
      - H0 = +(w/2) sigma_z, so basis(2,1) is the lower-energy ground state.
      - The excited state is basis(2,0), i.e. the +z state.
      - Under mesolve with c_ops, states are density matrices, so population is rho[0,0], not |rho[0,0]|^2.
    """
    arr = state.full()
    if arr.shape == (2, 1):
        return float(np.abs(arr[0, 0]) ** 2)
    return float(np.real(arr[0, 0]))


def run_frequency_sweep(start_freq, stop_freq, num_points, t_final, n_steps,
                        omega_q, omega_rabi, T1, T2, num_shots):
    frequencies = np.linspace(start_freq, stop_freq, num_points)
    time_list = np.linspace(0.0, t_final, n_steps)
    expectation_values = {"sigma_x": [], "sigma_y": [], "sigma_z": []}
    prob_1_time_series = []

    user_vector_I = np.ones(n_steps, dtype=float)
    user_vector_Q = np.zeros(n_steps, dtype=float)

    for omega_d in frequencies:
        exp_values, prob_1, _ = run_quantum_simulation(
            omega_q, omega_rabi, t_final, n_steps, omega_d,
            user_vector_I, user_vector_Q, num_shots, T1, T2
        )
        expectation_values['sigma_x'].append(exp_values[0])
        expectation_values['sigma_y'].append(exp_values[1])
        expectation_values['sigma_z'].append(exp_values[2])
        prob_1_time_series.append(prob_1)

    return {
        'frequencies': frequencies,
        'time_list': time_list,
        'expectation_values': expectation_values,
        'prob_1_time_series': prob_1_time_series,
    }


def run_quantum_simulation(omega_q, omega_rabi, t_final, n_steps, omega_d,
                           user_vector_I, user_vector_Q, num_shots, T1, T2):
    tlist = np.linspace(0.0, t_final, n_steps)

    H0 = 2.0 * np.pi * omega_q * sigmaz() / 2.0
    H1 = 2.0 * np.pi * omega_rabi * sigmax()
    H2 = 2.0 * np.pi * omega_rabi * sigmay()

    def _coeff(env):
        return lambda t, args: float(env[min(int(t / t_final * n_steps), n_steps - 1)]) * np.cos(args['w'] * t)

    H = [H0, [H1, _coeff(user_vector_I)], [H2, _coeff(user_vector_Q)]]

    # Start in the lower-energy state of H0. A resonant pi pulse transfers population to |1>.
    psi0 = basis(2, 1)

    if T1 and T1 > 0:
        T2 = 2.0 * T1

    c_ops = []
    if T1 and T1 > 0:
        # Standard physical convention: excited-state population decays as exp(-t/T1).
        c_ops.append(np.sqrt(1.0 / T1) * sigmam())

    if T2 and T2 > 0 and T1 and T1 > 0:
        gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
        if gamma_phi > 1e-15:
            c_ops.append(np.sqrt(gamma_phi / 2.0) * sigmaz())

    try:
        options = Options()
        options.store_states = True
        options.nsteps = 10000
    except Exception:
        options = {"store_states": True, "nsteps": 10000}

    result = mesolve(
        H, psi0, tlist, c_ops, [sigmax(), sigmay(), sigmaz()],
        args={'w': 2.0 * np.pi * omega_d}, options=options
    )

    probabilities = []
    sampled_probabilities = []

    for state in result.states:
        prob_1 = np.clip(_state_excited_population(state), 0.0, 1.0)
        prob_0 = 1.0 - prob_1
        probabilities.append(prob_1)

        samples = np.random.choice([0, 1], size=num_shots, p=[prob_0, prob_1])
        sampled_probabilities.append(float(np.mean(samples == 1)))

    return result.expect, probabilities, sampled_probabilities
