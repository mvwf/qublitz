# quantum_simulator.py

import numpy as np
from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmam

# Define your run_quantum_simulation function
def run_quantum_simulation(omega_z, omega_rabi, t_final, n_steps, omega_d, user_vector_I, user_vector_Q, num_shots, T1, T2):
    tlist = np.linspace(0, t_final, n_steps)

    # Hamiltonian and other setup as before...
    H0 = 2 * np.pi * omega_z * sigmaz() / 2
    H1 = 2 * np.pi * omega_rabi * sigmax()
    H2 = 2 * np.pi * omega_rabi * sigmay()

    H = [H0, [H1, lambda t, args: user_vector_I[min(int(t / t_final * n_steps), n_steps - 1)] * np.cos(args['w'] * t)], 
         [H2, lambda t, args: user_vector_Q[min(int(t / t_final * n_steps), n_steps - 1)] * np.sin(args['w'] * t)]]

    # Initial state and ME solver
    psi0 = basis(2, 1)

    # Collapse operators for T1 and T2
    c_ops = []
    if T1 > 0:
        rate_1 = 1.0 / T1
        c_ops.append(np.sqrt(rate_1) * sigmam())  # Relaxation

    if T2 > 0:
        rate_2 = 1.0 / T2 - 1.0 / (2 * T1)  # Dephasing rate is T2* - 1/(2*T1)
        c_ops.append(np.sqrt(rate_2) * sigmaz())  # Dephasing

    # Mesolve with collapse operators
    result = mesolve(H, psi0, tlist, c_ops, [sigmaz()], args={'w': 2 * np.pi * omega_d})

    # Probability calculation for each time step
    prob_0_list = [(expect + 1) / 2 for expect in result.expect[0]]

    # Sampling and reaveraging for each time step
    sampled_results_list = []
    for prob_0 in prob_0_list:
        sampled_results = []
        for _ in range(num_shots):
            random_number = np.random.rand()
            if random_number > prob_0:
                sampled_results.append(0)
            else:
                sampled_results.append(1)
        # Calculate the reaveraged probability for this time step
        reaveraged_prob_0 = np.mean(sampled_results)
        sampled_results_list.append(reaveraged_prob_0)

    return sampled_results_list

def add_gaussian(pulse_vector, amplitude, sigma, desired_time, n_steps):
    """
    Adds a Gaussian to the pulse sequence at the desired time.

    Parameters:
    pulse_vector: The original pulse vector to modify.
    amplitude: The amplitude of the Gaussian.
    sigma: The standard deviation of the Gaussian.
    desired_time: The desired time in the time trace where the Gaussian should be centered.
    n_steps: Number of steps in the pulse vector.
    
    Returns:
    Updated pulse vector with the Gaussian applied.
    """
    tlist = np.linspace(0, n_steps-1, n_steps)
    center = (desired_time / tlist[-1]) * (n_steps - 1)
    gaussian = amplitude * np.exp(-((tlist - center) ** 2) / (2 * sigma ** 2))
    new_pulse = pulse_vector + gaussian
    # Constrain the values between -1 and 1
    new_pulse = np.clip(new_pulse, -1, 1)
    return new_pulse
