# quantum_simulator.py
import numpy as np
from qutip import basis, sigmaz, sigmax, sigmay, mesolve, sigmam, Options


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


    # Set QuTiP Options to increase nsteps
    options = Options(nsteps=10000)

    # Mesolve with collapse operators
    result = mesolve(H, psi0, tlist, c_ops, [sigmax(), sigmay(), sigmaz()], args={'w': 2 * np.pi * omega_d}, options=options)

    exp_values = result.expect
    return exp_values