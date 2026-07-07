"""Regression tests for the cached simulator hot path (C02-T10).

Guards the @st.cache_data refactor of ``run_quantum_simulation``: the
deterministic Lindblad solve must be stable across identical calls, while the
vectorized shot sampling stays stochastic and tracks P(|1>).
"""
import numpy as np

import quantum_simulator as qs

N = 24
COMMON = dict(
    omega_q=5.0, omega_rabi=0.05, t_final=20.0, n_steps=N, omega_d=5.0,
    user_vector_I=np.ones(N), user_vector_Q=np.zeros(N), T1=0.0, T2=0.0,
)


def test_solve_is_cached():
    assert hasattr(qs._solve_mesolve, "clear"), "expected an @st.cache_data function"


def test_shapes_and_bounds():
    expect, probs, sampled = qs.run_quantum_simulation(num_shots=256, **COMMON)
    assert len(expect) == 3 and all(len(e) == N for e in expect)
    assert len(probs) == N and all(0.0 <= p <= 1.0 for p in probs)
    assert len(sampled) == N and all(0.0 <= s <= 1.0 for s in sampled)


def test_solve_deterministic_but_sampling_stochastic():
    _, p1, s1 = qs.run_quantum_simulation(num_shots=256, **COMMON)
    _, p2, s2 = qs.run_quantum_simulation(num_shots=256, **COMMON)
    # The expensive solve is deterministic (and cache-stable)…
    assert np.allclose(p1, p2)
    # …but the shot sampling is fresh each call.
    assert not np.allclose(s1, s2)


def test_sampling_tracks_probabilities():
    _, probs, sampled = qs.run_quantum_simulation(num_shots=4000, **COMMON)
    # With many shots the sampled |1> fraction is close to the true probability.
    assert float(np.mean(np.abs(np.array(sampled) - np.array(probs)))) < 0.05
