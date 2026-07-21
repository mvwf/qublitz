"""Regression test for the Custom Qubit Query frequency-window fix.

Bug (fix/custom-query-freq-window): the non-debug default drive-sweep window was
hardcoded to 4.8-5.2 GHz, but every qubit this app assigns sits BELOW it
(deployed secrets example omega_q = 4.027 GHz; debug student = 4.671 GHz). A
first sweep therefore returned a flat spectrum with no resonance -- which reads
to a student as "the qubit frequency can't be tuned / doesn't respond".

These tests lock the fix at the physics level: the OLD window is flat (documents
the bug), the NEW default window resolves a clear resonance at omega_q.
"""
import numpy as np

import quantum_simulator as qs

# Qubit frequencies this app actually assigns (from the page source).
ASSIGNED_QUBIT_FREQS_GHZ = (4.027, 4.671431666715805)

OLD_WINDOW = (4.8, 5.2)   # pre-fix default -- excludes every assigned qubit
NEW_WINDOW = (3.5, 5.5)   # post-fix default -- contains the assigned band


def _max_prob_over_window(omega_q, start, stop, num_points=21):
    """Peak-over-time P(|1>) at each swept drive frequency."""
    t_final, n_steps = 25.0, 200
    T1 = 77.9
    out = qs.run_frequency_sweep(
        start_freq=start, stop_freq=stop, num_points=num_points,
        t_final=t_final, n_steps=n_steps,
        omega_q=omega_q, omega_rabi=0.2, T1=T1, T2=2.0 * T1, num_shots=0,
    )
    Z = np.asarray(out["prob_1_time_series"], dtype=float)  # (num_points, n_steps)
    freqs = np.asarray(out["frequencies"], dtype=float)
    return freqs, Z.max(axis=1)


def test_new_default_window_contains_assigned_band():
    lo, hi = NEW_WINDOW
    for f in ASSIGNED_QUBIT_FREQS_GHZ:
        assert lo <= f <= hi, f"assigned qubit {f} GHz falls outside {NEW_WINDOW}"


def test_old_default_window_missed_the_assigned_band():
    # Documents the bug: the old window excluded every assigned qubit.
    lo, hi = OLD_WINDOW
    assert all(not (lo <= f <= hi) for f in ASSIGNED_QUBIT_FREQS_GHZ)


def test_new_window_resolves_resonance_old_window_is_flat():
    omega_q = 4.027  # the deployed secrets example
    freqs, max_new = _max_prob_over_window(omega_q, *NEW_WINDOW)
    _, max_old = _max_prob_over_window(omega_q, *OLD_WINDOW)

    # NEW window: a real resonance peak, located at omega_q.
    assert max_new.max() > 0.6, "new window should show a clear resonance"
    peak_freq = float(freqs[int(np.argmax(max_new))])
    assert abs(peak_freq - omega_q) < 0.15, "peak should sit at the qubit frequency"

    # OLD window: flat -- no resonance anywhere (this is what looked broken).
    assert max_old.max() < 0.35, "old window should be flat (bug reproduction)"
