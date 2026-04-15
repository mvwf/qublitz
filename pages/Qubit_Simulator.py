#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ENGINE CONVENTION
================================================================================
The simulator uses the assigned physical parameters directly:
  - omega_rabi is the physical Rabi rate Ω_R / 2π
  - T1 is the physical energy-relaxation time
  - T2 is hardcoded internally as 2*T1 for the teaching model
================================================================================
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

from quantum_simulator import run_quantum_simulation, run_frequency_sweep


# ============================================================
# Engine-convention helpers
# ============================================================
def _engine_rabi(omega_rabi_physical_GHz: float) -> float:
    return float(omega_rabi_physical_GHz)


def _engine_T1(T1_physical_ns: float) -> float:
    return float(T1_physical_ns)


def _engine_T2(T1_physical_ns: float) -> float:
    return 2.0 * float(T1_physical_ns)


# ----------------------------
# Helpers
# ----------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _time_grid(t_final_ns: float, pts_per_ns: int = 25):
    n_steps = max(2, int(float(pts_per_ns) * float(t_final_ns)))
    tlist = np.linspace(0.0, float(t_final_ns), n_steps)
    return tlist, n_steps


def _interp_to_grid(vec, t_old, t_new, n_new):
    if vec is None:
        return np.zeros(n_new, dtype=float)
    vec = np.asarray(vec, dtype=float)
    if len(vec) == n_new and float(t_old) == float(t_new):
        return vec
    x_old = np.linspace(0.0, float(t_old), len(vec))
    x_new = np.linspace(0.0, float(t_new), n_new)
    return np.interp(x_new, x_old, vec).astype(float)


def _clip_env(env):
    return np.clip(np.asarray(env, dtype=float), -1.0, 1.0)


def _extract_Z(prob_1_data, freqs, tlist):
    prob_1_data = np.asarray(prob_1_data, dtype=float)
    if prob_1_data.shape == (len(freqs), len(tlist)):
        return prob_1_data.T
    if prob_1_data.shape == (len(tlist), len(freqs)):
        return prob_1_data
    return prob_1_data


def _clear_results_and_pulses():
    for key in (
        "freq_out", "freq_peak_cut", "td_out",
        "td_tfinal_last", "td_sigma_x_vec", "td_sigma_y_vec",
        "ramsey_chevron_out", "td_rabi_fit", "td_t1_fit", "freq_rabi_fit",
    ):
        st.session_state[key] = None


def _init_state():
    for key in (
        "user_data",
        "freq_out", "freq_peak_cut",
        "td_out", "td_tfinal_last", "td_sigma_x_vec", "td_sigma_y_vec",
        "ramsey_chevron_out", "td_rabi_fit", "td_t1_fit", "freq_rabi_fit",
    ):
        st.session_state.setdefault(key, None)


def _load_params_from_secrets() -> Dict[str, Any]:
    """
    Accept either of these Streamlit secrets formats:

    [params]
    omega_q = 4.027
    omega_rabi = 0.35
    T1 = 194.0

    or

    omega_q = 4.027
    omega_rabi = 0.35
    T1 = 194.0
    """
    if "params" in st.secrets:
        params = st.secrets["params"]
    else:
        params = st.secrets

    required = ("omega_q", "omega_rabi", "T1")
    missing = [k for k in required if k not in params]
    if missing:
        raise RuntimeError(
            "Missing parameter(s) in Streamlit secrets: " + ", ".join(missing)
        )

    return {
        "user": "student",
        "omega_q": float(params["omega_q"]),
        "omega_rabi": float(params["omega_rabi"]),
        "T1": float(params["T1"]),
    }


def _format_pm(val: Optional[float], err: Optional[float], unit: str = "", digits: int = 3) -> str:
    if val is None or not np.isfinite(val):
        return "fit unavailable"
    val = float(val)
    if digits < 1:
        digits = 1
    fmt = f"{{:.{digits}g}}"
    val_s = fmt.format(val)
    if err is None or not np.isfinite(err) or float(err) <= 0:
        return f"{val_s} {unit}".strip()
    err = float(err)
    scale = max(abs(val), 1e-12)
    if err > 10.0 * scale:
        return f"{val_s} {unit}".strip()
    err_s = fmt.format(err)
    return f"{val_s} ± {err_s} {unit}".strip()


def _active_segments(env, threshold=1e-8):
    env = np.asarray(env, dtype=float)
    mask = np.abs(env) > threshold
    if not np.any(mask):
        return []
    idx = np.where(mask)[0]
    segments = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k != prev + 1:
            segments.append((start, prev))
            start = k
        prev = k
    segments.append((start, prev))
    return segments


# ----------------------------
# Bloch sphere
# ----------------------------
def bloch_sphere_pretty(exp_x, exp_y, exp_z, tlist, t_final):
    exp_x = np.asarray(exp_x, dtype=float)
    exp_y = np.asarray(exp_y, dtype=float)
    exp_z = np.asarray(exp_z, dtype=float)
    tlist = np.asarray(tlist, dtype=float)

    exp_z_plot = -exp_z

    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = -np.cos(v)

    fig = go.Figure(data=[
        go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.25, showscale=False, hoverinfo="skip",
        ),
        go.Scatter3d(
            x=exp_x, y=exp_y, z=exp_z_plot,
            mode="markers",
            marker=dict(
                size=4, color=tlist, opacity=0.9,
                colorscale="Cividis",
                colorbar=dict(
                    title="Time [ns]", len=0.85, y=0.5, thickness=25,
                    tickvals=[float(tlist[0]), float(t_final)],
                    ticktext=[f"{tlist[0]:.0f}", f"{t_final:.0f}"],
                ),
            ),
            name="State",
        ),
    ])
    fig.update_layout(
        title="State vector on the Bloch sphere",
        height=520, margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title="⟨σx⟩", yaxis_title="⟨σy⟩", zaxis_title="⟨σz⟩",
            xaxis=dict(range=[-1, 1], showbackground=False),
            yaxis=dict(range=[-1, 1], showbackground=False),
            zaxis=dict(range=[-1, 1], showbackground=False),
            aspectmode="cube",
        ),
        showlegend=False,
    )
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[1.0, -1.0],
        mode="text", text=["|0⟩", "|1⟩"],
        textposition=["top center", "bottom center"],
        textfont=dict(color=["white", "white"], size=18),
        showlegend=False, hoverinfo="skip",
    ))
    return fig


# ----------------------------
# Pulse UI
# ----------------------------
def pulse_ui(t_final_ns: float):
    tlist, n_steps = _time_grid(t_final_ns, pts_per_ns=25)
    dt_ns = float(tlist[1] - tlist[0]) if len(tlist) > 1 else 0.04
    step_ns = max(0.01, round(dt_ns, 3))

    if st.session_state["td_tfinal_last"] is None:
        st.session_state["td_tfinal_last"] = float(t_final_ns)
    if st.session_state["td_sigma_x_vec"] is None:
        st.session_state["td_sigma_x_vec"] = np.zeros(n_steps, dtype=float)
    if st.session_state["td_sigma_y_vec"] is None:
        st.session_state["td_sigma_y_vec"] = np.zeros(n_steps, dtype=float)

    t_old = float(st.session_state["td_tfinal_last"])
    if float(t_old) != float(t_final_ns) or len(st.session_state["td_sigma_x_vec"]) != n_steps:
        st.session_state["td_sigma_x_vec"] = _interp_to_grid(
            st.session_state["td_sigma_x_vec"], t_old, float(t_final_ns), n_steps
        )
        st.session_state["td_sigma_y_vec"] = _interp_to_grid(
            st.session_state["td_sigma_y_vec"], t_old, float(t_final_ns), n_steps
        )
        st.session_state["td_tfinal_last"] = float(t_final_ns)

    st.markdown("### Pulse parameters")
    st.caption(f"Pulse schedule resolution: approximately {dt_ns:.3f} ns per sample.")

    col1, col2 = st.columns([1, 1])
    with col1:
        target = st.selectbox("Target channel", ["σ_x", "σ_y"], key="td_target")
        ptype = st.selectbox("Pulse type", ["Square", "Gaussian"], key="td_ptype")
        amp = st.slider("Amplitude", -1.0, 1.0, 1.0, step=0.05, key="td_amp")

        if ptype == "Square":
            ns1, ns2 = st.columns(2)
            with ns1:
                start = st.number_input(
                    "Start [ns]",
                    min_value=0.0,
                    max_value=float(max(0.0, t_final_ns - step_ns)),
                    value=min(5.0, float(max(0.0, t_final_ns - step_ns))),
                    step=step_ns,
                    format="%.3f",
                    key="td_sq_start",
                )
            with ns2:
                duration = st.number_input(
                    "Duration [ns]",
                    min_value=step_ns,
                    max_value=float(max(step_ns, t_final_ns - float(start))),
                    value=min(20.0, float(max(step_ns, t_final_ns - float(start)))),
                    step=step_ns,
                    format="%.3f",
                    key="td_sq_duration",
                )
            stop = min(float(t_final_ns), float(start) + float(duration))
            st.caption(f"Computed stop time: {stop:.3f} ns")
        else:
            center = st.number_input(
                "Center [ns]",
                min_value=0.0,
                max_value=float(t_final_ns),
                value=float(min(t_final_ns * 0.5, 50.0)),
                step=step_ns,
                format="%.3f",
                key="td_g_center",
            )
            sigma = st.number_input(
                "Sigma [ns]",
                min_value=step_ns,
                max_value=float(max(step_ns, t_final_ns / 2.0)),
                value=float(max(step_ns, min(10.0, t_final_ns / 10.0))),
                step=step_ns,
                format="%.3f",
                key="td_g_sigma",
            )

        add_btn = st.button("Add pulse", key="td_add")
        clear_btn = st.button("Clear schedule", key="td_clear")

    with col2:
        st.caption(
            "Square pulses use decimal start times and durations. This makes π and π/2 calibrations much easier than forcing integer nanoseconds."
        )

    if clear_btn:
        st.session_state["td_sigma_x_vec"] = np.zeros(n_steps, dtype=float)
        st.session_state["td_sigma_y_vec"] = np.zeros(n_steps, dtype=float)

    if add_btn:
        env = np.zeros(n_steps, dtype=float)
        if ptype == "Square":
            env[(tlist >= float(start)) & (tlist < float(stop))] = float(amp)
        else:
            sig = max(1e-9, float(sigma))
            env = float(amp) * np.exp(-0.5 * ((tlist - float(center)) / sig) ** 2)

        if target == "σ_x":
            st.session_state["td_sigma_x_vec"] = np.asarray(st.session_state["td_sigma_x_vec"], dtype=float) + env
        else:
            st.session_state["td_sigma_y_vec"] = np.asarray(st.session_state["td_sigma_y_vec"], dtype=float) + env

    sx = np.asarray(st.session_state["td_sigma_x_vec"], dtype=float)
    sy = np.asarray(st.session_state["td_sigma_y_vec"], dtype=float)

    fig_sigma = go.Figure()
    fig_sigma.add_trace(go.Scatter(x=tlist, y=sx, mode="lines", name="Ω_x(t)", line=dict(width=3)))
    fig_sigma.add_trace(go.Scatter(x=tlist, y=sy, mode="lines", name="Ω_y(t)", line=dict(width=3)))
    fig_sigma.update_layout(
        xaxis_title="Time [ns]", yaxis_title="Amplitude",
        title="Time-dependent envelopes",
        xaxis=dict(range=[0, float(t_final_ns)]),
        yaxis=dict(range=[-1.05, 1.05]),
        height=320, margin=dict(t=60, b=40, l=60, r=20),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_sigma, use_container_width=True)
    return tlist, sx, sy


# ----------------------------
# Core simulation wrappers
# ----------------------------
def run_time_domain(omega_q_GHz, omega_rabi_GHz, T1_ns, omega_d_GHz,
                    t_final_ns, sx_sched, sy_sched, shots):
    tlist, n_steps = _time_grid(t_final_ns, pts_per_ns=25)
    sx_in = _clip_env(sx_sched)
    sy_in = _clip_env(sy_sched)

    rabi_eng = _engine_rabi(float(omega_rabi_GHz))
    T1_eng = _engine_T1(float(T1_ns))
    T2_eng = _engine_T2(float(T1_ns))

    exp_values, _, sampled_prob = run_quantum_simulation(
        float(omega_q_GHz), rabi_eng, float(t_final_ns), int(n_steps),
        float(omega_d_GHz), sx_in, sy_in, int(shots), T1_eng, T2_eng,
    )

    time_array = np.linspace(0.0, float(t_final_ns), n_steps)
    wd = float(omega_d_GHz)
    exp_y_rot = -(
        exp_values[0] * np.cos(2 * np.pi * wd * time_array)
        + exp_values[1] * np.sin(2 * np.pi * wd * time_array)
    )
    exp_x_rot = (
        exp_values[0] * np.sin(2 * np.pi * wd * time_array)
        - exp_values[1] * np.cos(2 * np.pi * wd * time_array)
    )

    return {
        "tlist": tlist,
        "exp_x_rot": np.asarray(exp_x_rot, dtype=float),
        "exp_y_rot": np.asarray(exp_y_rot, dtype=float),
        "exp_z": np.asarray(exp_values[2], dtype=float),
        "p1_meas": np.asarray(sampled_prob, dtype=float),
    }


def _run_corrected_freq_sweep(start_freq, stop_freq, num_points, spec_tfinal,
                               n_steps, omega_q, omega_rabi, T1_ns, shots):
    rabi_eng = _engine_rabi(float(omega_rabi))
    T1_eng = _engine_T1(float(T1_ns))
    T2_eng = _engine_T2(float(T1_ns))

    return run_frequency_sweep(
        float(start_freq), float(stop_freq), int(num_points),
        float(spec_tfinal), int(n_steps), float(omega_q),
        rabi_eng, T1_eng, T2_eng, int(shots),
    )


# ----------------------------
# Fit models
# ----------------------------
def _rabi_fit_model(t_ns, A, f_MHz, phi, tau_ns, C):
    return C + A * np.exp(-t_ns / tau_ns) * np.cos(2.0 * np.pi * (f_MHz * 1e-3) * t_ns + phi)


def _t1_fit_model(t_ns, A, T1_ns, C):
    return C + A * np.exp(-t_ns / T1_ns)


def _fit_rabi_trace(t_ns, p1_trace):
    t_ns = np.asarray(t_ns, dtype=float)
    p1_trace = np.asarray(p1_trace, dtype=float)
    if len(t_ns) < 10:
        return None
    dt = float(np.median(np.diff(t_ns)))
    if dt <= 0:
        return None

    centered = p1_trace - np.mean(p1_trace)
    freqs = np.fft.rfftfreq(len(t_ns), d=dt)
    spec = np.abs(np.fft.rfft(centered))
    if len(freqs) > 1:
        idx = 1 + int(np.argmax(spec[1:]))
        f0_MHz = max(1e3 * float(freqs[idx]), 0.01)
    else:
        f0_MHz = 1.0

    A0 = 0.5 * (np.max(p1_trace) - np.min(p1_trace))
    tau0 = max(0.6 * float(t_ns[-1] - t_ns[0]), dt)
    C0 = float(np.mean(p1_trace))
    p0 = [A0, f0_MHz, 0.0, tau0, C0]

    if curve_fit is None:
        return {
            "f_MHz": float(f0_MHz),
            "f_MHz_err": None,
            "tau_ns": float(tau0),
            "tau_ns_err": None,
            "fit_curve": _rabi_fit_model(t_ns, *p0),
            "ok": False,
        }

    try:
        popt, pcov = curve_fit(
            _rabi_fit_model,
            t_ns,
            p1_trace,
            p0=p0,
            maxfev=30000,
            bounds=([-1.5, 0.0, -4.0 * np.pi, dt, -0.5], [1.5, 5e4, 4.0 * np.pi, 1e7, 1.5]),
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(5, np.nan)
        return {
            "A": float(popt[0]),
            "f_MHz": float(popt[1]),
            "f_MHz_err": float(perr[1]) if np.isfinite(perr[1]) else None,
            "phi": float(popt[2]),
            "tau_ns": float(popt[3]),
            "tau_ns_err": float(perr[3]) if np.isfinite(perr[3]) else None,
            "C": float(popt[4]),
            "fit_curve": _rabi_fit_model(t_ns, *popt),
            "ok": True,
        }
    except Exception:
        return None


def _fit_t1_trace(t_ns, p1_trace):
    t_ns = np.asarray(t_ns, dtype=float)
    p1_trace = np.asarray(p1_trace, dtype=float)
    if len(t_ns) < 8:
        return None
    dt = float(np.median(np.diff(t_ns)))
    if dt <= 0:
        return None

    A0 = float(np.max(p1_trace) - np.min(p1_trace))
    C0 = float(np.min(p1_trace))
    T10 = max(0.5 * float(t_ns[-1] - t_ns[0]), dt)
    p0 = [A0, T10, C0]

    if curve_fit is None:
        return {
            "T1_ns": float(T10),
            "T1_ns_err": None,
            "fit_curve": _t1_fit_model(t_ns, *p0),
            "ok": False,
        }

    try:
        popt, pcov = curve_fit(
            _t1_fit_model,
            t_ns,
            p1_trace,
            p0=p0,
            maxfev=20000,
            bounds=([0.0, dt, -0.2], [2.0, 1e7, 1.2]),
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(3, np.nan)
        return {
            "A": float(popt[0]),
            "T1_ns": float(popt[1]),
            "T1_ns_err": float(perr[1]) if np.isfinite(perr[1]) else None,
            "C": float(popt[2]),
            "fit_curve": _t1_fit_model(t_ns, *popt),
            "ok": True,
        }
    except Exception:
        return None


def _analyze_time_domain_fits(tlist, p1_trace, sx_sched, sy_sched):
    sx_sched = np.asarray(sx_sched, dtype=float)
    sy_sched = np.asarray(sy_sched, dtype=float)
    tlist = np.asarray(tlist, dtype=float)
    p1_trace = np.asarray(p1_trace, dtype=float)

    results = {
        "rabi": None,
        "t1": None,
        "rabi_slice": None,
        "t1_slice": None,
        "messages": [],
    }

    sx_segments = _active_segments(sx_sched)
    sy_segments = _active_segments(sy_sched)

    if len(sy_segments) == 0 and len(sx_segments) == 1:
        s0, s1 = sx_segments[0]
        if s1 - s0 >= 8:
            rabi_fit = _fit_rabi_trace(tlist[s0:s1 + 1] - tlist[s0], p1_trace[s0:s1 + 1])
            if rabi_fit is not None:
                results["rabi"] = rabi_fit
                results["rabi_slice"] = (s0, s1)
        else:
            results["messages"].append("Rabi fit skipped: the active pulse is too short.")
    else:
        results["messages"].append(
            "Rabi fit is only shown for a single σ_x pulse with no σ_y pulse, because that corresponds to the standard pulse-duration sweep used to extract Ω_R."
        )

    all_segments = sorted(sx_segments + sy_segments, key=lambda x: x[0])
    if all_segments:
        _, last_end = all_segments[-1]
        start_idx = min(last_end + 1, len(tlist) - 1)
        if len(tlist) - start_idx >= 8:
            tail = np.asarray(p1_trace[start_idx:], dtype=float)
            if len(tail) >= 8:
                baseline_est = float(np.median(tail[-max(5, len(tail)//8):]))
                peak_rel = int(np.argmax(np.abs(tail - baseline_est)))
                fit_start = min(start_idx + peak_rel, len(tlist) - 8)
                t_fit = tlist[fit_start:] - tlist[fit_start]
                y_fit = p1_trace[fit_start:]
                t1_fit = _fit_t1_trace(t_fit, y_fit)
                if t1_fit is not None:
                    results["t1"] = t1_fit
                    results["t1_slice"] = (fit_start, len(tlist) - 1)
                else:
                    results["messages"].append("T1 fit skipped: the post-pulse relaxation segment could not be fit.")
        else:
            results["messages"].append("T1 fit skipped: there is not enough idle time after the last pulse.")
    else:
        results["messages"].append("T1 fit skipped: there is no pulse in the current schedule.")

    return results


# ----------------------------
# Ramsey chevron helpers
# ----------------------------
def _rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=float,
    )


def _idle_propagator(delta_GHz, tau_ns, T1_ns, T2_ns):
    delta_ang = 2.0 * np.pi * float(delta_GHz)
    tau_ns = float(tau_ns)
    e2 = np.exp(-tau_ns / float(T2_ns))
    e1 = np.exp(-tau_ns / float(T1_ns))
    c = np.cos(delta_ang * tau_ns)
    s = np.sin(delta_ang * tau_ns)
    M = np.array(
        [[e2 * c, -e2 * s, 0.0],
         [e2 * s,  e2 * c, 0.0],
         [0.0,     0.0,    e1]],
        dtype=float,
    )
    b = np.array([0.0, 0.0, e1 - 1.0], dtype=float)
    return M, b


def _sample_binomial(prob, shots, rng):
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    counts = rng.binomial(int(shots), prob)
    return counts / float(shots)


def _ramsey_p1_trace(delay_list_ns, detuning_MHz, T1_ns, T2_ns, shots=None, seed=None):
    delay_list_ns = np.asarray(delay_list_ns, dtype=float)
    delta_GHz = 1e-3 * float(detuning_MHz)

    Rx_pi2 = _rot_x(np.pi / 2.0)
    r0 = np.array([0.0, 0.0, -1.0], dtype=float)

    p1 = np.zeros_like(delay_list_ns, dtype=float)
    for i, tau in enumerate(delay_list_ns):
        r = Rx_pi2 @ r0
        M, b = _idle_propagator(delta_GHz, tau, T1_ns, T2_ns)
        r = M @ r + b
        r = Rx_pi2 @ r
        p1[i] = np.clip(0.5 * (1.0 + r[2]), 0.0, 1.0)

    if shots is not None and int(shots) > 0:
        rng = np.random.default_rng(seed)
        p1 = _sample_binomial(p1, int(shots), rng)
    return p1


def _ramsey_chevron(delay_list_ns, detuning_list_MHz, T1_ns, T2_ns, shots=None, seed=None, progress_bar=None):
    delay_list_ns = np.asarray(delay_list_ns, dtype=float)
    detuning_list_MHz = np.asarray(detuning_list_MHz, dtype=float)
    Z = np.zeros((len(delay_list_ns), len(detuning_list_MHz)), dtype=float)

    total = max(1, len(delay_list_ns) * len(detuning_list_MHz))
    done = 0
    for j, det in enumerate(detuning_list_MHz):
        col = _ramsey_p1_trace(
            delay_list_ns=delay_list_ns,
            detuning_MHz=float(det),
            T1_ns=float(T1_ns),
            T2_ns=float(T2_ns),
            shots=shots,
            seed=None if seed is None else int(seed) + j,
        )
        Z[:, j] = col
        done += len(delay_list_ns)
        if progress_bar is not None:
            progress_bar.progress(min(done / total, 1.0))
    return Z


def _ramsey_fit_model(t_ns, A, T2_ns, f_MHz, phi, C):
    return C + A * np.exp(-t_ns / T2_ns) * np.cos(2.0 * np.pi * (f_MHz * 1e-3) * t_ns + phi)


def _fit_ramsey_trace(delay_list_ns, p1_trace, freq_guess_MHz=None):
    delay_list_ns = np.asarray(delay_list_ns, dtype=float)
    p1_trace = np.asarray(p1_trace, dtype=float)

    if len(delay_list_ns) < 8:
        return None

    centered = p1_trace - np.mean(p1_trace)
    dt = float(np.median(np.diff(delay_list_ns)))
    if dt <= 0:
        return None

    freqs = np.fft.rfftfreq(len(delay_list_ns), d=dt)
    spec = np.abs(np.fft.rfft(centered))
    if len(freqs) > 1:
        idx = 1 + int(np.argmax(spec[1:]))
        f0_MHz = max(1e3 * float(freqs[idx]), 0.02)
    else:
        f0_MHz = 0.1

    if freq_guess_MHz is not None and np.isfinite(freq_guess_MHz) and abs(freq_guess_MHz) >= 0.02:
        f0_MHz = float(abs(freq_guess_MHz))

    A0 = 0.5 * (np.max(p1_trace) - np.min(p1_trace))
    C0 = float(np.mean(p1_trace))
    T20 = 0.6 * float(np.max(delay_list_ns)) if np.max(delay_list_ns) > 0 else 2.0 * dt
    p0 = [max(A0, 1e-3), max(T20, dt), max(abs(f0_MHz), 0.05), 0.0, C0]

    if curve_fit is None:
        fit_curve = _ramsey_fit_model(delay_list_ns, *p0)
        return {
            "A": float(p0[0]),
            "A_err": None,
            "T2_ns": float(p0[1]),
            "T2_ns_err": None,
            "freq_MHz": float(p0[2]),
            "freq_MHz_err": None,
            "phi": float(p0[3]),
            "C": float(p0[4]),
            "fit_curve": fit_curve,
            "ok": False,
        }

    try:
        popt, pcov = curve_fit(
            _ramsey_fit_model,
            delay_list_ns,
            p1_trace,
            p0=p0,
            maxfev=20000,
            bounds=(
                [-1.5, dt, 0.0, -4.0 * np.pi, -0.5],
                [1.5, 1e6, 5e3, 4.0 * np.pi, 1.5],
            ),
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(5, np.nan)
        return {
            "A": float(popt[0]),
            "A_err": float(perr[0]) if np.isfinite(perr[0]) else None,
            "T2_ns": float(popt[1]),
            "T2_ns_err": (float(perr[1]) if (np.isfinite(perr[1]) and perr[1] <= 10.0 * max(abs(popt[1]), 1e-12)) else None),
            "freq_MHz": float(popt[2]),
            "freq_MHz_err": (float(perr[2]) if (np.isfinite(perr[2]) and perr[2] <= 10.0 * max(abs(popt[2]), 1e-12)) else None),
            "phi": float(popt[3]),
            "C": float(popt[4]),
            "fit_curve": _ramsey_fit_model(delay_list_ns, *popt),
            "ok": True,
        }
    except Exception:
        return None


def _pi_times_from_rabi(omega_rabi_GHz: float):
    omega_rabi_GHz = float(omega_rabi_GHz)
    if omega_rabi_GHz <= 0:
        return None, None
    t_pi_ns = 1.0 / (2.0 * omega_rabi_GHz)
    t_pi2_ns = 1.0 / (4.0 * omega_rabi_GHz)
    return t_pi_ns, t_pi2_ns


# ----------------------------
# Page
# ----------------------------
def page():
    _init_state()

    if st.session_state.get("user_data") is None:
        try:
            st.session_state["user_data"] = _load_params_from_secrets()
            _clear_results_and_pulses()
            st.rerun()
        except Exception as e:
            st.title("Qublitz Virtual Qubit Lab")
            st.error(f"Could not load qubit parameters from Streamlit secrets: {e}")
            st.stop()

    user_data = st.session_state["user_data"]

    try:
        st.sidebar.image(Image.open("images/qublitz.png"))
    except Exception:
        pass
    try:
        st.sidebar.image(Image.open("images/logo.png"))
    except Exception:
        pass

    st.sidebar.write("Loaded from deployment secrets")
    if st.sidebar.button("Reload secrets", key="reload_btn"):
        st.session_state["user_data"] = _load_params_from_secrets()
        _clear_results_and_pulses()
        st.rerun()

    omega_q = float(user_data["omega_q"])
    omega_rabi = float(user_data["omega_rabi"])
    T1_ns = float(user_data["T1"])
    T2_ns = 2.0 * T1_ns

    shots = st.sidebar.number_input("Shots", min_value=32, max_value=4096, value=256, step=32, key="shots")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Hidden qubit parameters")
    st.sidebar.caption("The qubit parameters are loaded from secrets and are intentionally not shown.")

    with st.sidebar.expander("Simulation convention"):
        st.caption("This app uses the hidden physical parameters directly: ΩR/2π is passed unchanged to the engine, T₁ is passed unchanged, and the engine enforces T₂ = 2T₁.")

    st.title("Custom Qubit Query")
    tab_freq, tab_time, tab_ramsey = st.tabs(["Frequency Domain", "Time Domain", "Ramsey Sequence"])

    with tab_freq:
        fd_start_default, fd_stop_default = 4.8, 5.2

        start_freq = st.number_input(r"Start $\omega_d/2\pi$ [GHz]", value=fd_start_default, step=0.01, format="%.6f", key="fd_start")
        stop_freq = st.number_input(r"Stop $\omega_d/2\pi$ [GHz]", value=fd_stop_default, step=0.01, format="%.6f", key="fd_stop")
        num_points = st.number_input("Number of frequencies", value=41, min_value=5, max_value=201, step=2, key="fd_n")
        spec_tfinal = st.number_input("Pulse duration [ns]", value=25.0, min_value=1.0, max_value=500.0, step=1.0, key="fd_tfinal")
        n_steps_fd = int(25 * float(spec_tfinal))

        if st.button("Run Frequency Sweep", key="fd_run"):
            try:
                results = _run_corrected_freq_sweep(
                    float(start_freq), float(stop_freq), int(num_points),
                    float(spec_tfinal), n_steps_fd, omega_q, omega_rabi,
                    T1_ns, int(shots),
                )

                prob_1_data = np.array(results["prob_1_time_series"], dtype=float)
                frequencies = np.array(results["frequencies"], dtype=float)
                time_list = np.array(results["time_list"], dtype=float)

                Z = _extract_Z(prob_1_data, frequencies, time_list)
                max_prob = np.max(Z, axis=0)
                avg_prob = np.mean(Z, axis=0)
                peak_idx = int(np.argmax(max_prob))
                peak_freq = float(frequencies[peak_idx])
                p1_cut = np.asarray(Z[:, peak_idx], dtype=float)

                st.session_state["freq_out"] = {
                    "frequencies": frequencies, "time_list": time_list, "Z": Z,
                    "max_prob": max_prob, "avg_prob": avg_prob,
                    "peak_idx": peak_idx, "peak_freq": peak_freq,
                }
                st.session_state["freq_peak_cut"] = {
                    "time_list": time_list, "p1_cut": p1_cut, "peak_freq": peak_freq,
                }
                st.session_state["freq_rabi_fit"] = _fit_rabi_trace(time_list, p1_cut)
            except Exception as e:
                st.error(f"Frequency sweep failed: {e}")
                st.exception(e)

        out = st.session_state.get("freq_out")
        if out is not None:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                subplot_titles=("Time-resolved P(|1⟩)", "Max P(|1⟩) over time"),
                row_heights=[0.75, 0.25],
            )

            fig.add_trace(
                go.Heatmap(
                    x=out["frequencies"],
                    y=out["time_list"],
                    z=out["Z"],
                    colorscale="Viridis",
                    colorbar=dict(title="P(|1⟩)", len=0.75, y=0.78, thickness=25),
                    hovertemplate="ωd=%{x:.6f} GHz<br>t=%{y:.2f} ns<br>P1=%{z:.3f}<extra></extra>",
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=out["frequencies"],
                    y=out["max_prob"],
                    mode="lines",
                    showlegend=False,
                    line=dict(width=3),
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=820,
                title_text="Frequency Domain Simulation Results",
                xaxis2_title="Drive frequency [GHz]",
                yaxis1_title="Time [ns]",
                yaxis2_title="Max P(|1⟩)",
                margin=dict(t=70, b=50, l=70, r=50),
            )
            st.plotly_chart(fig, use_container_width=True)

            df = pd.DataFrame({"omega_d_GHz": out["frequencies"], "max_prob_1": out["max_prob"], "avg_prob_1": out["avg_prob"]})
            st.download_button("Download sweep summary CSV", to_csv_bytes(df), "frequency_sweep_summary.csv", "text/csv", key="fd_dl")

        cut = st.session_state.get("freq_peak_cut")
        if cut is not None:
            freq_rabi_fit = st.session_state.get("freq_rabi_fit")
            fig_cut = go.Figure()
            fig_cut.add_trace(go.Scatter(
                x=cut["time_list"], y=cut["p1_cut"], mode="lines+markers",
                name="P(|1⟩) at peak frequency", line=dict(width=3), marker=dict(size=4)
            ))
            if freq_rabi_fit is not None:
                fig_cut.add_trace(go.Scatter(
                    x=cut["time_list"], y=freq_rabi_fit["fit_curve"], mode="lines",
                    name="Rabi fit", line=dict(width=3, dash="dash")
                ))
            fig_cut.update_layout(
                height=420,
                title=f"P(|1⟩) vs time at peak frequency (ωd = {cut['peak_freq']:.6f} GHz)",
                xaxis_title="Time [ns]", yaxis_title="P(|1⟩)",
                yaxis=dict(range=[-0.05, 1.05]),
                margin=dict(t=70, b=50, l=70, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cut, use_container_width=True)
            if freq_rabi_fit is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Fitted ΩR / 2π from peak-frequency trace", _format_pm(freq_rabi_fit.get("f_MHz"), freq_rabi_fit.get("f_MHz_err"), "MHz"))
                with c2:
                    st.caption("This fit is obtained directly from the oscillation period of the peak-frequency time trace, using the simulator's physical Rabi-rate convention.")
                st.latex(r"P_{1,\mathrm{Rabi}}(t) \approx C + A e^{-t/\tau_R} \cos\!\left(2\pi f_R t + \phi\right)")

    with tab_time:
        omega_d_default = 5.0
        omega_d = st.number_input(r"$\omega_d/2\pi$ [GHz]", value=omega_d_default, step=1e-6, format="%.9f", key="td_wd")
        t_final = st.number_input(r"Duration $\Delta t$ [ns]", value=200.0, min_value=1.0, max_value=2000.0, step=1.0, key="td_tfinal")
        _, sx_sched, sy_sched = pulse_ui(float(t_final))

        st.markdown("### Time-domain results")
        st.caption("Use this tab mainly to extract T₁ from the post-pulse decay. The Rabi-rate fit is now taken from the peak-frequency time trace in the Frequency Domain tab.")

        if st.session_state.get("td_out") is None:
            try:
                st.session_state["td_out"] = run_time_domain(
                    omega_q, omega_rabi, T1_ns, float(omega_d),
                    float(t_final), sx_sched, sy_sched, int(shots),
                )
            except Exception:
                st.session_state["td_out"] = None

        if st.button("Re-run simulation with current settings", key="td_run"):
            try:
                st.session_state["td_out"] = run_time_domain(
                    omega_q, omega_rabi, T1_ns, float(omega_d),
                    float(t_final), sx_sched, sy_sched, int(shots),
                )
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)

        out_td = st.session_state.get("td_out")

        if out_td is not None:
            fit_summary = _analyze_time_domain_fits(out_td["tlist"], out_td["p1_meas"], sx_sched, sy_sched)
            st.session_state["td_t1_fit"] = fit_summary["t1"]

            fig_results = go.Figure()
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_x_rot"], mode="lines", name="⟨σx⟩", line=dict(width=3)))
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_y_rot"], mode="lines", name="⟨σy⟩", line=dict(width=3)))
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_z"], mode="lines", name="⟨σz⟩", line=dict(width=3)))
            fig_results.update_layout(
                yaxis=dict(range=[-1.05, 1.05]),
                xaxis_title="Time [ns]", yaxis_title="Expectation values",
                title=f"Rotating-frame dynamics (ωd = {float(omega_d):.9f} GHz)",
                height=420, margin=dict(t=70, b=50, l=70, r=30),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_results, use_container_width=True)

            fig_bloch = bloch_sphere_pretty(
                out_td["exp_x_rot"], out_td["exp_y_rot"], out_td["exp_z"],
                out_td["tlist"], float(t_final),
            )
            st.plotly_chart(fig_bloch, use_container_width=True)

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["p1_meas"], mode="lines", name="Measured P(|1⟩)", line=dict(width=3)))

            t1_fit = fit_summary["t1"]
            if t1_fit is not None and fit_summary["t1_slice"] is not None:
                i0, i1 = fit_summary["t1_slice"]
                fig_p.add_trace(go.Scatter(
                    x=out_td["tlist"][i0:i1 + 1],
                    y=t1_fit["fit_curve"],
                    mode="lines",
                    name="T₁ fit",
                    line=dict(width=3, dash="dot"),
                ))

            fig_p.update_layout(
                xaxis_title="Time [ns]", yaxis=dict(range=[-0.05, 1.05]),
                yaxis_title="Measured P(|1⟩)",
                title=f"Measurement record (shots = {int(shots)})",
                height=420, margin=dict(t=70, b=50, l=70, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_p, use_container_width=True)
            if t1_fit is not None:
                st.metric("Fitted T₁", _format_pm(t1_fit.get("T1_ns"), t1_fit.get("T1_ns_err"), "ns"))
            else:
                st.info("T₁ fit unavailable for the current schedule.")

            for msg in fit_summary["messages"]:
                st.caption(msg)

            st.latex(r"P_{1,\mathrm{T1}}(t) \approx C + A e^{-t/T_1}")

            df = pd.DataFrame({
                "time_ns": out_td["tlist"],
                "exp_x_rot": out_td["exp_x_rot"],
                "exp_y_rot": out_td["exp_y_rot"],
                "exp_z": out_td["exp_z"],
                "p1_meas": out_td["p1_meas"],
            })
            if t1_fit is not None and fit_summary["t1_slice"] is not None:
                i0, i1 = fit_summary["t1_slice"]
                fit_full = np.full(len(df), np.nan)
                fit_full[i0:i1 + 1] = t1_fit["fit_curve"]
                df["t1_fit"] = fit_full
            st.download_button("Download time-domain CSV", to_csv_bytes(df), "time_domain_sim.csv", "text/csv", key="td_dl")
        else:
            st.info("Time-domain plots will appear once the simulator returns results.")

    with tab_ramsey:
        st.markdown("## Ramsey sequence")
        st.markdown(r"Use a $\pi/2 - \tau - \pi/2$ sequence to generate a Ramsey chevron in measured $P(|1\rangle)$.")
        st.latex(r"\pi/2_x \;\rightarrow\; \tau \;\rightarrow\; \pi/2_x \;\rightarrow\; \mathrm{measure}")
        st.latex(r"t_\pi = \frac{1}{2\,\Omega_R/2\pi}, \qquad t_{\pi/2} = \frac{1}{4\,\Omega_R/2\pi}")

        st.caption(
            "The colormap is built by sweeping the idle time τ and the relative detuning Δ. "
            "For each point, the code applies an ideal π/2 rotation, propagates the idle segment with T₁ and T₂, applies a second π/2 rotation, and records the final P(|1⟩)."
        )

        q1, q2 = st.columns([1, 1])
        with q1:
            ramsey_qubit_freq = st.number_input(
                r"Reference qubit frequency $\omega_q/2\pi$ [GHz]",
                value=float(omega_q),
                step=1e-6,
                format="%.9f",
                key="ramsey_qubit_freq_ref",
                help="This sets the reference frequency used to convert detuning into drive frequency.",
            )
        with q2:
            st.metric("Reference ωq / 2π", f"{float(ramsey_qubit_freq):.9f} GHz")

        source_options = ["Manual entry"]
        fitted_rabi = st.session_state.get("freq_rabi_fit")
        if fitted_rabi is not None and fitted_rabi.get("f_MHz") is not None:
            source_options.insert(0, "Use fitted Rabi rate from Frequency Domain")

        duration_source = st.radio("π/2 duration source", source_options, horizontal=False, key="ramsey_duration_source")

        manual_default = 2.5
        if fitted_rabi is not None and fitted_rabi.get("f_MHz"):
            _, fitted_pi2 = _pi_times_from_rabi(1e-3 * fitted_rabi["f_MHz"])
            if fitted_pi2 is not None:
                manual_default = float(fitted_pi2)

        if duration_source == "Use fitted Rabi rate from Frequency Domain":
            fitted_rabi_GHz = 1e-3 * float(fitted_rabi["f_MHz"])
            t_pi_ns, pi2_ns = _pi_times_from_rabi(fitted_rabi_GHz)
            st.metric("Rabi rate used for Ramsey", _format_pm(fitted_rabi.get("f_MHz"), fitted_rabi.get("f_MHz_err"), "MHz"))
            st.caption("Taken from the peak-frequency time trace in the Frequency Domain tab.")
            st.metric("Computed π/2 duration", f"{pi2_ns:.3f} ns")
        else:
            pi2_ns = st.number_input(
                "π/2 pulse duration [ns]",
                min_value=0.01,
                max_value=500.0,
                value=float(manual_default),
                step=0.01,
                format="%.3f",
                key="ramsey_pi2_manual",
            )
            t_pi_ns = 2.0 * float(pi2_ns)
            st.caption(f"Corresponding π duration: {t_pi_ns:.3f} ns")

        st.markdown("### Chevron sweep")
        r1, r2, r3 = st.columns(3)
        with r1:
            detuning_min = st.number_input("Detuning min [MHz]", value=-2.0, step=0.1, format="%.3f", key="ramsey_detuning_min")
        with r2:
            detuning_max = st.number_input("Detuning max [MHz]", value=2.0, step=0.1, format="%.3f", key="ramsey_detuning_max")
        with r3:
            n_detuning = st.number_input("Detuning points", value=81, min_value=11, max_value=401, step=2, key="ramsey_detuning_points")

        r4, r5, r6 = st.columns(3)
        with r4:
            delay_min = st.number_input("Idle-time min [ns]", value=0.0, min_value=0.0, step=10.0, key="ramsey_delay_min")
        with r5:
            delay_max = st.number_input("Idle-time max [ns]", value=float(max(500.0, 2.5 * T2_ns)), min_value=10.0, step=50.0, key="ramsey_delay_max")
        with r6:
            n_delay = st.number_input("Idle-time points", value=121, min_value=11, max_value=501, step=2, key="ramsey_delay_points")

        ramsey_shots = st.number_input("Shots per point", value=256, min_value=32, max_value=4096, step=32, key="ramsey_shots_simple")
        st.caption(f"Total Ramsey points: {int(n_detuning) * int(n_delay)}")

        if st.button("Run Ramsey Chevron", key="ramsey_chevron_run_simple"):
            detuning_list = np.linspace(float(detuning_min), float(detuning_max), int(n_detuning))
            delay_list = np.linspace(float(delay_min), float(delay_max), int(n_delay))
            drive_freq_list = float(ramsey_qubit_freq) + 1e-3 * detuning_list

            prog = st.progress(0, text="Running Ramsey chevron...")
            Z = _ramsey_chevron(
                delay_list_ns=delay_list,
                detuning_list_MHz=detuning_list,
                T1_ns=float(T1_ns),
                T2_ns=float(T2_ns),
                shots=int(ramsey_shots),
                progress_bar=prog,
            )

            st.session_state["ramsey_chevron_out"] = {
                "detuning_list_MHz": detuning_list,
                "drive_freq_list_GHz": drive_freq_list,
                "qubit_freq_GHz": float(ramsey_qubit_freq),
                "delay_list_ns": delay_list,
                "Z": Z,
                "pi2_ns": float(pi2_ns),
                "shots": int(ramsey_shots),
            }

        chev = st.session_state.get("ramsey_chevron_out")
        if chev is not None:
            fig_chev = go.Figure()
            fig_chev.add_trace(go.Heatmap(
                x=chev["detuning_list_MHz"],
                y=chev["delay_list_ns"],
                z=chev["Z"],
                colorscale="Inferno",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="P(|1⟩)", thickness=25),
                customdata=np.tile(np.asarray(chev["drive_freq_list_GHz"])[None, :], (len(chev["delay_list_ns"]), 1)),
                hovertemplate="Δ=%{x:.3f} MHz<br>ωd/2π=%{customdata:.9f} GHz<br>τ=%{y:.1f} ns<br>P1=%{z:.3f}<extra></extra>",
            ))
            fig_chev.update_layout(
                height=650,
                title=f"Ramsey chevron, π/2 = {chev['pi2_ns']:.3f} ns",
                xaxis_title="Relative detuning Δ/2π [MHz]",
                yaxis_title="Idle time τ [ns]",
                margin=dict(t=70, b=60, l=70, r=40),
            )
            st.plotly_chart(fig_chev, use_container_width=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Reference ωq / 2π", f"{float(chev['qubit_freq_GHz']):.9f} GHz")
            with m2:
                st.caption("Drive frequency is computed as ωd/2π = ωq/2π + Δ/2π.")

            st.markdown("### Fit a vertical cut")
            chosen_detuning = st.number_input(
                "Detuning for vertical cut [MHz]",
                value=0.500,
                step=0.1,
                format="%.3f",
                key="ramsey_cut_detuning",
            )
            nearest_idx = int(np.argmin(np.abs(chev["detuning_list_MHz"] - float(chosen_detuning))))
            nearest_det = float(chev["detuning_list_MHz"][nearest_idx])
            nearest_drive = float(chev["drive_freq_list_GHz"][nearest_idx])
            p1_cut = np.asarray(chev["Z"][:, nearest_idx], dtype=float)
            fit_cut = _fit_ramsey_trace(np.asarray(chev["delay_list_ns"]), p1_cut, freq_guess_MHz=abs(nearest_det))

            info1, info2 = st.columns(2)
            with info1:
                st.metric("Selected detuning", f"{nearest_det:+.3f} MHz")
            with info2:
                st.metric("Corresponding ωd / 2π", f"{nearest_drive:.9f} GHz")

            fig_cut = go.Figure()
            fig_cut.add_trace(go.Scatter(
                x=chev["delay_list_ns"],
                y=p1_cut,
                mode="lines+markers",
                name=f"Ramsey data (Δ = {nearest_det:+.3f} MHz)",
                line=dict(width=3),
                marker=dict(size=4),
            ))
            if fit_cut is not None:
                fig_cut.add_trace(go.Scatter(
                    x=chev["delay_list_ns"],
                    y=fit_cut["fit_curve"],
                    mode="lines",
                    name="Ramsey fit",
                    line=dict(width=3, dash="dash"),
                ))
                
            fig_cut.update_layout(
                height=450,
                title="Ramsey trace from a vertical cut of the chevron",
                xaxis_title="Idle time τ [ns]",
                yaxis_title="P(|1⟩)",
                yaxis=dict(range=[-0.05, 1.05]),
                margin=dict(t=70, b=60, l=70, r=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cut, use_container_width=True)

            if fit_cut is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Fitted T₂", _format_pm(fit_cut.get("T2_ns"), fit_cut.get("T2_ns_err"), "ns"))
                with c2:
                    st.metric("Fitted Ramsey fringe frequency", _format_pm(fit_cut.get("freq_MHz"), fit_cut.get("freq_MHz_err"), "MHz"))
            else:
                st.info("Ramsey fit unavailable for this vertical cut.")

            rows = []
            for i, tau in enumerate(chev["delay_list_ns"]):
                for j, det in enumerate(chev["detuning_list_MHz"]):
                    rows.append({
                        "qubit_freq_GHz": float(chev["qubit_freq_GHz"]),
                        "drive_freq_GHz": float(chev["drive_freq_list_GHz"][j]),
                        "detuning_MHz": float(det),
                        "delay_ns": float(tau),
                        "p1": float(chev["Z"][i, j]),
                    })
            st.download_button(
                "Download chevron CSV",
                to_csv_bytes(pd.DataFrame(rows)),
                "ramsey_chevron_p1.csv",
                "text/csv",
                key="ramsey_chevron_download_simple",
            )

            st.markdown("---")
            st.markdown("## Reveal Parameters")

            reveal = st.toggle("Show hidden simulator parameters", value=False, key="ramsey_reveal_params")

            if reveal:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(r"Qubit frequency:  $\omega_q / 2\pi$", f"{omega_q:.6f} GHz")
                with c2:
                    st.metric(r"Rabi rate:  $\Omega_R / 2\pi$", f"{omega_rabi * 1e3:.3f} MHz")
                with c3:
                    st.metric(r"$T_1 $", f"{T1_ns:.2f} ns")
                with c4:
                    st.metric(r"$T_2 = 2 T_1 $", f"{T2_ns:.2f} ns")
            else:
                st.caption("Toggle to reveal the hidden simulator parameters.")
page()
