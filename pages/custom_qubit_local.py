#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local (API-based) version of the Qublitz app.

Login resolves the student's parameters from the on-premise API.

================================================================================
FACTOR-OF-2 FIX  (see custom_qubit_deployed.py for detailed derivation)
================================================================================
Engine conventions:
  - H_drive = Ω_eng · env(t) · σ_x cos(ω_d t)   (no 1/2 prefactor)
      → after RWA, observed Rabi freq = Ω_eng
      → to match assigned Ω_phys: Ω_eng = Ω_phys / 2

  - Lindblad decay:  T1_obs = T1_eng / 2
      → to match assigned T1:  T1_eng = 2 · T1_phys

  - T2 = 2·T1 (hardcoded, pure-dephasing-free limit)
      → T2_eng = 2 · T2_phys = 4 · T1_phys
================================================================================
"""

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantum_simulator import run_quantum_simulation, run_frequency_sweep

# ----------------------------
# Assignment resolution config
# ----------------------------
API_BASE_URL = (
    os.environ.get("QUBLITZ_API_BASE_URL", "http://10.28.54.127:8001")
    .strip()
    .rstrip("/")
)

# ----------------------------
# Debug student
# ----------------------------
INSTRUCTOR_DEBUG = os.environ.get(
    "QUBLITZ_INSTRUCTOR_DEBUG", "0"
).strip().lower() in ("1", "true", "yes")

TEST_STUDENT_API_KEY = "b8e60fc199646f8e712948304a65d52cd43b9bc3"
TEST_STUDENT = {
    "user": "debug_test_student",
    "omega_q": 4.671431666715805,       # GHz  (physical)
    "omega_rabi": 0.20214882218103472,  # GHz  (physical)
    "T1": 77.88719688437668,            # ns   (physical)
}


# ============================================================
# Engine-convention helpers
# ============================================================
def _engine_rabi(omega_rabi_physical_GHz: float) -> float:
    return omega_rabi_physical_GHz / 2.0


def _engine_T1(T1_physical_ns: float) -> float:
    return 2.0 * T1_physical_ns


def _engine_T2(T1_physical_ns: float) -> float:
    return 4.0 * T1_physical_ns


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
        "ramsey_out", "ramsey_chevron_out",
    ):
        st.session_state[key] = None


def _init_state():
    for key in (
        "api_key", "user_data",
        "freq_out", "freq_peak_cut",
        "td_out", "td_tfinal_last", "td_sigma_x_vec", "td_sigma_y_vec",
        "ramsey_out", "ramsey_chevron_out",
    ):
        st.session_state.setdefault(key, None)


def _try_backend_login(api_key: str) -> Dict[str, Any]:
    if not API_BASE_URL:
        raise RuntimeError("QUBLITZ_API_BASE_URL is empty.")
    try:
        r = requests.get(
            f"{API_BASE_URL}/me",
            headers={"X-API-Key": api_key},
            timeout=8,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            raise RuntimeError(f"Login rejected ({r.status_code}): {r.text}")
    except requests.RequestException:
        pass
    r = requests.get(f"{API_BASE_URL}/me/{api_key}", timeout=8)
    if r.status_code != 200:
        raise RuntimeError(f"Login rejected ({r.status_code}): {r.text}")
    return r.json()


def _vpn_check_hint():
    st.info(
        "If login fails:\n"
        "- Make sure you are connected to the Dartmouth VPN.\n"
        "- Check that the assignment server is reachable.\n"
        "- If needed, set `QUBLITZ_API_BASE_URL` to the correct URL."
    )


# ----------------------------
# Bloch sphere
# ----------------------------
def bloch_sphere_pretty(exp_x, exp_y, exp_z, tlist, t_final):
    exp_x = np.asarray(exp_x, dtype=float)
    exp_y = np.asarray(exp_y, dtype=float)
    exp_z = np.asarray(exp_z, dtype=float)
    tlist = np.asarray(tlist, dtype=float)

    exp_z_plot = -exp_z  # flip so |0⟩ is top

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
                    title="Time [ns]", len=0.85, y=0.5, thickness=14,
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

    col1, col2 = st.columns([1, 1])
    with col1:
        target = st.selectbox("Target channel", ["σ_x", "σ_y"], key="td_target")
        ptype = st.selectbox("Pulse type", ["Square", "Gaussian"], key="td_ptype")
        amp = st.slider("Amplitude", -1.0, 1.0, 1.0, step=0.05, key="td_amp")

        if ptype == "Square":
            start = st.slider("Start [ns]", 0.0, float(max(0.0, t_final_ns - 1.0)), 5.0, step=1.0, key="td_sq_start")
            stop = st.slider("Stop [ns]", float(start), float(t_final_ns), float(min(t_final_ns, start + 20.0)), step=1.0, key="td_sq_stop")
        else:
            center = st.slider("Center [ns]", 0.0, float(t_final_ns), float(min(t_final_ns * 0.5, 50.0)), step=1.0, key="td_g_center")
            sigma = st.slider("Sigma [ns]", 1.0, float(max(2.0, t_final_ns / 2.0)), float(max(2.0, min(10.0, t_final_ns / 10.0))), step=1.0, key="td_g_sigma")

        add_btn = st.button("Add pulse", key="td_add")
        clear_btn = st.button("Clear schedule", key="td_clear")

    with col2:
        st.caption("You can stack pulses by adding multiple pulses sequentially.")

    if clear_btn:
        st.session_state["td_sigma_x_vec"] = np.zeros(n_steps, dtype=float)
        st.session_state["td_sigma_y_vec"] = np.zeros(n_steps, dtype=float)

    if add_btn:
        env = np.zeros(n_steps, dtype=float)
        if ptype == "Square":
            env[(tlist >= float(start)) & (tlist <= float(stop))] = float(amp)
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
# Ramsey / Echo engine
# ----------------------------
def _build_ramsey_schedule(tlist, pi2_dur, delay, echo=False):
    sx = np.zeros_like(tlist)

    t0, t1 = 0.0, pi2_dur
    sx[(tlist >= t0) & (tlist < t1)] = 1.0

    if echo:
        half_tau = delay / 2.0
        pi_dur = 2.0 * pi2_dur
        t2 = t1 + half_tau
        t3 = t2 + pi_dur
        sx[(tlist >= t2) & (tlist < t3)] = 1.0
        t4 = t3 + half_tau
        t5 = t4 + pi2_dur
        sx[(tlist >= t4) & (tlist < t5)] = 1.0
    else:
        t2 = t1 + delay
        t3 = t2 + pi2_dur
        sx[(tlist >= t2) & (tlist < t3)] = 1.0

    return _clip_env(sx)


def _ramsey_total_time(pi2_dur, delay, echo):
    if echo:
        return pi2_dur + delay / 2.0 + 2.0 * pi2_dur + delay / 2.0 + pi2_dur
    else:
        return pi2_dur + delay + pi2_dur


def run_ramsey_point(omega_q_GHz, omega_rabi_GHz, T1_ns, omega_d_GHz,
                     delay_ns, shots, pi2_dur_ns, echo=False):
    total = max(1.0, _ramsey_total_time(pi2_dur_ns, delay_ns, echo))
    tlist, n_steps = _time_grid(total, pts_per_ns=25)

    sx_env = _build_ramsey_schedule(tlist, pi2_dur_ns, delay_ns, echo)
    sy_env = np.zeros(n_steps, dtype=float)

    rabi_eng = _engine_rabi(float(omega_rabi_GHz))
    T1_eng = _engine_T1(float(T1_ns))
    T2_eng = _engine_T2(float(T1_ns))

    exp_values, _, sampled_prob = run_quantum_simulation(
        float(omega_q_GHz), rabi_eng, float(total), int(n_steps),
        float(omega_d_GHz), sx_env, sy_env, int(shots), T1_eng, T2_eng,
    )

    p1_final = float(sampled_prob[-1]) if len(sampled_prob) > 0 else 0.0
    return p1_final, total, tlist, sx_env, exp_values, sampled_prob


def run_ramsey_sweep(omega_q_GHz, omega_rabi_GHz, T1_ns, omega_d_GHz,
                     delays, shots, pi2_dur_ns, echo=False, progress_bar=None):
    p1_list = []
    n = len(delays)
    for i, tau in enumerate(delays):
        p1, *_ = run_ramsey_point(
            omega_q_GHz, omega_rabi_GHz, T1_ns, omega_d_GHz,
            float(tau), shots, pi2_dur_ns, echo=echo,
        )
        p1_list.append(p1)
        if progress_bar is not None:
            progress_bar.progress((i + 1) / n)
    return np.array(p1_list, dtype=float)


def run_ramsey_chevron(omega_q_GHz, omega_rabi_GHz, T1_ns, freq_list,
                       delay_list, shots, pi2_dur_ns, echo=False,
                       progress_bar=None):
    n_freq = len(freq_list)
    n_delay = len(delay_list)
    total_pts = n_freq * n_delay
    Z = np.zeros((n_delay, n_freq), dtype=float)

    count = 0
    for j, fd in enumerate(freq_list):
        for i, tau in enumerate(delay_list):
            p1, *_ = run_ramsey_point(
                omega_q_GHz, omega_rabi_GHz, T1_ns, float(fd),
                float(tau), shots, pi2_dur_ns, echo=echo,
            )
            Z[i, j] = p1
            count += 1
            if progress_bar is not None:
                progress_bar.progress(count / total_pts)
    return Z


# ----------------------------
# Page
# ----------------------------
def page():
    _init_state()

    # ── Login ──────────────────────────────────────────────────────
    if st.session_state.get("user_data") is None:
        st.title("Qublitz Virtual Qubit Lab (Local)")
        st.caption("Enter your API key (NetID).")

        default_val = TEST_STUDENT_API_KEY if INSTRUCTOR_DEBUG else ""
        api_key = st.text_input("API Key", type="password", value=default_val, key="login_api_key")

        if st.button("Login", key="login_btn"):
            try:
                key = (api_key or "").strip().lower()
                if not key:
                    raise ValueError("Empty API key.")
                if INSTRUCTOR_DEBUG and key == TEST_STUDENT_API_KEY:
                    user_data = dict(TEST_STUDENT)
                else:
                    user_data = _try_backend_login(key)

                st.session_state["api_key"] = key
                st.session_state["user_data"] = user_data
                _clear_results_and_pulses()
                st.success(f"Welcome {user_data.get('user', 'student')}!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
                _vpn_check_hint()
        return

    user_data = st.session_state["user_data"]

    # ── Sidebar ────────────────────────────────────────────────────
    try:
        st.sidebar.image(Image.open("images/qublitz.png"))
    except Exception:
        pass
    try:
        st.sidebar.image(Image.open("images/logo.png"))
    except Exception:
        pass

    st.sidebar.write(f"Logged in as: **{user_data.get('user', 'student')}**")
    if st.sidebar.button("Logout", key="logout_btn"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k, None)
        st.rerun()

    omega_q = float(user_data["omega_q"])
    omega_rabi = float(user_data["omega_rabi"])
    T1_ns = float(user_data["T1"])
    T2_ns = 2.0 * T1_ns

    shots = st.sidebar.number_input("Shots", min_value=32, max_value=4096, value=256, step=32, key="shots")
    is_debug_user = st.session_state.get("api_key", "") == TEST_STUDENT_API_KEY

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Your qubit parameters")
    st.sidebar.caption("*(Physical values you should recover from calibration)*")
    if is_debug_user:
        st.sidebar.markdown(f"- ω_q/2π = **{omega_q:.9f}** GHz")
        st.sidebar.markdown(f"- Ω_R/2π = **{omega_rabi * 1e3:.3f}** MHz")
        st.sidebar.markdown(f"- T₁ = **{T1_ns:.2f}** ns")
        st.sidebar.markdown(f"- T₂ = **{T2_ns:.2f}** ns  (= 2·T₁)")

    st.header("Custom Qubit Query")
    tab_freq, tab_time, tab_ramsey = st.tabs(
        ["Frequency Domain", "Time Domain", "Ramsey / Echo"]
    )

    # ================================================================
    #  TAB 1 — FREQUENCY DOMAIN
    # ================================================================
    with tab_freq:
        if is_debug_user:
            fd_start_default = float(omega_q - 0.25)
            fd_stop_default = float(omega_q + 0.25)
        else:
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
            except Exception as e:
                st.error(f"Frequency sweep failed: {e}")
                st.exception(e)

        out = st.session_state.get("freq_out")
        if out is not None:
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                subplot_titles=("Time-resolved P(|1⟩)", "Max P(|1⟩) over time", "Avg P(|1⟩) over time"),
                row_heights=[0.62, 0.20, 0.18],
            )
            fig.add_trace(go.Heatmap(
                x=out["frequencies"], y=out["time_list"], z=out["Z"],
                colorscale="Viridis",
                colorbar=dict(title="P(|1⟩)", len=0.62, y=0.79, thickness=16),
                hovertemplate="ωd=%{x:.6f} GHz<br>t=%{y:.2f} ns<br>P1=%{z:.3f}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=out["frequencies"], y=out["max_prob"], mode="lines", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=out["frequencies"], y=out["avg_prob"], mode="lines", showlegend=False), row=3, col=1)
            fig.update_layout(
                height=980, title_text="Frequency Domain Simulation Results",
                xaxis3_title="Drive frequency [GHz]",
                yaxis1_title="Time [ns]", yaxis2_title="Max P(|1⟩)", yaxis3_title="Avg P(|1⟩)",
                margin=dict(t=70, b=50, l=70, r=50),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Peak response near ωd ≈ {out['peak_freq']:.6f} GHz.")

            df = pd.DataFrame({"omega_d_GHz": out["frequencies"], "max_prob_1": out["max_prob"], "avg_prob_1": out["avg_prob"]})
            st.download_button("Download sweep summary CSV", to_csv_bytes(df), "frequency_sweep_summary.csv", "text/csv", key="fd_dl")

        cut = st.session_state.get("freq_peak_cut")
        if cut is not None:
            fig_cut = go.Figure()
            fig_cut.add_trace(go.Scatter(x=cut["time_list"], y=cut["p1_cut"], mode="lines", line=dict(width=3)))
            fig_cut.update_layout(
                height=360,
                title=f"P(|1⟩) vs time at peak frequency (ωd = {cut['peak_freq']:.6f} GHz)",
                xaxis_title="Time [ns]", yaxis_title="P(|1⟩)",
                yaxis=dict(range=[-0.05, 1.05]),
                margin=dict(t=70, b=50, l=70, r=30),
            )
            st.plotly_chart(fig_cut, use_container_width=True)

    # ================================================================
    #  TAB 2 — TIME DOMAIN
    # ================================================================
    with tab_time:
        omega_d_default = float(omega_q) if is_debug_user else 5.0
        omega_d = st.number_input(r"$\omega_d/2\pi$ [GHz]", value=omega_d_default, step=1e-6, format="%.9f", key="td_wd")
        t_final = st.number_input(r"Duration $\Delta t$ [ns]", value=200.0, min_value=1.0, max_value=2000.0, step=1.0, key="td_tfinal")
        tlist_td, sx_sched, sy_sched = pulse_ui(float(t_final))

        st.markdown("### Time-domain results")

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
            fig_results = go.Figure()
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_x_rot"], mode="lines", name="⟨σx⟩", line=dict(width=3)))
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_y_rot"], mode="lines", name="⟨σy⟩", line=dict(width=3)))
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_z"], mode="lines", name="⟨σz⟩", line=dict(width=3)))
            fig_results.update_layout(
                yaxis=dict(range=[-1.05, 1.05]),
                xaxis_title="Time [ns]", yaxis_title="Expectation values",
                title=f"Rotating-frame dynamics (ωd={float(omega_d):.9f} GHz)",
                height=420, margin=dict(t=70, b=50, l=70, r=30),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_results, use_container_width=True)
        else:
            st.info("Time-domain plots will appear once the simulator returns results.")

        if out_td is not None:
            fig_bloch = bloch_sphere_pretty(
                out_td["exp_x_rot"], out_td["exp_y_rot"], out_td["exp_z"],
                out_td["tlist"], float(t_final),
            )
            st.plotly_chart(fig_bloch, use_container_width=True)

        if out_td is not None:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["p1_meas"], mode="lines", line=dict(width=3)))
            fig_p.update_layout(
                xaxis_title="Time [ns]", yaxis=dict(range=[-0.05, 1.05]),
                yaxis_title="Measured P(|1⟩)",
                title=f"Measurement record (shots={int(shots)})",
                height=360, margin=dict(t=70, b=50, l=70, r=30),
            )
            st.plotly_chart(fig_p, use_container_width=True)

            df = pd.DataFrame({
                "time_ns": out_td["tlist"], "exp_x_rot": out_td["exp_x_rot"],
                "exp_y_rot": out_td["exp_y_rot"], "exp_z": out_td["exp_z"],
                "p1_meas": out_td["p1_meas"],
            })
            st.download_button("Download time-domain CSV", to_csv_bytes(df), "time_domain_sim.csv", "text/csv", key="td_dl")

    # ================================================================
    #  TAB 3 — RAMSEY / ECHO
    # ================================================================
    with tab_ramsey:
        st.markdown("""
## Ramsey & Echo Experiments

### Goal
Extract **T₂\\*** (Ramsey) and **T₂** (Hahn echo) for your qubit.

### Prerequisites
1. Use the **Frequency Domain** tab to find **ω_q**.
2. Use the **Time Domain** tab to calibrate a **π-pulse**:
   drive on resonance (ω_d = ω_q) with a square pulse on σ_x
   and find the pulse duration where ⟨σ_z⟩ reaches −1 (full inversion).
3. Your **π/2 pulse duration** is half of that.

### Pulse sequences

| Ramsey | Echo (Hahn) |
|--------|-------------|
| π/2 — τ — π/2 — measure | π/2 — τ/2 — π — τ/2 — π/2 — measure |

All pulses are square, amplitude = 1, on the σ_x channel.

### Ramsey-Chevron pattern
The Ramsey-Chevron is a 2-D map of P(|1⟩) vs **(ω_d, τ)**.
At the correct ω_q you see the slowest fringe (longest period).
Away from resonance the fringes speed up with detuning Δ = ω_d − ω_q.
This is a powerful visual tool for precisely identifying ω_q and T₂*.
""")

        st.markdown("---")

        st.markdown("### Ramsey sweep settings")

        col_a, col_b = st.columns(2)
        with col_a:
            ramsey_wd = st.number_input(
                r"Drive frequency $\omega_d/2\pi$ [GHz]",
                value=float(omega_q) if is_debug_user else 5.0,
                step=1e-6, format="%.9f", key="ramsey_wd",
                help="On resonance → pure decay. Add a few MHz detuning to see fringes.",
            )
            pi2_dur = st.number_input(
                "π/2 pulse duration [ns]",
                value=2.5, min_value=0.1, max_value=500.0, step=0.1,
                format="%.2f", key="ramsey_pi2",
                help="Half your calibrated π-pulse duration.",
            )
        with col_b:
            ramsey_shots = st.number_input(
                "Shots per point", value=int(shots),
                min_value=32, max_value=4096, step=32, key="ramsey_shots",
            )
            do_echo = st.checkbox("Also run Hahn echo", value=True, key="ramsey_do_echo")

        col_c, col_d, col_e = st.columns(3)
        with col_c:
            delay_start = st.number_input("Start delay τ [ns]", value=0.0, min_value=0.0, step=1.0, key="ramsey_d0")
        with col_d:
            delay_stop = st.number_input("Stop delay τ [ns]", value=float(min(600.0, 6.0 * T1_ns)), min_value=1.0, step=10.0, key="ramsey_d1")
        with col_e:
            n_delays = st.number_input("Number of delay points", value=61, min_value=5, max_value=301, step=2, key="ramsey_nd")

        if st.button("Run Ramsey Sweep (1-D)", key="ramsey_run"):
            delay_list = np.linspace(float(delay_start), float(delay_stop), int(n_delays))

            prog = st.progress(0, text="Running Ramsey...")
            p1_ramsey = run_ramsey_sweep(
                omega_q, omega_rabi, T1_ns, float(ramsey_wd),
                delay_list, int(ramsey_shots), float(pi2_dur),
                echo=False, progress_bar=prog,
            )
            p1_echo = None
            if do_echo:
                prog2 = st.progress(0, text="Running Echo...")
                p1_echo = run_ramsey_sweep(
                    omega_q, omega_rabi, T1_ns, float(ramsey_wd),
                    delay_list, int(ramsey_shots), float(pi2_dur),
                    echo=True, progress_bar=prog2,
                )

            st.session_state["ramsey_out"] = {
                "delay_list": delay_list, "p1_ramsey": p1_ramsey, "p1_echo": p1_echo,
                "omega_d": float(ramsey_wd), "pi2_dur": float(pi2_dur),
                "detuning_MHz": (float(ramsey_wd) - omega_q) * 1e3,
            }

        r_out = st.session_state.get("ramsey_out")
        if r_out is not None:
            st.markdown("### 1-D Ramsey Results")
            det = r_out["detuning_MHz"]
            st.info(f"Detuning  Δ = ω_d − ω_q = {det:+.3f} MHz")

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=r_out["delay_list"], y=r_out["p1_ramsey"],
                mode="lines+markers", name="Ramsey (T₂*)",
                line=dict(width=3), marker=dict(size=4),
            ))
            if r_out["p1_echo"] is not None:
                fig_r.add_trace(go.Scatter(
                    x=r_out["delay_list"], y=r_out["p1_echo"],
                    mode="lines+markers", name="Hahn echo (T₂)",
                    line=dict(width=3, dash="dash"), marker=dict(size=4),
                ))

            fig_r.add_vline(x=T1_ns, line_width=2, line_dash="dot", line_color="red",
                            annotation_text=f"T₁ = {T1_ns:.1f} ns", annotation_position="top left")
            fig_r.add_vline(x=T2_ns, line_width=2, line_dash="dot", line_color="gray",
                            annotation_text=f"T₂ = 2T₁ = {T2_ns:.1f} ns", annotation_position="top right")

            fig_r.update_layout(
                xaxis_title="Delay τ [ns]", yaxis_title="P(|1⟩)",
                yaxis=dict(range=[-0.05, 1.05]),
                title=f"Ramsey experiment  (π/2 = {r_out['pi2_dur']:.2f} ns,  Δ = {det:+.3f} MHz)",
                height=500, margin=dict(t=70, b=50, l=70, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_r, use_container_width=True)

            # Example pulse sequence
            mid_idx = len(r_out["delay_list"]) // 4
            tau_ex = float(r_out["delay_list"][mid_idx])
            _, _, ex_t, ex_sx, _, ex_samp = run_ramsey_point(
                omega_q, omega_rabi, T1_ns, r_out["omega_d"],
                tau_ex, int(ramsey_shots), r_out["pi2_dur"], echo=False,
            )
            fig_ex = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                subplot_titles=(f"σ_x envelope  (τ = {tau_ex:.1f} ns)", "P(|1⟩) vs time"),
            )
            fig_ex.add_trace(go.Scatter(x=ex_t, y=ex_sx, mode="lines", line=dict(width=3, color="steelblue")), row=1, col=1)
            fig_ex.add_trace(go.Scatter(x=ex_t, y=np.asarray(ex_samp, dtype=float), mode="lines", line=dict(width=3, color="firebrick")), row=2, col=1)
            fig_ex.update_layout(
                height=460, margin=dict(t=70, b=50, l=70, r=30),
                yaxis1=dict(title="Amplitude", range=[-0.1, 1.1]),
                yaxis2=dict(title="P(|1⟩)", range=[-0.05, 1.05]),
                xaxis2_title="Time [ns]", showlegend=False,
            )
            st.plotly_chart(fig_ex, use_container_width=True)

            dl = {"delay_ns": r_out["delay_list"], "p1_ramsey": r_out["p1_ramsey"]}
            if r_out["p1_echo"] is not None:
                dl["p1_echo"] = r_out["p1_echo"]
            st.download_button("Download Ramsey CSV", to_csv_bytes(pd.DataFrame(dl)),
                               "ramsey_data.csv", "text/csv", key="ramsey_dl")

            st.markdown(r"""
**Fitting guide:**

| Experiment | Model | Key parameter |
|------------|-------|---------------|
| Ramsey | $P(|1\rangle) = A\,e^{-\tau/T_2^*}\cos(2\pi\Delta\,\tau + \phi) + B$ | $T_2^*$ |
| Hahn echo | $P(|1\rangle) = A\,e^{-\tau/T_2} + B$ | $T_2$ |

At zero detuning the Ramsey fringe has no oscillation — add a small Δ (few MHz) to see fringes.
For this simulator T₂ = 2T₁ (no pure dephasing).
""")

        # ── Ramsey-Chevron ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Ramsey-Chevron pattern (2-D)")
        st.markdown(
            "Sweep **both** drive frequency and delay time to produce a 2-D "
            "chevron pattern.  The chevron vertex sits at ω_q."
        )

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            chev_f_start = st.number_input("Start freq [GHz]",
                value=float(omega_q - 0.05) if is_debug_user else 4.95,
                step=0.001, format="%.6f", key="chev_f0")
        with col_f2:
            chev_f_stop = st.number_input("Stop freq [GHz]",
                value=float(omega_q + 0.05) if is_debug_user else 5.05,
                step=0.001, format="%.6f", key="chev_f1")
        with col_f3:
            chev_nf = st.number_input("Freq points", value=21, min_value=5, max_value=101, step=2, key="chev_nf")

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            chev_d_start = st.number_input("Start delay [ns]", value=0.0, min_value=0.0, step=1.0, key="chev_d0")
        with col_g2:
            chev_d_stop = st.number_input("Stop delay [ns]",
                value=float(min(200.0, 3.0 * T1_ns)),
                min_value=1.0, step=10.0, key="chev_d1")
        with col_g3:
            chev_nd = st.number_input("Delay points", value=31, min_value=5, max_value=101, step=2, key="chev_nd")

        chev_pi2 = st.number_input("π/2 pulse duration [ns] (chevron)",
            value=float(pi2_dur) if "ramsey_pi2" in st.session_state else 2.5,
            min_value=0.1, max_value=500.0, step=0.1, format="%.2f", key="chev_pi2")
        chev_shots = st.number_input("Shots per point (chevron)",
            value=128, min_value=32, max_value=4096, step=32, key="chev_shots")
        chev_echo = st.checkbox("Use echo sequence for chevron", value=False, key="chev_echo")

        total_chev_pts = int(chev_nf) * int(chev_nd)
        st.caption(f"Total simulations: {total_chev_pts}")

        if st.button("Run Ramsey-Chevron", key="chev_run"):
            freq_list = np.linspace(float(chev_f_start), float(chev_f_stop), int(chev_nf))
            delay_list = np.linspace(float(chev_d_start), float(chev_d_stop), int(chev_nd))

            prog_c = st.progress(0, text="Running chevron...")
            Z_chev = run_ramsey_chevron(
                omega_q, omega_rabi, T1_ns,
                freq_list, delay_list,
                int(chev_shots), float(chev_pi2),
                echo=chev_echo, progress_bar=prog_c,
            )
            st.session_state["ramsey_chevron_out"] = {
                "freq_list": freq_list, "delay_list": delay_list,
                "Z": Z_chev, "echo": chev_echo, "pi2_dur": float(chev_pi2),
            }

        chev_out = st.session_state.get("ramsey_chevron_out")
        if chev_out is not None:
            st.markdown("### Ramsey-Chevron Results")
            mode_label = "Echo" if chev_out["echo"] else "Ramsey"

            fig_chev = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                subplot_titles=(f"{mode_label}-Chevron  P(|1⟩)", "Max P(|1⟩) vs frequency"),
                row_heights=[0.75, 0.25],
            )
            fig_chev.add_trace(go.Heatmap(
                x=chev_out["freq_list"], y=chev_out["delay_list"], z=chev_out["Z"],
                colorscale="RdBu_r", zmin=0, zmax=1,
                colorbar=dict(title="P(|1⟩)", len=0.70, y=0.72, thickness=16),
                hovertemplate="ωd=%{x:.6f} GHz<br>τ=%{y:.1f} ns<br>P1=%{z:.3f}<extra></extra>",
            ), row=1, col=1)

            max_over_delay = np.max(chev_out["Z"], axis=0)
            fig_chev.add_trace(go.Scatter(
                x=chev_out["freq_list"], y=max_over_delay,
                mode="lines", showlegend=False, line=dict(width=2),
            ), row=2, col=1)

            fig_chev.update_layout(
                height=780, yaxis1_title="Delay τ [ns]",
                xaxis2_title="Drive frequency ω_d/2π [GHz]",
                yaxis2_title="Max P(|1⟩)",
                margin=dict(t=70, b=50, l=70, r=50),
            )
            st.plotly_chart(fig_chev, use_container_width=True)

            peak_f_idx = int(np.argmax(max_over_delay))
            peak_f = float(chev_out["freq_list"][peak_f_idx])
            st.success(f"Chevron vertex (peak response) near ω_d/2π ≈ **{peak_f:.6f} GHz**.")

            p1_at_peak = chev_out["Z"][:, peak_f_idx]
            fig_vcut = go.Figure()
            fig_vcut.add_trace(go.Scatter(x=chev_out["delay_list"], y=p1_at_peak, mode="lines", line=dict(width=3)))
            fig_vcut.update_layout(
                height=360, title=f"P(|1⟩) vs τ at ω_d = {peak_f:.6f} GHz",
                xaxis_title="Delay τ [ns]", yaxis_title="P(|1⟩)",
                yaxis=dict(range=[-0.05, 1.05]),
                margin=dict(t=70, b=50, l=70, r=30),
            )
            st.plotly_chart(fig_vcut, use_container_width=True)

            rows_dl = []
            for i, tau in enumerate(chev_out["delay_list"]):
                for j, fd in enumerate(chev_out["freq_list"]):
                    rows_dl.append({"omega_d_GHz": fd, "delay_ns": tau, "p1": chev_out["Z"][i, j]})
            st.download_button("Download Chevron CSV", to_csv_bytes(pd.DataFrame(rows_dl)),
                               "ramsey_chevron.csv", "text/csv", key="chev_dl")

            st.markdown("""
**Interpreting the Chevron:**
- The **vertex** (brightest/widest fringe) marks the qubit frequency ω_q.
- Horizontal cuts at fixed ω_d show Ramsey fringes whose period is 1/Δ.
- The fringe envelope decays with T₂* (Ramsey) or T₂ (echo).
- Vertical cuts at ω_d = ω_q give the on-resonance decay.
""")


page()