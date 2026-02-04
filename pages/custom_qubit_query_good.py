#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qublitz Virtual Qubit Lab (minimal, two tabs only)
=================================================

What this version does (and ONLY this):
- Two tabs: Frequency Domain + Time Domain.
- Students infer: omega_q, omega_rabi, T1.
- We do NOT infer or display T2. Internally, we still pass a T2 value to the simulator
  because run_quantum_simulation / run_frequency_sweep signatures require it.
  We set T2_internal = 2*T1 (legacy physical upper bound) and we never show it.

Your requested changes applied:
1) Frequency-domain: DO NOT fix pulse duration. User sets pulse duration (ns).
2) Time-domain: allow more sig figs for omega_d input (format="%.9f", step=1e-6).
3) Remove "experiment controls" section (no Rabi sweep / no T1 sweep UI).
4) Do NOT show detuning anywhere (no Δ displayed).
5) Do NOT show ideal P(|1>) / sigma_z-derived curve in time-domain probability plot.
   (We still plot ⟨σz⟩ in the expectation-value plot, as in your original.)
6) Do NOT set wide default sweep ranges using hidden omega_q for non-debug students.
   - For the DEBUG test student key: we may use omega_q-centered defaults for convenience.
   - For everyone else: conservative generic defaults around 5 GHz.

No IQ blobs, no extra tabs.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantum_simulator import run_quantum_simulation, run_frequency_sweep

# Optional local assignment support (only used if enabled and available)
try:
    from assignment_server.utils.student_assignment import (
        build_assignments,
        lookup_assignment_by_api_key,
        assignment_to_user_data,
        DEFAULT_SECRET,
    )
except Exception:
    build_assignments = None
    lookup_assignment_by_api_key = None
    assignment_to_user_data = None
    DEFAULT_SECRET = "DEFAULT_SECRET_NOT_FOUND"

# ============================================================
# Config
# ============================================================
API_BASE_URL = os.environ.get("QUBLITZ_API_BASE_URL", "http://127.0.0.1:8001")
LOCAL_ASSIGNMENT_MODE = os.environ.get("QUBLITZ_LOCAL_ASSIGNMENT_MODE", "0").strip().lower() in ("1", "true", "yes")
ASSIGNMENT_SECRET = os.environ.get("ASSIGNMENT_SECRET", DEFAULT_SECRET)
API_KEY_MODE = os.environ.get("QUBLITZ_API_KEY_MODE", "netid").strip().lower()
NETIDS_CSV = os.environ.get("QUBLITZ_NETIDS_CSV", "net_IDs.csv")
NETID_COLUMN = os.environ.get("QUBLITZ_NETID_COLUMN", "SIS Login ID")

INSTRUCTOR_DEBUG = os.environ.get("QUBLITZ_INSTRUCTOR_DEBUG", "1").strip().lower() in ("1", "true", "yes")

# Debug test student parameters (only loaded if you enter this key)
TEST_STUDENT_API_KEY = "b8e60fc199646f8e712948304a65d52cd43b9bc3"
TEST_STUDENT = {
    "user": "debug_test_student",
    "omega_q": 4.671431666715805,            # GHz
    "omega_rabi": 0.20214882218103472,       # GHz (Ω/2π in GHz)
    "T1": 77.88719688437668,                 # ns
}

# ============================================================
# Helpers
# ============================================================
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _time_grid(t_final_ns: float, pts_per_ns: int = 25):
    n_steps = max(2, int(float(pts_per_ns) * float(t_final_ns)))
    tlist = np.linspace(0.0, float(t_final_ns), n_steps)
    return tlist, n_steps


def _interp_to_grid(vec: np.ndarray, t_old: float, t_new: float, n_new: int):
    if vec is None:
        return np.zeros(n_new, dtype=float)
    vec = np.asarray(vec, dtype=float)
    if len(vec) == n_new and float(t_old) == float(t_new):
        return vec
    x_old = np.linspace(0.0, float(t_old), len(vec))
    x_new = np.linspace(0.0, float(t_new), n_new)
    return np.interp(x_new, x_old, vec).astype(float)


def _clip_env(env):
    return np.clip(env, -1.0, 1.0)


# ============================================================
# Session state init
# ============================================================
def _init_state():
    st.session_state.setdefault("api_key", None)
    st.session_state.setdefault("user_data", None)

    # Pulse schedule state (time domain)
    st.session_state.setdefault("td_tfinal_last", None)
    st.session_state.setdefault("td_sigma_x_vec", None)
    st.session_state.setdefault("td_sigma_y_vec", None)

    # Results cache
    st.session_state.setdefault("freq_out", None)
    st.session_state.setdefault("td_out", None)


# ============================================================
# Auth / assignments
# ============================================================
@st.cache_resource(show_spinner=False)
def _load_assignments_cached():
    if build_assignments is None:
        return None
    return build_assignments(
        csv_path=NETIDS_CSV,
        column=NETID_COLUMN,
        secret=ASSIGNMENT_SECRET,
        api_key_mode=API_KEY_MODE,
    )

# ============================================================
# Auth (NO local assignment fallback)
# ============================================================

def _try_backend_login(api_key: str):
    r = requests.get(f"{API_BASE_URL}/me", headers={"X-API-Key": api_key}, timeout=8)
    r.raise_for_status()
    return r.json()


def login_ui():
    st.title("Qublitz Virtual Qubit Lab")
    st.caption("Enter your API key.")

    default_val = TEST_STUDENT_API_KEY if INSTRUCTOR_DEBUG else ""
    api_key = st.text_input("API Key", type="password", value=default_val, key="login_api_key")

    if st.button("Login"):
        try:
            key = api_key.strip()
            if not key:
                raise ValueError("Empty API key.")

            if INSTRUCTOR_DEBUG and key == TEST_STUDENT_API_KEY:
                user_data = dict(TEST_STUDENT)
            else:
                user_data = _try_backend_login(key)

            st.session_state["api_key"] = key
            st.session_state["user_data"] = user_data
            st.success(f"Welcome {user_data.get('user', 'student')}!")
            st.rerun()

        except Exception as e:
            st.error(f"Login failed: {e}")
            st.exception(e)


# ============================================================
# Pulse UI (square + gaussian, minimal)
# ============================================================
def pulse_ui(t_final_ns: float):
    """
    Maintains td_sigma_x_vec, td_sigma_y_vec in session_state.
    Re-grids when t_final changes.
    Returns (tlist, sx, sy) as scheduled (unclipped) vectors.
    """
    tlist, n_steps = _time_grid(t_final_ns, pts_per_ns=25)

    if st.session_state["td_tfinal_last"] is None:
        st.session_state["td_tfinal_last"] = float(t_final_ns)

    # Initialize
    if st.session_state["td_sigma_x_vec"] is None:
        st.session_state["td_sigma_x_vec"] = np.zeros(n_steps, dtype=float)
    if st.session_state["td_sigma_y_vec"] is None:
        st.session_state["td_sigma_y_vec"] = np.zeros(n_steps, dtype=float)

    # Re-grid if needed
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
            sigma = st.slider(
                "Sigma [ns]",
                1.0,
                float(max(2.0, t_final_ns / 2.0)),
                float(max(2.0, min(10.0, t_final_ns / 10.0))),
                step=1.0,
                key="td_g_sigma",
            )

        add_btn = st.button("Add pulse", key="td_add")
        clear_btn = st.button("Clear schedule", key="td_clear")

    with col2:
        st.caption("You can stack pulses. Input to simulator is clipped to [-1, 1].")

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
        xaxis_title="Time [ns]",
        yaxis_title="Amplitude",
        title="Time-dependent envelopes",
        xaxis=dict(range=[0, float(t_final_ns)]),
        yaxis=dict(range=[-1.05, 1.05]),
        height=320,
    )
    st.plotly_chart(fig_sigma, use_container_width=True)

    return tlist, sx, sy


# ============================================================
# Simulator wrapper (T2 hidden internally)
# ============================================================
def run_time_domain(
    omega_q_GHz: float,
    omega_rabi_GHz: float,
    T1_ns: float,
    omega_d_GHz: float,
    t_final_ns: float,
    sx_sched: np.ndarray,
    sy_sched: np.ndarray,
    shots: int,
):
    # Hide T2: still required by simulator signature
    T2_internal = 2.0 * float(T1_ns)

    tlist, n_steps = _time_grid(t_final_ns, pts_per_ns=25)

    sx_in = _clip_env(np.asarray(sx_sched, dtype=float))
    sy_in = _clip_env(np.asarray(sy_sched, dtype=float))

    exp_values, _, sampled_prob = run_quantum_simulation(
        float(omega_q_GHz),
        float(omega_rabi_GHz),
        float(t_final_ns),
        int(n_steps),
        float(omega_d_GHz),
        sx_in,
        sy_in,
        int(shots),
        float(T1_ns),
        float(T2_internal),
    )

    time_array = np.linspace(0.0, float(t_final_ns), n_steps)

    exp_y_rot = -(
        exp_values[0] * np.cos(2 * np.pi * omega_d_GHz * time_array)
        + exp_values[1] * np.sin(2 * np.pi * omega_d_GHz * time_array)
    )
    exp_x_rot = (
        exp_values[0] * np.sin(2 * np.pi * omega_d_GHz * time_array)
        - exp_values[1] * np.cos(2 * np.pi * omega_d_GHz * time_array)
    )

    return {
        "tlist": tlist,
        "exp_x_rot": np.asarray(exp_x_rot, dtype=float),
        "exp_y_rot": np.asarray(exp_y_rot, dtype=float),
        "exp_z": np.asarray(exp_values[2], dtype=float),
        "p1_meas": np.asarray(sampled_prob, dtype=float),
    }


# ============================================================
# Main app
# ============================================================
def main_app():
    user_data = st.session_state.get("user_data", None)
    if not user_data:
        st.error("No user data. Please log in again.")
        return

    # Sidebar logos
    try:
        st.sidebar.image(Image.open("images/qublitz.png"))
    except Exception:
        pass
    try:
        st.sidebar.image(Image.open("images/logo.png"))
    except Exception:
        pass

    st.sidebar.markdown(
        '<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" '
        'style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>',
        unsafe_allow_html=True,
    )

    st.sidebar.write(f"Logged in as: **{user_data.get('user', 'student')}**")
    if st.sidebar.button("Logout"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k, None)
        st.rerun()

    # Hidden parameters from API
    omega_q = float(user_data["omega_q"])          # GHz
    omega_rabi = float(user_data["omega_rabi"])    # GHz (Ω/2π)
    T1 = float(user_data["T1"])                    # ns

    # Minimal global controls (safe to show students)
    st.sidebar.header("Measurement settings")
    shots = st.sidebar.number_input("Shots", min_value=32, max_value=4096, value=256, step=32)

    is_debug_user = (st.session_state.get("api_key", "") == TEST_STUDENT_API_KEY)

    # Optional debug truth display (only if debug key entered)
    if INSTRUCTOR_DEBUG and is_debug_user:
        st.sidebar.header("Debug (truth)")
        st.sidebar.code(
            f"omega_q    = {omega_q:.12f} GHz\n"
            f"omega_rabi = {omega_rabi:.12f} GHz\n"
            f"T1         = {T1:.9f} ns\n",
            language="text",
        )

    st.title("Qublitz Virtual Qubit Lab")
    st.header("Driven qubit (two-level system)")
    st.subheader(
        r'$\hat{H}/\hbar = \frac{\omega_q}{2}\hat{\sigma}_z + \frac{\Omega(t)}{2}\hat{\sigma}_x\cos(\omega_d t) + \frac{\Omega(t)}{2}\hat{\sigma}_y\cos(\omega_d t)$'
    )
    st.latex(
        r'''\text{Where } |1\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \text{ and } |0\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}'''
    )

    tab_freq, tab_time = st.tabs(["Frequency Domain", "Time Domain"])

    # ============================================================
    # Frequency Domain tab (pulse duration is user-controlled)
    # ============================================================
    with tab_freq:
        st.subheader("Frequency domain")

        if is_debug_user:
            fd_start_default = float(omega_q - 0.25)
            fd_stop_default = float(omega_q + 0.25)
        else:
            # Generic defaults (do NOT leak ωq-centered bounds)
            fd_start_default = 4.8
            fd_stop_default = 5.2

        start_freq = st.number_input(
            r"Start $\omega_d/2\pi$ [GHz]",
            value=fd_start_default,
            step=0.01,
            format="%.6f",
            key="fd_start",
        )
        stop_freq = st.number_input(
            r"Stop $\omega_d/2\pi$ [GHz]",
            value=fd_stop_default,
            step=0.01,
            format="%.6f",
            key="fd_stop",
        )
        num_points = st.number_input("Number of frequencies", value=41, min_value=5, max_value=201, step=2, key="fd_nfreq")

        spec_tfinal = st.number_input(
            "Pulse duration [ns]",
            value=25.0,
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            key="fd_tfinal",
        )
        n_steps = int(25 * float(spec_tfinal))

        if st.button("Run Frequency Sweep", key="fd_run"):
            prog = st.progress(0.0)
            txt = st.empty()
            txt.write("Running frequency sweep...")

            try:
                prog.progress(0.2)
                results = run_frequency_sweep(
                    float(start_freq),
                    float(stop_freq),
                    int(num_points),
                    float(spec_tfinal),
                    int(n_steps),
                    float(omega_q),
                    float(omega_rabi),
                    float(T1),
                    float(2.0 * T1),   # internal only
                    int(shots),
                )

                prog.progress(0.8)
                txt.write("Processing results...")

                prob_1_data = np.array(results["prob_1_time_series"], dtype=float)
                frequencies = np.array(results["frequencies"], dtype=float)
                time_list = np.array(results["time_list"], dtype=float)

                # Safer orientation: we want Z as [time, freq]
                if prob_1_data.shape == (len(frequencies), len(time_list)):
                    Z = prob_1_data.T
                elif prob_1_data.shape == (len(time_list), len(frequencies)):
                    Z = prob_1_data
                else:
                    Z = prob_1_data  # fallback

                max_prob = np.max(Z, axis=0)
                avg_prob = np.mean(Z, axis=0)

                st.session_state["freq_out"] = {
                    "frequencies": frequencies,
                    "time_list": time_list,
                    "Z": Z,
                    "max_prob": max_prob,
                    "avg_prob": avg_prob,
                }

                prog.progress(1.0)
                txt.empty()
                prog.empty()

            except Exception as e:
                txt.empty()
                prog.empty()
                st.error(f"Frequency sweep failed: {e}")
                st.exception(e)

        out = st.session_state.get("freq_out", None)
        if out is not None:
            frequencies = out["frequencies"]
            time_list = out["time_list"]
            Z = out["Z"]
            max_prob = out["max_prob"]
            avg_prob = out["avg_prob"]

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=("Time-resolved P(|1⟩)", "Max P(|1⟩) over time", "Avg P(|1⟩) over time"),
                row_heights=[0.6, 0.2, 0.2],
            )

            fig.add_trace(go.Heatmap(x=frequencies, y=time_list, z=Z, coloraxis="coloraxis"), row=1, col=1)
            fig.add_trace(go.Scatter(x=frequencies, y=max_prob, mode="lines", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=frequencies, y=avg_prob, mode="lines", showlegend=False), row=3, col=1)

            fig.update_layout(
                height=900,
                title_text="Frequency Domain Simulation Results",
                coloraxis=dict(colorscale="Viridis"),
                xaxis3_title="Drive frequency [GHz]",
                yaxis1_title="Time [ns]",
                yaxis2_title="Max P(|1⟩)",
                yaxis3_title="Avg P(|1⟩)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Keep the hint line (doesn't leak hidden params; it uses computed peak)
            idx = int(np.argmax(max_prob))
            st.success(f"Peak response near ωd ≈ {frequencies[idx]:.6f} GHz.")

            df = pd.DataFrame({
                "omega_d_GHz": frequencies,
                "max_prob_1": max_prob,
                "avg_prob_1": avg_prob,
            })
            st.download_button(
                label="Download sweep summary CSV",
                data=to_csv_bytes(df),
                file_name="frequency_sweep_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("Click 'Run Frequency Sweep' to generate the heatmap and traces.")

    # ============================================================
    # Time Domain tab (no experiment controls, no detuning, no ideal curve)
    # ============================================================
    with tab_time:
        st.subheader("Time domain")

        omega_d = st.number_input(
            r'$\omega_d/2\pi$ [GHz]',
            value=5.0 if not is_debug_user else float(omega_q),
            step=1e-6,
            format="%.9f",
            key="td_wd",
        )

        st.markdown("## Pulse schedule (σx, σy) and simulation")
        t_final = st.number_input(
            r"Duration $\Delta t$ [ns]",
            value=200.0,
            min_value=1.0,
            max_value=2000.0,
            step=10.0,
            key="td_tfinal",
        )
        tlist, sx_sched, sy_sched = pulse_ui(float(t_final))

        if st.button("Run Simulation", key="td_run_sim"):
            try:
                out_td = run_time_domain(
                    omega_q_GHz=omega_q,
                    omega_rabi_GHz=omega_rabi,
                    T1_ns=T1,
                    omega_d_GHz=float(omega_d),
                    t_final_ns=float(t_final),
                    sx_sched=sx_sched,
                    sy_sched=sy_sched,
                    shots=int(shots),
                )
                st.session_state["td_out"] = out_td
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)

        out_td = st.session_state.get("td_out", None)
        if out_td is not None:
            # Rotating frame expectations (same as your original intent)
            fig_results = go.Figure()
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_x_rot"], mode="lines", name="⟨σx⟩", line=dict(width=3)))
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_y_rot"], mode="lines", name="⟨σy⟩", line=dict(width=3)))
            fig_results.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["exp_z"], mode="lines", name="⟨σz⟩", line=dict(width=3)))
            fig_results.update_layout(
                yaxis=dict(range=[-1.05, 1.05]),
                xaxis_title="Time [ns]",
                yaxis_title="Expectation values",
                title=f"Rotating-frame dynamics (ωd={float(omega_d):.9f} GHz)",
                height=420,
            )
            st.plotly_chart(fig_results, use_container_width=True)

            # Measured probability ONLY (no ideal overlay)
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["p1_meas"], mode="lines", line=dict(width=3)))
            fig_p.update_layout(
                xaxis_title="Time [ns]",
                yaxis=dict(range=[-0.05, 1.05]),
                yaxis_title="Measured P(|1⟩)",
                title=f"Measurement record (shots={int(shots)})",
                height=320,
            )
            st.plotly_chart(fig_p, use_container_width=True)

            # Bloch sphere
            time_values = out_td["tlist"]
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_sphere = np.cos(u) * np.sin(v)
            y_sphere = np.sin(u) * np.sin(v)
            z_sphere = np.cos(v)

            fig_bloch = go.Figure(data=[
                go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3, showscale=False),
                go.Scatter3d(
                    x=out_td["exp_x_rot"],
                    y=out_td["exp_y_rot"],
                    z=out_td["exp_z"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=time_values,
                        opacity=0.8,
                        colorscale="inferno",
                        colorbar=dict(title="Time [ns]"),
                    ),
                    showlegend=False,
                ),
            ])

            fig_bloch.add_trace(go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[1.0, -1.0],
                mode="text",
                text=["|1⟩", "|0⟩"],
                textposition=["top center", "bottom center"],
                textfont=dict(color=["white", "white"], size=20),
                showlegend=False,
            ))

            fig_bloch.update_layout(
                title="State vector on the Bloch sphere",
                scene=dict(
                    xaxis_title="σ_x",
                    yaxis_title="σ_y",
                    zaxis_title="σ_z",
                    xaxis=dict(range=[-1, 1]),
                    yaxis=dict(range=[-1, 1]),
                    zaxis=dict(range=[-1, 1]),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=650,
            )
            st.plotly_chart(fig_bloch, use_container_width=True)

            # CSV download
            df = pd.DataFrame({
                "time_ns": out_td["tlist"],
                "exp_x_rot": out_td["exp_x_rot"],
                "exp_y_rot": out_td["exp_y_rot"],
                "exp_z": out_td["exp_z"],
                "p1_meas": out_td["p1_meas"],
            })
            st.download_button("Download time-domain CSV", to_csv_bytes(df), "time_domain_sim.csv", "text/csv")


def run():
    _init_state()
    if st.session_state.get("user_data", None) is None:
        login_ui()
    else:
        main_app()


run()
