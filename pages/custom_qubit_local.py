#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# Debug student (optional)
# ----------------------------
INSTRUCTOR_DEBUG = os.environ.get("QUBLITZ_INSTRUCTOR_DEBUG", "0").strip().lower() in ("1", "true", "yes")

TEST_STUDENT_API_KEY = "b8e60fc199646f8e712948304a65d52cd43b9bc3"
TEST_STUDENT = {
    "user": "debug_test_student",
    "omega_q": 4.671431666715805,            # GHz
    "omega_rabi": 0.20214882218103472,       # GHz
    "T1": 77.88719688437668,                 # ns (debug only)
}

# ----------------------------
# Assignment resolution config
# ----------------------------
API_BASE_URL = os.environ.get("QUBLITZ_API_BASE_URL", "http://10.28.54.127:8001").strip().rstrip("/")


# ----------------------------
# Helpers
# ----------------------------
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


def _clip_env(env: np.ndarray) -> np.ndarray:
    return np.clip(env, -1.0, 1.0)


def _extract_Z(prob_1_data: np.ndarray, freqs: np.ndarray, tlist: np.ndarray) -> np.ndarray:
    """
    Normalize prob_1_data shape to Z with shape (len(tlist), len(freqs)).
    """
    prob_1_data = np.asarray(prob_1_data, dtype=float)
    if prob_1_data.shape == (len(freqs), len(tlist)):
        return prob_1_data.T
    if prob_1_data.shape == (len(tlist), len(freqs)):
        return prob_1_data
    return prob_1_data


def _clear_results_and_pulses():
    st.session_state["freq_out"] = None
    st.session_state["freq_peak_cut"] = None
    st.session_state["td_out"] = None

    st.session_state["td_tfinal_last"] = None
    st.session_state["td_sigma_x_vec"] = None
    st.session_state["td_sigma_y_vec"] = None


def _init_state():
    st.session_state.setdefault("api_key", None)
    st.session_state.setdefault("user_data", None)

    st.session_state.setdefault("freq_out", None)
    st.session_state.setdefault("freq_peak_cut", None)

    st.session_state.setdefault("td_out", None)
    st.session_state.setdefault("td_tfinal_last", None)
    st.session_state.setdefault("td_sigma_x_vec", None)
    st.session_state.setdefault("td_sigma_y_vec", None)


def _try_backend_login(api_key: str) -> Dict[str, Any]:
    """
    Robust login:
    1) Try header-based: GET /me with X-API-Key
    2) Fallback to path-based: GET /me/{api_key}
    """
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
    except Exception:
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


def bloch_sphere_pretty(
    exp_x: np.ndarray,
    exp_y: np.ndarray,
    exp_z: np.ndarray,
    tlist: np.ndarray,
    t_final: float
) -> go.Figure:
    exp_x = np.asarray(exp_x, dtype=float)
    exp_y = np.asarray(exp_y, dtype=float)
    exp_z = np.asarray(exp_z, dtype=float)
    tlist = np.asarray(tlist, dtype=float)

    # ---- Flip Bloch sphere so |0> is at the top ----
    exp_z = -exp_z  # flip dynamics

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = -np.cos(v)  # flip sphere itself

    colors = tlist

    fig_bloch = go.Figure(data=[
        go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.25,
            showscale=False,
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=exp_x,
            y=exp_y,
            z=exp_z,
            mode="markers",
            marker=dict(
                size=4,
                color=colors,
                opacity=0.9,
                colorscale="Cividis",
                colorbar=dict(
                    title="Time [ns]",
                    len=0.85,
                    y=0.5,
                    thickness=14,
                    tickvals=[float(tlist[0]), float(t_final)],
                    ticktext=[f"{float(tlist[0]):.0f}", f"{float(t_final):.0f}"],
                ),
            ),
            name="State",
        ),
    ])

    fig_bloch.update_layout(
        title="State vector on the Bloch sphere",
        height=520,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title="⟨σx⟩",
            yaxis_title="⟨σy⟩",
            zaxis_title="⟨σz⟩",
            xaxis=dict(range=[-1, 1], showbackground=False, gridcolor="rgba(200,200,200,0.35)"),
            yaxis=dict(range=[-1, 1], showbackground=False, gridcolor="rgba(200,200,200,0.35)"),
            zaxis=dict(range=[-1, 1], showbackground=False, gridcolor="rgba(200,200,200,0.35)"),
            aspectmode="cube",
        ),
        showlegend=False,
    )

    # ---- Correctly labeled poles after flip ----
    fig_bloch.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[1.0, -1.0],
        mode="text",
        text=["|0⟩", "|1⟩"],
        textposition=["top center", "bottom center"],
        textfont=dict(color=["white", "white"], size=18),
        showlegend=False,
        hoverinfo="skip",
    ))

    return fig_bloch



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
        st.session_state["td_sigma_x_vec"] = _interp_to_grid(st.session_state["td_sigma_x_vec"], t_old, float(t_final_ns), n_steps)
        st.session_state["td_sigma_y_vec"] = _interp_to_grid(st.session_state["td_sigma_y_vec"], t_old, float(t_final_ns), n_steps)
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
        st.caption("You can stack pulses.")

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
        margin=dict(t=60, b=40, l=60, r=20),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_sigma, use_container_width=True)
    return tlist, sx, sy


def run_time_domain(omega_q_GHz, omega_rabi_GHz, T1_ns, omega_d_GHz, t_final_ns, sx_sched, sy_sched, shots):
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
    exp_y_rot = -(exp_values[0] * np.cos(2 * np.pi * omega_d_GHz * time_array) + exp_values[1] * np.sin(2 * np.pi * omega_d_GHz * time_array))
    exp_x_rot = (exp_values[0] * np.sin(2 * np.pi * omega_d_GHz * time_array) - exp_values[1] * np.cos(2 * np.pi * omega_d_GHz * time_array))

    return {
        "tlist": tlist,
        "exp_x_rot": np.asarray(exp_x_rot, dtype=float),
        "exp_y_rot": np.asarray(exp_y_rot, dtype=float),
        "exp_z": np.asarray(exp_values[2], dtype=float),
        "p1_meas": np.asarray(sampled_prob, dtype=float),
    }


# ----------------------------
# Page
# ----------------------------
def page():
    _init_state()

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

    # Sidebar images
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

    # JSON/API provides T1 in microseconds -> convert to ns for simulator
    T1_ns = float(user_data["T1"])

    shots = st.sidebar.number_input("Shots", min_value=32, max_value=4096, value=256, step=32, key="shots")
    is_debug_user = (st.session_state.get("api_key", "") == TEST_STUDENT_API_KEY)

    st.header("Custom Qubit Query")
    tab_freq, tab_time = st.tabs(["Frequency Domain", "Time Domain"])

    # ----------------------------
    # Frequency domain
    # ----------------------------
    with tab_freq:
        if is_debug_user:
            fd_start_default, fd_stop_default = float(omega_q - 0.25), float(omega_q + 0.25)
        else:
            fd_start_default, fd_stop_default = 4.8, 5.2

        start_freq = st.number_input(r"Start $\omega_d/2\pi$ [GHz]", value=fd_start_default, step=0.01, format="%.6f", key="fd_start")
        stop_freq = st.number_input(r"Stop $\omega_d/2\pi$ [GHz]", value=fd_stop_default, step=0.01, format="%.6f", key="fd_stop")
        num_points = st.number_input("Number of frequencies", value=41, min_value=5, max_value=201, step=2, key="fd_n")

        spec_tfinal = st.number_input("Pulse duration [ns]", value=25.0, min_value=1.0, max_value=500.0, step=1.0, key="fd_tfinal")
        n_steps = int(25 * float(spec_tfinal))

        if st.button("Run Frequency Sweep", key="fd_run"):
            try:
                results = run_frequency_sweep(
                    float(start_freq),
                    float(stop_freq),
                    int(num_points),
                    float(spec_tfinal),
                    int(n_steps),
                    float(omega_q),
                    float(omega_rabi),
                    float(T1_ns),
                    float(2.0 * T1_ns),
                    int(shots),
                )

                prob_1_data = np.array(results["prob_1_time_series"], dtype=float)
                frequencies = np.array(results["frequencies"], dtype=float)
                time_list = np.array(results["time_list"], dtype=float)

                Z = _extract_Z(prob_1_data, frequencies, time_list)

                max_prob = np.max(Z, axis=0)
                avg_prob = np.mean(Z, axis=0)
                peak_idx = int(np.argmax(max_prob))
                peak_freq = float(frequencies[peak_idx])

                # vertical cut at peak frequency: P1 vs time
                p1_cut = np.asarray(Z[:, peak_idx], dtype=float)

                st.session_state["freq_out"] = {
                    "frequencies": frequencies,
                    "time_list": time_list,
                    "Z": Z,
                    "max_prob": max_prob,
                    "avg_prob": avg_prob,
                    "peak_idx": peak_idx,
                    "peak_freq": peak_freq,
                }
                st.session_state["freq_peak_cut"] = {
                    "time_list": time_list,
                    "p1_cut": p1_cut,
                    "peak_freq": peak_freq,
                }

            except Exception as e:
                st.error(f"Frequency sweep failed: {e}")
                st.exception(e)

        out = st.session_state.get("freq_out")
        if out is not None:
            # Heatmap + summary curves (keep as-is, just fix colorbar extent)
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.09,
                subplot_titles=("Time-resolved P(|1⟩)", "Max P(|1⟩) over time", "Avg P(|1⟩) over time"),
                row_heights=[0.62, 0.20, 0.18],
            )

            # IMPORTANT: colorbar only for the heatmap trace (not a shared coloraxis)
            fig.add_trace(
                go.Heatmap(
                    x=out["frequencies"],
                    y=out["time_list"],
                    z=out["Z"],
                    colorscale="Viridis",
                    colorbar=dict(
                        title="P(|1⟩)",
                        len=0.62,   # matches top row height fraction
                        y=0.79,     # centered on the top panel
                        thickness=16,
                    ),
                    hovertemplate="ωd=%{x:.6f} GHz<br>t=%{y:.2f} ns<br>P1=%{z:.3f}<extra></extra>",
                ),
                row=1, col=1
            )
            fig.add_trace(go.Scatter(x=out["frequencies"], y=out["max_prob"], mode="lines", showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=out["frequencies"], y=out["avg_prob"], mode="lines", showlegend=False), row=3, col=1)

            fig.update_layout(
                height=980,
                title_text="Frequency Domain Simulation Results",
                xaxis3_title="Drive frequency [GHz]",
                yaxis1_title="Time [ns]",
                yaxis2_title="Max P(|1⟩)",
                yaxis3_title="Avg P(|1⟩)",
                margin=dict(t=70, b=50, l=70, r=50),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.success(f"Peak response near ωd ≈ {out['peak_freq']:.6f} GHz.")

            df = pd.DataFrame({"omega_d_GHz": out["frequencies"], "max_prob_1": out["max_prob"], "avg_prob_1": out["avg_prob"]})
            st.download_button("Download sweep summary CSV", to_csv_bytes(df), "frequency_sweep_summary.csv", "text/csv", key="fd_dl")

            # New: vertical cut at the peak, no new simulations
            cut = st.session_state.get("freq_peak_cut")
            if cut is not None:
                fig_cut = go.Figure()
                fig_cut.add_trace(go.Scatter(
                    x=cut["time_list"],
                    y=cut["p1_cut"],
                    mode="lines",
                    line=dict(width=3),
                ))
                fig_cut.update_layout(
                    height=360,
                    title=f"P(|1⟩) vs time at peak frequency (ωd = {cut['peak_freq']:.6f} GHz)",
                    xaxis_title="Time [ns]",
                    yaxis_title="P(|1⟩)",
                    yaxis=dict(range=[-0.05, 1.05]),
                    margin=dict(t=70, b=50, l=70, r=30),
                )
                st.plotly_chart(fig_cut, use_container_width=True)

    # ----------------------------
    # Time domain
    # ----------------------------
    with tab_time:
        omega_d_default = float(omega_q) if is_debug_user else 5.0
        omega_d = st.number_input(r'$\omega_d/2\pi$ [GHz]', value=omega_d_default, step=1e-6, format="%.9f", key="td_wd")

        t_final = st.number_input(r"Duration $\Delta t$ [ns]", value=200.0, min_value=1.0, max_value=2000.0, step=1.0, key="td_tfinal")
        tlist, sx_sched, sy_sched = pulse_ui(float(t_final))

        # Always show the three plots in their own rows (even before running):
        st.markdown("### Time-domain results")

        # Compute if needed, but display placeholders by default
        if st.session_state.get("td_out") is None:
            # initialize with a first run so plots appear immediately
            try:
                st.session_state["td_out"] = run_time_domain(
                    omega_q_GHz=omega_q,
                    omega_rabi_GHz=omega_rabi,
                    T1_ns=T1_ns,
                    omega_d_GHz=float(omega_d),
                    t_final_ns=float(t_final),
                    sx_sched=sx_sched,
                    sy_sched=sy_sched,
                    shots=int(shots),
                )
            except Exception:
                st.session_state["td_out"] = None

        if st.button("Re-run simulation with current settings", key="td_run"):
            try:
                st.session_state["td_out"] = run_time_domain(
                    omega_q_GHz=omega_q,
                    omega_rabi_GHz=omega_rabi,
                    T1_ns=T1_ns,
                    omega_d_GHz=float(omega_d),
                    t_final_ns=float(t_final),
                    sx_sched=sx_sched,
                    sy_sched=sy_sched,
                    shots=int(shots),
                )
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)

        out_td = st.session_state.get("td_out")

        # Row 1: rotating-frame dynamics
        if out_td is not None:
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
                margin=dict(t=70, b=50, l=70, r=30),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_results, use_container_width=True)
        else:
            st.info("Time-domain plots will appear once the simulator returns results.")

        # Row 2: Bloch sphere
        if out_td is not None:
            fig_bloch = bloch_sphere_pretty(
                exp_x=out_td["exp_x_rot"],
                exp_y=out_td["exp_y_rot"],
                exp_z=out_td["exp_z"],
                tlist=out_td["tlist"],
                t_final=float(t_final),
            )
            st.plotly_chart(fig_bloch, use_container_width=True)

        # Row 3: measurement record
        if out_td is not None:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=out_td["tlist"], y=out_td["p1_meas"], mode="lines", line=dict(width=3)))
            fig_p.update_layout(
                xaxis_title="Time [ns]",
                yaxis=dict(range=[-0.05, 1.05]),
                yaxis_title="Measured P(|1⟩)",
                title=f"Measurement record (shots={int(shots)})",
                height=360,
                margin=dict(t=70, b=50, l=70, r=30),
            )
            st.plotly_chart(fig_p, use_container_width=True)

            df = pd.DataFrame({
                "time_ns": out_td["tlist"],
                "exp_x_rot": out_td["exp_x_rot"],
                "exp_y_rot": out_td["exp_y_rot"],
                "exp_z": out_td["exp_z"],
                "p1_meas": out_td["p1_meas"],
            })
            st.download_button("Download time-domain CSV", to_csv_bytes(df), "time_domain_sim.csv", "text/csv", key="td_dl")


page()
