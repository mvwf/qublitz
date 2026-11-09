'''
Authors:
    M. Weiner

Release Date:
    V 1.0: 10/12/2024
    V 2.0: 11/2026, rewritten as a real IQ mixer (single-sideband up-conversion, imbalance, LO leakage)

'''
######################
###### IQ Mixer ######
# Author: Max Weiner #
### Date: 8/5/2024 ###
######################
# Modified by: Mattias Fitzpatrick
# Date: 07/27/2025

import numpy as np
from scipy.fft import rfft, rfftfreq
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from utils.branding import load_logo
from utils.ui import page_header


def mixer_signals(t, A, f_if, phi, A_lo, f_lo, sideband, gain_db, skew_deg, dc_i, dc_q):
    """The two arms of an IQ mixer and their sum.

    I(t) = A cos(2 pi f_if t + phi) + dc_i,   Q(t) = g A sin(2 pi f_if t + phi + delta) + dc_q
    RF(t) = I(t) cos(2 pi f_lo t) -/+ Q(t) sin(2 pi f_lo t)   (minus: upper sideband, plus: lower sideband)

    With g = 1, delta = 0 and no offsets the products' unwanted sidebands cancel and only f_lo + f_if (or f_lo - f_if)
    is left. An amplitude imbalance g or a phase skew delta leaves part of the image; a DC offset leaks the LO.
    """
    g = 10 ** (gain_db / 20)
    delta = np.deg2rad(skew_deg)
    w_if, w_lo, ph = 2 * np.pi * f_if, 2 * np.pi * f_lo, np.deg2rad(phi)
    I = A * np.cos(w_if * t + ph) + dc_i
    Q = g * A * np.sin(w_if * t + ph + delta) + dc_q
    lo_i = A_lo * np.cos(w_lo * t)
    lo_q = (-1.0 if sideband == "upper" else 1.0) * A_lo * np.sin(w_lo * t)
    arm_i, arm_q = I * lo_i, Q * lo_q
    return I, Q, arm_i, arm_q, arm_i + arm_q


def spectrum_db(x, dt):
    """Single-sided amplitude spectrum in dB (Hann window), referenced to a full-scale sine of amplitude 1."""
    n = len(x)
    w = np.hanning(n)
    X = np.abs(rfft(x * w)) * 2 / w.sum()
    return rfftfreq(n, dt), 20 * np.log10(np.maximum(X, 1e-12))


def peak_db(f, db, f0, df):
    """Largest spectral value within +-df of f0."""
    m = (f > f0 - df) & (f < f0 + df)
    return float(db[m].max()) if m.any() else float("nan")


def main():
    page_header(
        "IQ Mixing", "〰️",
        "See how in-phase/quadrature mixing up-converts a baseband signal onto a microwave carrier, "
        "and what imbalance and leakage do to the output.",
        "Learn",
    )
    qublitz_logo = load_logo("images/qublitz.png")
    st.sidebar.image(qublitz_logo)
    logo = load_logo("images/logo.png")
    st.sidebar.image(logo)
    st.sidebar.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)

    st.write(
        "An IQ mixer has two arms. The in-phase input $I(t)$ is multiplied by the local oscillator $\\cos(\\omega_{LO} t)$, "
        "the quadrature input $Q(t)$ by $\\mp\\sin(\\omega_{LO} t)$ (the LO shifted by 90°), and the two products are added:"
    )
    st.latex(r"RF(t) = I(t)\cos(\omega_{LO} t) \mp Q(t)\sin(\omega_{LO} t)")
    st.write(
        "Each product on its own is a double-sideband signal at $\\omega_{LO} \\pm \\omega_{IF}$. With "
        "$I = A\\cos(\\omega_{IF} t)$ and $Q = A\\sin(\\omega_{IF} t)$ the two lower sidebands arrive with opposite signs and "
        "cancel, so only $\\omega_{LO} + \\omega_{IF}$ survives: **single-sideband up-conversion**. The sign in front of $Q$ "
        "picks the sideband. In a real mixer the arms are never perfectly matched: an amplitude or phase imbalance leaves "
        "a residue of the **image** at $\\omega_{LO} \\mp \\omega_{IF}$, and a DC offset on $I$ or $Q$ lets the **LO leak** through. "
        "Qubit control electronics calibrate exactly these three knobs."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Baseband (IF) inputs**")
        A = st.slider('IF amplitude A [arb.]', 0.1, 2.0, 1.0, 0.1)
        f_if = st.slider('IF frequency [MHz]', 10, 500, 100, 10)
        phi = st.slider('IF phase φ [°]', 0.0, 360.0, 0.0, 1.0)
        sideband = st.radio('Sideband to keep', ['upper', 'lower'], horizontal=True)
    with col2:
        st.markdown("**Local oscillator**")
        A_lo = st.slider('LO amplitude [arb.]', 0.1, 2.0, 1.0, 0.1)
        f_lo = st.slider('LO frequency [GHz]', 1.0, 10.0, 5.0, 0.1)
    with col3:
        st.markdown("**Imperfections**")
        gain_db = st.slider('Amplitude imbalance of the Q arm [dB]', -3.0, 3.0, 0.0, 0.1)
        skew_deg = st.slider('Phase skew of the Q arm [°]', -30.0, 30.0, 0.0, 0.5)
        dc_i = st.slider('DC offset on I', -0.5, 0.5, 0.0, 0.01)
        dc_q = st.slider('DC offset on Q', -0.5, 0.5, 0.0, 0.01)

    # Time grid: 20 IF periods, sampled finely enough to resolve the LO (dt well below 1 / (2 f_LO)).
    f_if_hz, f_lo_hz = f_if * 1e6, f_lo * 1e9
    nsteps = 2 ** 16
    t = np.linspace(0.0, 20.0 / f_if_hz, nsteps, endpoint=False)
    dt = t[1] - t[0]
    I, Q, arm_i, arm_q, rf = mixer_signals(t, A, f_if_hz, phi, A_lo, f_lo_hz, sideband, gain_db, skew_deg, dc_i, dc_q)

    f, db_i = spectrum_db(arm_i, dt)
    _, db_q = spectrum_db(arm_q, dt)
    _, db_rf = spectrum_db(rf, dt)
    f_ghz = f / 1e9

    f_want = f_lo_hz + (f_if_hz if sideband == "upper" else -f_if_hz)
    f_image = f_lo_hz - (f_if_hz if sideband == "upper" else -f_if_hz)
    half_bin = 3 * (f[1] - f[0])
    p_want, p_image, p_lo = (peak_db(f, db_rf, x, half_bin) for x in (f_want, f_image, f_lo_hz))

    m1, m2, m3 = st.columns(3)
    m1.metric("Wanted sideband", f"{f_want / 1e9:.3f} GHz", f"{p_want:.1f} dB")
    m2.metric("Image rejection", f"{p_want - p_image:.1f} dB", "wanted − image", delta_color="off")
    m3.metric("LO leakage below wanted", f"{p_want - p_lo:.1f} dB", f"LO at {f_lo:.2f} GHz", delta_color="off")

    span = 3.5 * f_if_hz / 1e9
    fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Output spectra around the LO", "Output RF(t) in time"),
                           vertical_spacing=0.18, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=f_ghz, y=db_i, mode='lines', name='I arm: I(t)·cos(ω_LO t)', line=dict(color='#1f77b4', width=1), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_ghz, y=db_q, mode='lines', name='Q arm: ∓Q(t)·sin(ω_LO t)', line=dict(color='#d62728', width=1), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_ghz, y=db_rf, mode='lines', name='sum: RF(t)', line=dict(color='black', width=2)), row=1, col=1)
    for x, label, dash in ((f_want / 1e9, 'wanted', 'solid'), (f_image / 1e9, 'image', 'dot'), (f_lo, 'LO', 'dash')):
        fig.add_vline(x=x, line=dict(color='gray', width=1, dash=dash), annotation_text=label, annotation_position='top', row=1, col=1)
    fig.update_xaxes(title_text='Frequency [GHz]', range=[f_lo - span, f_lo + span], row=1, col=1)
    fig.update_yaxes(title_text='Amplitude [dB]', range=[-120, 10], row=1, col=1)

    n_show = int(min(nsteps, 2.5 / f_if_hz / dt))
    fig.add_trace(go.Scatter(x=t[:n_show] * 1e9, y=rf[:n_show], mode='lines', name='RF(t)', line=dict(color='black', width=1)), row=2, col=1)
    env = np.hypot(I, Q)[:n_show] * A_lo
    fig.add_trace(go.Scatter(x=t[:n_show] * 1e9, y=env, mode='lines', name='envelope |I + iQ|·A_LO', line=dict(color='#2ca02c', width=1.5, dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t[:n_show] * 1e9, y=-env, mode='lines', showlegend=False, line=dict(color='#2ca02c', width=1.5, dash='dash')), row=2, col=1)
    fig.update_xaxes(title_text='Time [ns]', row=2, col=1)
    fig.update_yaxes(title_text='Amplitude [arb.]', row=2, col=1)
    fig.update_layout(height=820, legend=dict(orientation='h', yanchor='bottom', y=1.03, xanchor='right', x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Try it.** With everything ideal the I and Q arms each show two lines at $f_{LO} \\pm f_{IF}$ (double sideband), "
        "and their sum keeps only one. Add a few degrees of phase skew or a fraction of a dB of imbalance and watch the "
        "image reappear: the image rejection is roughly $20\\log_{10}(2/\\sqrt{\\epsilon^2 + \\delta^2})$ for a fractional "
        "amplitude error $\\epsilon$ and a skew $\\delta$ in radians. A DC offset leaks the LO, because a constant times "
        "$\\cos(\\omega_{LO} t)$ is a tone at $\\omega_{LO}$. The time trace shows why a constant-envelope single-sideband "
        "output is just a carrier at a shifted frequency, whereas an imbalanced one beats at $2 f_{IF}$."
    )


if __name__ == "__main__":
    main()
