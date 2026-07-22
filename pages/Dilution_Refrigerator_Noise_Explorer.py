import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------
# Physical constants
# -------------------------
k_B = 1.380649e-23       # Boltzmann constant (J/K)
h = 6.62607015e-34       # Planck constant (J*s)
hbar = h / (2 * np.pi)


# -------------------------
# Helpers
# -------------------------
def thermal_n(T, f_GHz):
    """
    Mean thermal photon number at temperature T [K] and frequency f [GHz].
    """
    f = f_GHz * 1e9
    x = h * f / (k_B * T)
    with np.errstate(over='ignore'):
        n = 1.0 / (np.exp(x) - 1.0)
    return n


def n_to_Teff(n, f_GHz):
    """
    Effective temperature corresponding to photon number n at frequency f [GHz].
    Solves n = 1/(exp(hf/kT)-1) for T.
    """
    f = f_GHz * 1e9
    hf_over_k = h * f / k_B
    n = np.maximum(n, 1e-20)  # avoid division by zero / log issues
    T = hf_over_k / np.log(1.0 + 1.0 / n)
    return T


def propagate_chain(freqs_GHz, stage_temps, stage_atten_dB):
    """
    Propagate thermal noise through attenuation chain.

    Parameters
    ----------
    freqs_GHz : array-like, shape (N_freq,)
    stage_temps : list of temperatures [K] for each stage (length N_stage)
    stage_atten_dB : list of attenuation [dB] at each stage (length N_stage)

    Returns
    -------
    n_eff : np.ndarray, shape (N_stage, N_freq)
        Effective thermal photon number at the output of each stage.
    T_eff : np.ndarray, shape (N_stage, N_freq)
        Effective temperature corresponding to n_eff at each stage.
    """
    freqs_GHz = np.array(freqs_GHz)
    n_stages = len(stage_temps)
    n_freq = len(freqs_GHz)

    n_eff = np.zeros((n_stages, n_freq))
    T_eff = np.zeros((n_stages, n_freq))

    # Stage 0: assume input noise at stage 0 is at T0
    T0 = stage_temps[0]
    for k, f in enumerate(freqs_GHz):
        n_in = thermal_n(T0, f)
        for i in range(n_stages):
            T_att = stage_temps[i]
            A_dB = stage_atten_dB[i]
            L = 10 ** (A_dB / 10.0)  # linear attenuation
            n_att = thermal_n(T_att, f)

            if L == 1.0:
                n_out = n_in
            else:
                # thermal beam-splitter model
                n_out = n_in / L + (1.0 - 1.0 / L) * n_att

            n_eff[i, k] = n_out
            T_eff[i, k] = n_to_Teff(n_out, f)
            n_in = n_out  # feed forward

    return n_eff, T_eff


def format_temp(T):
    """
    Nicely formatted temperature string.
    - < 0.5 K  -> mK
    - otherwise -> K
    """
    if T < 0.5:
        return f"{T * 1e3:.1f} mK"
    else:
        return f"{T:.3g} K"


def dBm_to_W(P_dBm):
    """Convert power from dBm to Watts."""
    return 10 ** ((P_dBm - 30.0) / 10.0)


def W_to_dBm(P_W):
    """Convert power from Watts to dBm."""
    P_W = np.maximum(P_W, 1e-30)
    return 10.0 * np.log10(P_W) + 30.0


# -------------------------
# Streamlit app
# -------------------------
st.set_page_config(
    page_title="Dilution Fridge Noise Explorer",
    layout="wide"
)

# st.sidebar.image("logo.png")

# CSS: color T_eff metrics wrapped in .teff-metric in red (we'll use this only for MXC)
st.markdown(
    """
    <style>
    .teff-metric [data-testid="stMetricValue"] {
        color: #e74c3c !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dilution Refrigerator Line Noise Explorer")

st.markdown(
    """
This app lets you explore the **effective thermal noise** along a microwave
drive line in a dilution refrigerator.

You can:
- Set **temperatures** and **attenuations** at each stage (300 K, 50 K, 4 K, Still, MXC),
- Choose the **frequency band**,
- See the **effective photon number**, **effective temperature**, and **noise level (dB)** 
  at each stage and frequency.
"""
)

# -------------------------
# Math section
# -------------------------
st.header("Math used in this app")

st.markdown(
    r"""
For a mode at frequency $f$ (in GHz) and temperature $T$, the mean thermal
photon number is
"""
)
st.latex(
    r"\bar n(T,f) = \frac{1}{\exp\!\left(\frac{h f}{k_{\mathrm{B}} T}\right) - 1}."
)

st.markdown(
r"""
Each stage has a physical temperature $T_i$ (set by the temperature inputs)
and an attenuation $A_i$ in dB (set by the attenuation inputs).  
We convert to linear attenuation
"""
)
st.latex(
    r"L_i = 10^{A_i/10}."
)

st.markdown(
    r"""
An attenuator at stage $i$ is modeled as a thermal beam splitter at temperature
$T_i$. If the input noise occupation to that stage is $\bar n_{\text{in}}$,
the output occupation is
"""
)
st.latex(
    r"\bar n_{\text{out}}"
    r"= \frac{\bar n_{\text{in}}}{L_i}"
    r" + \left(1 - \frac{1}{L_i}\right)\bar n(T_i,f)."
)

st.markdown(
    r"""
The code applies this map stage by stage, so that the output of stage $i$
feeds the input of stage $i+1$ for each frequency.

From a given occupation $\bar n_{\mathrm{eff}}(f)$ at some stage we define an
**effective temperature** $T_{\mathrm{eff}}$ by inverting the Bose–Einstein
relation:
"""
)

st.latex(
    r"\bar n_{\text{eff}}(f) ="
    r" \frac{1}{\exp\!\left(\frac{h f}{k_{\mathrm{B}} T_{\text{eff}}}\right) - 1}."
)

st.markdown(
    """
Finally, the **noise reduction vs room temperature** that is plotted in dB is
computed as
"""
)
st.latex(
    r"10\log_{10}\!\left(\frac{\bar n_{\mathrm{room}}(f)}{\bar n_{\mathrm{eff}}(f)}\right),"
)

st.markdown(
    r"where $\bar n_{\mathrm{room}}(f)$ is the thermal occupation at the "
    r"300 K stage and $\bar n_{\mathrm{eff}}(f)$ is the noise level after a "
    r"given stage for the same frequency."
)

st.markdown("---")

# -------------------------
# Default stage definitions
# -------------------------
default_stage_names = ["300 K (Room)", "50 K", "4 K", "Still", "MXC"]
default_stage_temps = [300.0, 50.0, 4.0, 0.7, 0.01]  # Kelvin

st.sidebar.header("Fridge Configuration")

# Temperatures (K for warm stages, mK for Still/MXC)
stage_temps = []
for name, T_default in zip(default_stage_names, default_stage_temps):

    if "Still" in name or "MXC" in name:
        T_default_mK = T_default * 1e3
        T_val_mK = st.sidebar.number_input(
            f"Stage temperature: {name} [mK]",
            min_value=0.0,
            value=float(T_default_mK),
            step=1.0,
            format="%.3f"
        )
        stage_temps.append(T_val_mK / 1e3)
    else:
        T_val = st.sidebar.number_input(
            f"Stage temperature: {name} [K]",
            min_value=0.0,
            value=float(T_default),
            step=0.01,
            format="%.4f"
        )
        stage_temps.append(T_val)

st.sidebar.markdown("---")

# Attenuation as number_inputs (with arrows)
st.sidebar.subheader("Attenuation at each stage [dB]")
atten_vals = []
for name in default_stage_names:
    default_A = 20.0 if ("300 K" not in name and "50 K" not in name) else 0.0
    A = st.sidebar.number_input(
        f"{name} atten. [dB]",
        min_value=0.0,
        max_value=60.0,
        value=default_A,
        step=1.0,
        format="%.1f"
    )
    atten_vals.append(A)

st.sidebar.markdown("---")

# Drive configuration: input power and cable insertion loss
st.sidebar.subheader("Drive configuration")
P_in_dBm = st.sidebar.number_input(
    "Signal generator power at 300 K [dBm]",
    min_value=-200.0,
    max_value=30.0,
    value=-60.0,
    step=1.0,
    format="%.1f",
)
cable_loss_dB = st.sidebar.number_input(
    "Cable insertion loss to MXC [dB]",
    min_value=0.0,
    max_value=60.0,
    value=15.0,
    step=0.5,
    format="%.1f",
)

st.sidebar.markdown("---")

# Toggle for showing n_eff in summary
show_n_eff = st.sidebar.checkbox("Show n_eff in per-stage summary", value=False)

st.sidebar.markdown("---")

# Frequency band
st.sidebar.subheader("Frequency band [GHz]")
f_min = st.sidebar.number_input("f_min [GHz]", 1.0, 20.0, 3.0, 0.1)
f_max = st.sidebar.number_input("f_max [GHz]", 1.0, 20.0, 6.0, 0.1)
if f_max <= f_min:
    st.sidebar.warning("f_max must be > f_min. Adjusting f_max.")
    f_max = f_min + 0.1
n_points = st.sidebar.slider("Number of frequency points", 50, 500, 200, 10)
f_ref = st.sidebar.slider("Reference frequency [GHz]", f_min, f_max, 5.0, 0.1)

freqs = np.linspace(f_min, f_max, n_points)

# Compute noise chain
n_eff, T_eff = propagate_chain(freqs, stage_temps, atten_vals)

# Noise relative to room temperature at same frequency
n_room = thermal_n(stage_temps[0], freqs)
noise_reduction_dB = 10 * np.log10(n_room / np.maximum(n_eff, 1e-30))

# -------------------------
# Drive power along the chain
# -------------------------
n_stages = len(default_stage_names)
atten_vals = np.array(atten_vals)
cum_atten = np.cumsum(atten_vals)

# Power at each stage due to the coherent drive (attenuators only)
P_dBm_stage = P_in_dBm - cum_atten

# Include additional cable insertion loss only on the MXC side
P_dBm_stage_with_cable = P_dBm_stage.copy()
P_dBm_stage_with_cable[-1] = P_dBm_stage_with_cable[-1] - cable_loss_dB

# For plotting vs frequency: power is flat vs f (ignore frequency-dependent attenuation)
P_dBm_stage_vs_f = np.zeros((n_stages, len(freqs)))
for i in range(n_stages):
    P_dBm_stage_vs_f[i, :] = P_dBm_stage_with_cable[i]

# -------------------------
# Per-stage summary at f_ref (main report)
# -------------------------
st.header("Per-stage summary at reference frequency")

idx_ref = (np.abs(freqs - f_ref)).argmin()
f_ref_Hz = freqs[idx_ref] * 1e9

summary_rows = []
drive_rows = []
for i, name in enumerate(default_stage_names):
    T_stage = stage_temps[i]
    A_dB = atten_vals[i]
    n_out = n_eff[i, idx_ref]
    T_out = T_eff[i, idx_ref]
    n_room_ref = n_room[idx_ref]
    red_dB = 10 * np.log10(n_room_ref / max(n_out, 1e-30))

    # Drive info for separate table
    P_stage_dBm = P_dBm_stage_with_cable[i]
    P_stage_W = dBm_to_W(P_stage_dBm)
    photons_per_s = P_stage_W / (h * f_ref_Hz)
    photons_per_us = photons_per_s * 1e-6

    summary_rows.append(
        {
            "Stage": name,
            "T_stage [K]": T_stage,
            "Atten [dB]": A_dB,
            "n_eff (ref)": n_out,
            "T_eff (ref) [K]": T_out,
            "Noise red vs room [dB]": red_dB,
        }
    )

    drive_rows.append(
        {
            "Stage": name,
            "Drive power (ref) [dBm]": P_stage_dBm,
            "Drive photons/us (ref)": photons_per_us,
        }
    )

summary_df = pd.DataFrame(summary_rows)
drive_df = pd.DataFrame(drive_rows)

cols = st.columns(len(default_stage_names))

for col, row in zip(cols, summary_rows):
    with col:
        st.markdown(f"### {row['Stage']}")
        st.metric("T_stage", format_temp(row["T_stage [K]"]))
        st.metric("Atten [dB]", f"{row['Atten [dB]']:.1f}")
        if show_n_eff:
            st.metric("n_eff (ref)", f"{row['n_eff (ref)']:.3e}")
        st.metric("Noise red vs room [dB]", f"{row['Noise red vs room [dB]']:.1f}")

        # T_eff: only highlight MXC in red/bold
        if "MXC" in row["Stage"]:
            st.markdown("<div class='teff-metric'>", unsafe_allow_html=True)
            st.metric("T_eff (ref)", format_temp(row["T_eff (ref) [K]"]))
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.metric("T_eff (ref)", format_temp(row["T_eff (ref) [K]"]))

with st.expander("Tabular per-stage summary at reference frequency"):
    table_df = summary_df.copy()
    if not show_n_eff:
        table_df = table_df.drop(columns=["n_eff (ref)"])

    fmt = {
        "T_stage [K]": "{:.3g}",
        "Atten [dB]": "{:.1f}",
        "T_eff (ref) [K]": "{:.3g}",
        "Noise red vs room [dB]": "{:.1f}",
    }
    if show_n_eff:
        fmt["n_eff (ref)"] = "{:.3e}"

    st.dataframe(table_df.style.format(fmt))

# -------------------------
# Separate drive power table (optional)
# -------------------------
show_drive_table = st.checkbox(
    "Show drive power & photon flux table at reference frequency",
    value=False,
)

if show_drive_table:
    st.subheader("Drive power and photon flux at reference frequency")
    st.dataframe(
        drive_df.style.format(
            {
                "Drive power (ref) [dBm]": "{:.1f}",
                "Drive photons/us (ref)": "{:.3e}",
            }
        )
    )

# -------------------------
# Plot: photon number vs frequency
# -------------------------
if st.checkbox("Show plot: thermal photon number vs frequency", value=True):
    st.header("Effective thermal photon number vs frequency")

    fig_n = go.Figure()
    for i, name in enumerate(default_stage_names):
        fig_n.add_trace(
            go.Scatter(
                x=freqs,
                y=n_eff[i, :],
                mode="lines",
                name=name
            )
        )

    fig_n.update_layout(
        xaxis_title="Frequency [GHz]",
        yaxis_title="Thermal photon number n_eff",
        yaxis_type="log",
        template="plotly_white",
        legend_title_text="Stage"
    )

    st.plotly_chart(fig_n, use_container_width=True)

# -------------------------
# Plot: effective temperature vs frequency
# -------------------------
if st.checkbox("Show plot: effective temperature vs frequency", value=True):
    st.header("Effective noise temperature vs frequency")

    fig_T = go.Figure()
    for i, name in enumerate(default_stage_names):
        fig_T.add_trace(
            go.Scatter(
                x=freqs,
                y=T_eff[i, :],
                mode="lines",
                name=name
            )
        )

    fig_T.update_layout(
        xaxis_title="Frequency [GHz]",
        yaxis_title="Effective temperature T_eff [K]",
        template="plotly_white",
        legend_title_text="Stage"
    )

    st.plotly_chart(fig_T, use_container_width=True)

# -------------------------
# Plot: noise level in dB relative to room
# -------------------------
if st.checkbox("Show plot: noise level vs room (in dB)", value=True):
    st.header("Noise level relative to 300 K (in dB)")

    fig_dB = go.Figure()
    for i, name in enumerate(default_stage_names):
        fig_dB.add_trace(
            go.Scatter(
                x=freqs,
                y=noise_reduction_dB[i, :],
                mode="lines",
                name=name
            )
        )

    fig_dB.update_layout(
        xaxis_title="Frequency [GHz]",
        yaxis_title="Noise reduction vs room [dB]",
        template="plotly_white",
        legend_title_text="Stage"
    )

    st.plotly_chart(fig_dB, use_container_width=True)

# -------------------------
# Plot: drive power vs frequency at each stage
# -------------------------
if st.checkbox("Show plot: drive power vs frequency (dBm)", value=False):
    st.header("Drive power vs frequency at each stage [dBm]")

    fig_P = go.Figure()
    for i, name in enumerate(default_stage_names):
        fig_P.add_trace(
            go.Scatter(
                x=freqs,
                y=P_dBm_stage_vs_f[i, :],
                mode="lines",
                name=name
            )
        )

    fig_P.update_layout(
        xaxis_title="Frequency [GHz]",
        yaxis_title="Drive power [dBm]",
        template="plotly_white",
        legend_title_text="Stage"
    )

    st.plotly_chart(fig_P, use_container_width=True)

# -------------------------
# Plot: drive photon flux vs frequency at each stage
# -------------------------
if st.checkbox("Show plot: drive photon flux vs frequency", value=False):
    st.header("Drive photon flux vs frequency at each stage")

    fig_flux = go.Figure()
    for i, name in enumerate(default_stage_names):
        P_W_stage = dBm_to_W(P_dBm_stage_with_cable[i])
        photon_flux = P_W_stage / (h * freqs * 1e9)  # photons/s

        fig_flux.add_trace(
            go.Scatter(
                x=freqs,
                y=photon_flux,
                mode="lines",
                name=name
            )
        )

    fig_flux.update_layout(
        xaxis_title="Frequency [GHz]",
        yaxis_title="Drive photon flux [photons/s]",
        yaxis_type="log",
        template="plotly_white",
        legend_title_text="Stage"
    )

    st.plotly_chart(fig_flux, use_container_width=True)

st.markdown(
    """
**Notes**

- Set attenuation at a stage to **0 dB** if there is no attenuator there.  
- Temperatures in the cards are shown in K or mK depending on the scale.  
- You can use this app to quickly explore how pushing attenuation to colder
  stages changes the effective bath seen by your sample.
- The drive section estimates the coherent signal power and corresponding photon
  flux at each stage, given the input power at 300 K and total attenuation
  (including cable loss to MXC).
- This app was mainly developed by **[Juan S. Salcedo-Gallo](https://www.linkedin.com/in/jussalcedoga/)**. Contributions are encouraged and welcome.
"""
)