import math
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Cryogenic Laser Heating Calculator", layout="wide")

# ============================================================
# Helpers
# ============================================================

LAMBDA_M = 450e-9
MFD_M = 3.5e-6
W0_M = MFD_M / 2.0

PS_PEAK_W = 32e-3
NS_PEAK_W = 12e-3
CW_AVG_W = 3e-3

TRIGGER_VPP_MIN_50OHM = 0.2
TRIGGER_VPP_MAX_50OHM = 5.0
TRIGGER_VIH_HIZ = 2.2

EXTERNAL_TRIGGER_DELAY_NS = 44.0


def fmt_eng(x, unit=""):
    if x == 0:
        return f"0 {unit}".strip()

    sign = "-" if x < 0 else ""
    x = abs(x)

    prefixes = [
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "µ"),
        (1e-3, "m"),
        (1, ""),
        (1e3, "k"),
        (1e6, "M"),
        (1e9, "G"),
    ]

    chosen_scale = 1
    chosen_prefix = ""
    for scale, prefix in prefixes:
        if x >= scale:
            chosen_scale = scale
            chosen_prefix = prefix

    value = x / chosen_scale
    return f"{sign}{value:.4g} {chosen_prefix}{unit}".strip()


def gaussian_beam_radius(z_m, w0_m=W0_M, wavelength_m=LAMBDA_M):
    zR = math.pi * w0_m * w0_m / wavelength_m
    return w0_m * math.sqrt(1.0 + (z_m / zR) ** 2)


def circular_capture_fraction(target_radius_m, beam_radius_m):
    if beam_radius_m <= 0:
        return 0.0
    return 1.0 - math.exp(-2.0 * target_radius_m * target_radius_m / (beam_radius_m * beam_radius_m))


def ns_max_rep_rate_mhz(optical_pulse_ns):
    if optical_pulse_ns <= 2:
        return 200.0
    if optical_pulse_ns <= 4:
        return 100.0
    if optical_pulse_ns <= 18:
        return 30.0
    if optical_pulse_ns <= 20:
        return 20.0
    if optical_pulse_ns <= 25:
        return 16.0
    if optical_pulse_ns <= 30:
        return 13.0
    if optical_pulse_ns <= 40:
        return 10.0
    if optical_pulse_ns <= 65:
        return 5.0
    return 0.0


def temperature_trace(t_values, bath_temp_K, deltaT_ss_K, tau_th_s):
    out = []
    for t in t_values:
        out.append(bath_temp_K + deltaT_ss_K * (1.0 - math.exp(-t / tau_th_s)))
    return out


def make_plot(t_values, T_values, goal_temperature):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_values,
            y=T_values,
            mode="lines",
            name="Target temperature",
        )
    )
    fig.add_hline(
        y=goal_temperature,
        line_dash="dash",
        annotation_text="Goal temperature",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Predicted target temperature during an optical burst",
        xaxis_title="Burst duration [s]",
        yaxis_title="Target temperature [K]",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def bullet_row(name, value):
    st.markdown(f"**{name}:** {value}")


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Inputs")

mode = st.sidebar.selectbox(
    "Laser mode",
    ["Nanosecond pulses", "Picosecond pulses", "CW"],
)

trigger_source = st.sidebar.selectbox(
    "Trigger input type",
    ["50 Ω AC-coupled", "Hi-Z DC-coupled"],
)

trigger_freq_mhz = st.sidebar.number_input(
    "Trigger frequency / repetition rate [MHz]",
    min_value=0.0001,
    max_value=200.0,
    value=5.0,
    step=0.1,
    format="%.4f",
)

trigger_vpp = st.sidebar.number_input(
    "Trigger amplitude [Vpp]",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

if mode == "Nanosecond pulses":
    optical_pulse_ns = st.sidebar.slider(
        "Optical pulse width [ns]",
        min_value=1.0,
        max_value=65.0,
        value=10.0,
        step=0.5,
    )
elif mode == "Picosecond pulses":
    optical_pulse_ns = 0.09
    st.sidebar.info("Picosecond mode uses a nominal optical pulse width of 90 ps.")
else:
    optical_pulse_ns = None

fiber_length_m = st.sidebar.number_input(
    "Fiber length [m]",
    min_value=0.0,
    max_value=100.0,
    value=3.0,
    step=0.5,
)

fiber_loss_db_per_m = st.sidebar.number_input(
    "Fiber attenuation [dB/m]",
    min_value=0.0,
    max_value=5.0,
    value=0.03,
    step=0.01,
)

extra_insertion_loss_db = st.sidebar.number_input(
    "Extra insertion / connector / bend loss [dB]",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
)

distance_inch = st.sidebar.number_input(
    "Target distance from fiber tip [inch]",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

target_diameter_mm = st.sidebar.number_input(
    "Target diameter [mm]",
    min_value=0.01,
    max_value=50.0,
    value=3.0,
    step=0.1,
)

absorptivity = st.sidebar.slider(
    "Target absorptivity",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Thermal model")

bath_temp_K = st.sidebar.number_input(
    "Bath temperature [K]",
    min_value=0.001,
    max_value=300.0,
    value=0.02,
    step=0.01,
    format="%.4f",
)

deltaT_goal_K = st.sidebar.number_input(
    "Desired temperature rise [K]",
    min_value=0.001,
    max_value=300.0,
    value=20.0,
    step=1.0,
)

thermal_model = st.sidebar.radio(
    "Thermal parameterization",
    ["Direct heat capacity + thermal conductance", "Mass + specific heat + thermal conductance"],
)

if thermal_model == "Direct heat capacity + thermal conductance":
    C_J_per_K = st.sidebar.number_input(
        "Effective heat capacity C [J/K]",
        min_value=1e-12,
        max_value=10.0,
        value=1e-6,
        step=1e-6,
        format="%.6e",
    )
else:
    mass_kg = st.sidebar.number_input(
        "Target mass [kg]",
        min_value=1e-12,
        max_value=10.0,
        value=1e-6,
        step=1e-6,
        format="%.6e",
    )
    cp_J_per_kgK = st.sidebar.number_input(
        "Effective specific heat c_p [J/(kg·K)]",
        min_value=1e-6,
        max_value=1e9,
        value=1000.0,
        step=10.0,
        format="%.6e",
    )
    C_J_per_K = mass_kg * cp_J_per_kgK

G_W_per_K = st.sidebar.number_input(
    "Thermal conductance to bath G [W/K]",
    min_value=1e-12,
    max_value=10.0,
    value=2e-6,
    step=1e-6,
    format="%.6e",
)

burst_plot_max_s = st.sidebar.number_input(
    "Max burst duration shown in plot [s]",
    min_value=1e-6,
    max_value=1e4,
    value=1.0,
    step=0.1,
    format="%.6g",
)

n_plot_points = st.sidebar.slider(
    "Number of plot points",
    min_value=100,
    max_value=3000,
    value=600,
    step=100,
)

# ============================================================
# Calculations
# ============================================================

rep_rate_hz = trigger_freq_mhz * 1e6
distance_m = distance_inch * 0.0254
target_radius_m = 0.5 * target_diameter_mm * 1e-3

trigger_ok = True
trigger_msg = ""

if trigger_source == "50 Ω AC-coupled":
    if not (TRIGGER_VPP_MIN_50OHM <= trigger_vpp <= TRIGGER_VPP_MAX_50OHM):
        trigger_ok = False
        trigger_msg = (
            f"Trigger amplitude is out of range for 50 Ω input. "
            f"Use about {TRIGGER_VPP_MIN_50OHM:.1f} to {TRIGGER_VPP_MAX_50OHM:.1f} Vpp."
        )
else:
    if trigger_vpp < TRIGGER_VIH_HIZ:
        trigger_ok = False
        trigger_msg = (
            f"Trigger amplitude may be too small for a reliable logic-high on Hi-Z input. "
            f"Use at least about {TRIGGER_VIH_HIZ:.1f} V."
        )

if mode == "Nanosecond pulses":
    max_rep_mhz = ns_max_rep_rate_mhz(optical_pulse_ns)
    rep_ok = trigger_freq_mhz <= max_rep_mhz
    P_peak_emit_W = NS_PEAK_W
    tau_opt_s = optical_pulse_ns * 1e-9
    E_pulse_emit_J = P_peak_emit_W * tau_opt_s
    P_avg_emit_W = E_pulse_emit_J * rep_rate_hz
elif mode == "Picosecond pulses":
    max_rep_mhz = 200.0
    rep_ok = trigger_freq_mhz <= max_rep_mhz
    P_peak_emit_W = PS_PEAK_W
    tau_opt_s = 90e-12
    E_pulse_emit_J = P_peak_emit_W * tau_opt_s
    P_avg_emit_W = E_pulse_emit_J * rep_rate_hz
else:
    max_rep_mhz = None
    rep_ok = True
    P_peak_emit_W = CW_AVG_W
    tau_opt_s = None
    E_pulse_emit_J = None
    P_avg_emit_W = CW_AVG_W

total_loss_db = fiber_length_m * fiber_loss_db_per_m + extra_insertion_loss_db
transmission = 10.0 ** (-total_loss_db / 10.0)

P_peak_tip_W = P_peak_emit_W * transmission
P_avg_tip_W = P_avg_emit_W * transmission

w_target_m = gaussian_beam_radius(distance_m)
spot_area_m2 = math.pi * w_target_m * w_target_m
capture_fraction = circular_capture_fraction(target_radius_m, w_target_m)

geom_eff = capture_fraction
P_peak_abs_W = P_peak_tip_W * absorptivity * geom_eff
P_avg_abs_W = P_avg_tip_W * absorptivity * geom_eff

tau_th_s = C_J_per_K / G_W_per_K
deltaT_ss_K = P_avg_abs_W / G_W_per_K

if deltaT_ss_K > deltaT_goal_K and deltaT_ss_K > 0:
    t_req_s = -tau_th_s * math.log(1.0 - deltaT_goal_K / deltaT_ss_K)
    reachable = True
else:
    t_req_s = None
    reachable = False

if t_req_s is not None:
    n_pulses_req = rep_rate_hz * t_req_s
else:
    n_pulses_req = None

if E_pulse_emit_J is not None:
    E_pulse_tip_J = E_pulse_emit_J * transmission
    E_pulse_abs_J = E_pulse_tip_J * absorptivity * geom_eff
    deltaT_single_pulse_K = E_pulse_abs_J / C_J_per_K
else:
    E_pulse_tip_J = None
    E_pulse_abs_J = None
    deltaT_single_pulse_K = None

t_values = []
if n_plot_points < 2:
    n_plot_points = 2

for i in range(n_plot_points):
    frac = i / (n_plot_points - 1)
    t_values.append(frac * burst_plot_max_s)

T_values = temperature_trace(t_values, bath_temp_K, deltaT_ss_K, tau_th_s)

# ============================================================
# Main page
# ============================================================

st.title("Cryogenic Laser Heating Calculator")

st.write(
    "This app estimates the local temperature rise of a target illuminated by a fiber-delivered 450 nm laser. "
    "It treats the target as a lumped thermal mass with heat capacity "
    "$C$ and thermal conductance $G$ to a bath at temperature $T_{\\mathrm{bath}}$."
)

st.write("The model used is")

st.latex(r"P_{\mathrm{avg}} = E_{\mathrm{pulse}} f_{\mathrm{rep}}, \qquad E_{\mathrm{pulse}} \approx P_{\mathrm{peak}}\tau_{\mathrm{opt}}")
st.latex(r"P_{\mathrm{abs}} = P_{\mathrm{avg}} \times 10^{-L/10} \times \eta_{\mathrm{abs}} \times \eta_{\mathrm{geom}}")
st.latex(r"C \frac{dT}{dt} = P_{\mathrm{abs}} - G\left(T - T_{\mathrm{bath}}\right)")
st.latex(r"T(t)=T_{\mathrm{bath}}+\frac{P_{\mathrm{abs}}}{G}\left(1-e^{-t/\tau_{\mathrm{th}}}\right), \qquad \tau_{\mathrm{th}}=\frac{C}{G}")

st.write(
    "This is meant as a back-of-the-envelope engineering estimate, not a substitute for a calibrated in situ measurement."
)

# Status
if trigger_ok:
    st.success("Trigger amplitude is within a reasonable operating range.")
else:
    st.error(trigger_msg)

if mode != "CW":
    if rep_ok:
        st.success(f"Repetition rate is within the approximate allowable range for this pulse width, max ~ {max_rep_mhz:.1f} MHz.")
    else:
        st.error(f"Repetition rate exceeds the approximate allowable limit for this pulse width, max ~ {max_rep_mhz:.1f} MHz.")

st.info(f"Estimated trigger-to-optical delay: ~{EXTERNAL_TRIGGER_DELAY_NS:.0f} ns.")

# Metrics
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Average optical power emitted", fmt_eng(P_avg_emit_W, "W"))
    st.metric("Average optical power at fiber tip", fmt_eng(P_avg_tip_W, "W"))

with c2:
    st.metric("Peak optical power emitted", fmt_eng(P_peak_emit_W, "W"))
    st.metric("Peak optical power at fiber tip", fmt_eng(P_peak_tip_W, "W"))

with c3:
    st.metric("Absorbed average power", fmt_eng(P_avg_abs_W, "W"))
    if deltaT_single_pulse_K is not None:
        st.metric("Single-pulse ΔT", fmt_eng(deltaT_single_pulse_K, "K"))
    else:
        st.metric("Single-pulse ΔT", "N/A")

with c4:
    st.metric("Steady-state ΔT", fmt_eng(deltaT_ss_K, "K"))
    if t_req_s is not None:
        st.metric("Burst time to goal ΔT", fmt_eng(t_req_s, "s"))
    else:
        st.metric("Burst time to goal ΔT", "Not reachable")

# Plot
fig = make_plot(t_values, T_values, bath_temp_K + deltaT_goal_K)
st.plotly_chart(fig, use_container_width=True)

# Reachability text
if reachable:
    st.success(
        f"Under the current assumptions, a {deltaT_goal_K:.3g} K rise is reachable. "
        f"Estimated burst duration: {fmt_eng(t_req_s, 's')}, corresponding to about {fmt_eng(n_pulses_req, '')} pulses."
    )
else:
    st.warning(
        f"Under the current assumptions, the desired {deltaT_goal_K:.3g} K rise is not reachable in steady state. "
        f"The asymptotic rise is only {fmt_eng(deltaT_ss_K, 'K')}."
    )

st.markdown("---")
st.subheader("Parameter summary")

left, right = st.columns(2)

with left:
    bullet_row("Laser mode", mode)
    bullet_row("Optical pulse width", "N/A" if tau_opt_s is None else fmt_eng(tau_opt_s, "s"))
    bullet_row("Repetition rate", fmt_eng(rep_rate_hz, "Hz"))
    bullet_row("Total optical loss", f"{total_loss_db:.3f} dB")
    bullet_row("Transmission to fiber tip", f"{100.0 * transmission:.2f} %")
    bullet_row("Beam radius at target", fmt_eng(w_target_m, "m"))
    bullet_row("1/e² spot area at target", fmt_eng(spot_area_m2, "m²"))

with right:
    bullet_row("Target capture fraction", f"{100.0 * geom_eff:.2f} %")
    bullet_row("Absorptivity", f"{100.0 * absorptivity:.1f} %")
    bullet_row("Heat capacity C", fmt_eng(C_J_per_K, "J/K"))
    bullet_row("Thermal conductance G", fmt_eng(G_W_per_K, "W/K"))
    bullet_row("Thermal time constant C/G", fmt_eng(tau_th_s, "s"))
    bullet_row("Bath temperature", fmt_eng(bath_temp_K, "K"))
    bullet_row("Desired temperature rise", fmt_eng(deltaT_goal_K, "K"))

with st.expander("Interpretation notes"):
    st.markdown(
        """
- The external source mainly sets whether the laser is properly triggered and what repetition rate it runs at.
- In pulse mode, the average absorbed power is what mainly controls heating on slow timescales.
- The target temperature depends strongly on loss, beam expansion, geometric overlap, absorptivity, and thermal anchoring.
- This is a lumped thermal estimate. It is useful for order-of-magnitude discussion and for planning measurements.
"""
    )