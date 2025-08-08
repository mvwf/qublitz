"""
Authors:
    S. Sacerdote

Release Date:
    V 1.0: 7/30/25
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.tpd_locations_nd import ep_location, tpd_location
from utils.pt_peaks_MODEL import peak_location, eigenvalues
from PIL import Image
# map degeneracy type to color and marker
color_marker_dict = {
    "PRIMARY_EP": ("red", "x-thin"),
    "PRIMARY_TPD": ("red", "circle-open"),
    "SECONDARY_EP": ("gray", "x-thin"),
    "SECONDARY_TPD": ("gray", "circle-open"),
    "ROGUE_TPD": ("gray", "diamond-open")
}

# function: plots contours
def plot_contours(fig, contour, x, y, color, linestyle, linewidth, label):
    fig.add_trace(go.Contour(
    z = contour,
    x = x,
    y = y,
    showscale = False,
    contours = dict(start=0, end=0, size=1, coloring='none'),
    line = dict(color=color, dash=linestyle, width=linewidth),
    name = label
))

# function: plots the degeneracies
def plot_degeneracies(fig, degeneracies):
    # Track which types have already been plotted to avoid duplicates in the legend
    plotted_types = set()

    # Plot each degeneracy with proper labels, colors, and markers
    for degen in degeneracies:
        dtype = degen.degeneracy_type
        
        if dtype not in plotted_types:
            label_list = dtype.name.split("_")
            label_list[0] = label_list[0].title()
            label = " ".join(label_list)
        else:
            label = None

        color, marker = color_marker_dict.get(dtype.name)
        fig.add_trace(go.Scatter(
            x = [degen.Delta_tilde_kappa],
            y = [degen.Delta_tilde_f],
            mode = 'markers',
            marker = dict(
                color=color, 
                symbol=marker, 
                size=16, 
                line=dict(width=4, color=color)),
            name = label,
            showlegend = (label != None)
        ))

        plotted_types.add(dtype)

# function: plots traces
def plot_trace(fig, x, y, name, legendgroup, showlegend, line):
    fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        mode='lines', 
        name=name, 
        legendgroup=legendgroup,
        showlegend=showlegend,
        line=line
    ))

# main function
def main():

    # create a title for the page
    st.title("Exceptional Point and Transmission Peak Degeneracy Locations")

    qublitz_logo = Image.open("images/qublitz.png")
    st.sidebar.image(qublitz_logo)
    logo = Image.open("images/logo.png") 
    st.sidebar.image(logo) # display logo on the side 
    st.sidebar.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)
    
    # link to preprint of the paper
    st.markdown('''
    <p style="font-size:18px;">
    <a href="https://arxiv.org/abs/2506.09141" target="_blank">
    Unification of Exceptional Points and Transmission Peak Degeneracies in a Highly Tunable Magnon-Photon Dimer
    </a>
    </p>
    ''', unsafe_allow_html=True)

    # state space matrix
    st.markdown('<p style="font-size:18px;">Based on the dynamical system matrix for a dimer:</p>', unsafe_allow_html=True)
    st.latex(r"""
    \tilde{A} = \left[
    \begin{array}{cc}
    -i(\tilde{f}_c - \tilde{f}_d) - \frac{\tilde{\kappa}_c}{2} & -i \\
    -i e^{i \phi} & -i(\tilde{f}_c - \tilde{f}_d - \tilde{\Delta}_f) + \left( \tilde{\Delta}_\kappa - \frac{\tilde{\kappa}_c}{2} \right)
    \end{array}
    \right]
    """)

    # create sidebar with the Parameters
    st.markdown('<p style="font-size:18px;">System Parameters:</p>', unsafe_allow_html=True)
    kappa_tilde_c = st.slider(r"$\tilde{\kappa}_c$", 0.0, 2.5, 0.68, step=0.01)

    phi_labels = {
        "0": 0,
        "π/8": np.pi / 8,
        "π/4": np.pi / 4,
        "3π/8": 3 * np.pi / 8,
        "π/2": np.pi / 2,
        "5π/8": 5 * np.pi / 8,
        "3π/4": 3 * np.pi / 4,
        "7π/8": 7 * np.pi / 8,
        "π": np.pi,
        "9π/8": 9 * np.pi / 8,
        "5π/4": 5 * np.pi / 4,
        "11π/8": 11 * np.pi / 8,
        "3π/2": 3 * np.pi / 2,
        "13π/8": 13 * np.pi / 8,
        "7π/4": 7 * np.pi / 4,
        "15π/8": 15 * np.pi / 8,
        "2π": 0
    }

    phi_label = st.select_slider("Φ - Coupling Phase", options=list(phi_labels.keys()), value="0")
    phi = phi_labels[phi_label]
    # phi = st.sidebar.slider("Φ - Coupling Phase (rad)", 0.0, (2 * np.pi), 0.0, step=0.01) # Option for if we want a continuous slider

    # define x and y axis for the plot (x is Delta_tilde_kappa and y is Delta_tilde_f)
    x1 = np.linspace(-4, 4, 500)
    y1 = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x1, y1)

    # define modulus squared of Delta_tilde_lambda in terms of X and Y
    Delta_tilde_lambda = np.sqrt(-Y ** 2 + 2j * Y * X + X ** 2 - 4 * np.exp(1j * phi))
    Delta_tilde_lambda_mod_squared = abs(Delta_tilde_lambda) ** 2

    # petermann Factor
    K_2_tilde = (Y ** 2 + X ** 2 + Delta_tilde_lambda_mod_squared + 4) / (2 * Delta_tilde_lambda_mod_squared)

    # contour values
    tilde_q = ((kappa_tilde_c - X) * (Delta_tilde_lambda ** 2).imag) / 8
    tilde_p = (((kappa_tilde_c - X) ** 2) + (Delta_tilde_lambda ** 2).real) / 4
    disc = -4 * (tilde_p ** 3) - 27 * (tilde_q ** 2)
    instability = (Delta_tilde_lambda.real) - (kappa_tilde_c - X)
    if phi == 0 or phi == 2 * np.pi:
        min_petermann = X
    elif phi == np.pi:
            min_petermann = Y
    else:
            min_petermann = (-1 / np.tan(phi / 2)) * X - Y

    # compute degeneracy locations
    eps = ep_location(phi)
    tpds = tpd_location(phi, kappa_tilde_c)

    # mask the petermann factor into two regions
    K_color = np.where(instability <= 0, K_2_tilde, np.nan)  # plasma region
    K_gray = np.where(instability > 0, K_2_tilde, np.nan)   # grayscale region

    # plot petermann factor color map
    fig1 = go.Figure()

    # plasma where instability ≥ 0
    fig1.add_trace(go.Heatmap(
        x=x1,
        y=y1,
        z=K_color,
        colorscale='plasma',
        zmin=1.0,
        zmax=1.6,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Petermann Factor",
                font=dict(size=20),
                side="right"),
            tickfont=dict(size=16)
        )
    ))

    # grayscale where instability < 0
    fig1.add_trace(go.Heatmap(
        x=x1,
        y=y1,
        z=K_gray,
        colorscale='gray',
        zmin=1.0,
        zmax=1.6,
        showscale=False  # hide second colorbar
    ))

    # plot the contours
    plot_contours(fig1, tilde_q, x1, y1, 'magenta', 'dash', 3, '𝑞̃ = 0')
    plot_contours(fig1, disc, x1, y1, 'cyan', 'dash', 3, 'Disc = 0')
    plot_contours(fig1, instability, x1, y1, 'chartreuse', 'dash', 3, 'Instability')
    plot_contours(fig1, min_petermann, x1, y1, 'white', 'dash', 3, 'Petermann Factor = 1')

    # add EPs and TPDs to plot
    plot_degeneracies(fig1, eps)
    plot_degeneracies(fig1, tpds)

    fig1.update_layout(
        xaxis_title= '𝛥̃ₖ',
        yaxis_title= '𝛥̃𝑓',
        xaxis_title_font=dict(size=30),
        yaxis_title_font=dict(size=30),
        legend=dict(
            bgcolor = 'lightgrey',
            orientation='h',
            yanchor='top',
            y = -0.2,
            xanchor='center',
            x = 0.5,
            font=dict(size=20)
            ),
        margin=dict(t=20),
        height=650,
    )

    st.plotly_chart(fig1, use_container_width=True)
    
    # -----------------------------------------
    # plot peak splitting of primary TPD and EP
    # set default parameters
    J = 1.0
    f_c = 0.0

    # find location of primary TPD
    primary_tpd = None
    for degen in tpds:
        if degen.degeneracy_type.name == "PRIMARY_TPD":
            primary_tpd = degen
            break
    dk_tpd = primary_tpd.Delta_tilde_kappa
    df_tpd = primary_tpd.Delta_tilde_f

    # find location of primary EP
    primary_ep = None
    for degen in eps:
        if degen.degeneracy_type.name == "PRIMARY_EP":
            primary_ep = degen
            break
    dk_ep = primary_ep.Delta_tilde_kappa
    df_ep = primary_ep.Delta_tilde_f

    # create different x axes depending on phi
    if phi == 0:
        x2_label = '𝛥̃ₖ'
        x2 = np.linspace(dk_ep - 1, dk_ep + 2.5, 2000)
    elif phi == np.pi:
        x2_label = '𝛥̃𝑓'
        x2 = np.linspace(df_ep - 1, df_ep + 2.5, 2000)
        delta_kappa = 0
    else:
        x2_label = '𝛥̃ₖ'
        x2 = np.linspace(dk_ep - 0.5, -0.18, 2000)

    # set up nu and lambda
    nu_plus = np.full_like(x2, np.nan)
    nu_minus = np.full_like(x2, np.nan)
    nu_0 = np.full_like(x2, np.nan)
    lambda_plus = np.full_like(x2, np.nan)
    lambda_minus = np.full_like(x2, np.nan)
    lambda_0 = np.full_like(x2, np.nan)

    # calculate EP and TPD peak locations for each x value
    instability_val = None
    instability_x_candidates = []
    previous_instability = None
    for i, x in enumerate(x2):
        if phi == np.pi:
            df = x
            dk = delta_kappa
        else:
            df = (2 * np.sin(phi)) / x
            dk = x
        
        tpd_result = peak_location(J, f_c, kappa_tilde_c, df, dk, phi)
        ep_result = -1 * eigenvalues(J, f_c, kappa_tilde_c, df, dk, phi).imag
        
        if len(tpd_result) == 2:
            nu_plus[i] = tpd_result[0]
            nu_minus[i] = tpd_result[1]
        else:
            nu_0[i] = tpd_result[0]

        if len(ep_result) == 2:
            lambda_plus[i] = ep_result[0]
            lambda_minus[i] = ep_result[1]
        else:
            lambda_0[i] = ep_result[0]
        
        # compute instability condition
        delta_lambda = np.sqrt(-df**2 + 2 * df * dk * 1j + dk**2 - 4 * J * np.exp(1j * phi))
        current_instability = delta_lambda.real - (kappa_tilde_c - dk)
        
        # detect sign change
        if previous_instability is not None:
            if np.sign(current_instability) != np.sign(previous_instability):
                # Store candidate (delta_f or delta_kappa depending on phi)
                instability_val_candidate = df if phi == np.pi else dk
                instability_x_candidates.append(instability_val_candidate)

        previous_instability = current_instability

    # find instability trace
    target_val = df_ep if phi == np.pi else dk_ep
    if instability_x_candidates:
        instability_val = min(instability_x_candidates, key=lambda val: abs(val - target_val))
    else:
        instability_val = None

    # calculate max and min y-value for vertical traces
    yvals = np.concatenate([lambda_plus, lambda_minus, lambda_0, nu_plus, nu_minus, nu_0])
    ymin = np.nanmin(yvals)
    ymax = np.nanmax(yvals)

    # create plot for EP and TPD peak splitting
    fig2 = go.Figure()
    plot_trace(fig2, x2, nu_plus, 'ν±', 'nu_pm', True, dict(width=4, color='purple'))
    plot_trace(fig2, x2, nu_minus, 'ν±', 'nu_pm', False, dict(width=4, color='purple'))
    plot_trace(fig2, x2, nu_0, 'ν±', 'nu_pm', False, dict(width=4, color='purple'))
    plot_trace(fig2, x2, lambda_plus, '|Im(λ±)|', 'lambda_pm', True, dict(width=4, color='black', dash='dash'))
    plot_trace(fig2, x2, lambda_minus, '|Im(λ±)|', 'lambda_pm', False, dict(width=4, color='black', dash='dash'))
    plot_trace(fig2, x2, lambda_0, '|Im(λ±)|', 'lambda_pm', False, dict(width=4, color='black', dash='dash'))

    # get values for the EP and TPD traces
    if phi == np.pi:
        tpd_val = df_tpd
        ep_val = df_ep
    else:
        tpd_val = dk_tpd
        ep_val = dk_ep

    # plot the vertical lines
    plot_trace(fig2, [tpd_val, tpd_val], [ymin, ymax], 'TPD', None, True, dict(width=4, color='cyan'))
    plot_trace(fig2, [ep_val, ep_val], [ymin, ymax], 'EP', None, True, dict(width=4, color='red'))
    plot_trace(fig2, [instability_val, instability_val], [ymin, ymax], 'Instability', None, (instability_val != None), dict(width=4, color='chartreuse'))

    fig2.update_layout(
        title=f"Primary EP and TPD Peak Splitting",
        title_font = dict(size=25),
        xaxis_title = x2_label,
        yaxis_title = "Frequency [arb.]",
        xaxis_title_font = dict(size=30),
        yaxis_title_font = dict(size=30),
        # height=500,
        legend=dict(font=dict(size=25)),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Relevant equations
    st.subheader("Relevant Equations:")
    st.markdown('<p style="font-size:18px;">Eigenvalues of the dynamical matrix:</p>', unsafe_allow_html=True)
    st.latex(r"""
    \tilde{\lambda}_0 = \left( \frac{\tilde{\Delta}_\kappa}{2} - \frac{\tilde{\kappa}_c}{2} \right)
    + i \left( \frac{\tilde{\Delta}_f}{2} - \tilde{f}_c + \tilde{f}_d \right)
    """)
    st.latex(r"""
    \tilde{\Delta}_\lambda = \sqrt{
    - \tilde{\Delta}_f^2
    + 2 i \tilde{\Delta}_f \tilde{\Delta}_\kappa
    + \tilde{\Delta}_\kappa^2
    - 4 e^{i \phi}
    }
    """)
    st.latex(r"""
    \tilde{\lambda}_\pm = \tilde{\lambda}_0 \pm \frac{\tilde{\Delta}_\lambda}{2}
    """)
    st.markdown('<p style="font-size:18px;">Petermann Factor:</p>', unsafe_allow_html=True)
    st.latex(r"""
    \bar{K}_2 = \frac{ \tilde{\Delta}_f^2 + \tilde{\Delta}_\kappa^2 + |\tilde{\Delta}_\lambda|^2 + 4 }{ 2 |\tilde{\Delta}_\lambda|^2 }
    """)
    st.markdown('<p style="font-size:18px;">TPD Equations:</p>', unsafe_allow_html=True)
    st.latex(r"""
    0 = f^3 + \tilde{p} f + \tilde{q},
    """)
    st.latex(r"""
    \tilde{p} = \frac{(\tilde{\kappa}_c - \tilde{\Delta}_\kappa)^2 + \mathrm{Re}(\tilde{\Delta}_\lambda^2)}{4},
    """)
    st.latex(r"""
    \tilde{q} = \frac{(\tilde{\kappa}_c - \tilde{\Delta}_\kappa) \, \mathrm{Im}(\tilde{\Delta}_\lambda^2)}{8}.
    """)

    # citation
    st.markdown("""Carney, A. S., Salcedo-Gallo, J. S., Bedkihal, S. K., & Fitzpatrick, M. (2025). 
            Unification of Exceptional Points and Transmission Peak Degeneracies in a Highly 
            Tunable Magnon-Photon Dimer. arXiv preprint arXiv:2506.09141.""")
    
if __name__ == "__main__":
    main()