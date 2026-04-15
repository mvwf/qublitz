'''
Authors:
    M. Weiner
Release Date: 
    V 1.0: 10/6/2024

'''
import numpy as np
from streamlit import *
import streamlit as st
from streamlit.components.v1 import html
from PIL import Image
import plotly.graph_objects as go

def main():
    title('Quantum Measurement Tutorial')
    qublitz_logo = Image.open("images/qublitz.png")
    st.sidebar.image(qublitz_logo)
    logo = Image.open("images/logo.png") 
    st.sidebar.image(logo) # display logo on the side 
    st.sidebar.markdown('<div style="text-align:center;"><a href="https://sites.google.com/view/fitzlab/home" target="_blank" style="font-size:1.2rem; font-weight:bold;">FitzLab Website</a></div>', unsafe_allow_html=True)
    
    markdown('''A qubit (short for quantum bit) is the basic unit of quantum information similar to a classical bit in traditional computing but with quantum mechanical properties. Unlike a classical bit, which can be in one of two states, 0 or 1, a qubit can exist in a superposition of both states simultaneously. This means it can be in a state represented as:''')

    latex(r'''\ket{\psi} = \alpha\ket{0} + \beta\ket{1} = \alpha\begin{bmatrix}
    1\\
    0
    \end{bmatrix}

    +\beta\begin{bmatrix}
    0\\
    1
    \end{bmatrix} = \begin{bmatrix}
    \alpha\\
    \beta
    \end{bmatrix}''')
    latex(r'''\text{Where } |0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \text{ and } |1\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}''') # basis vectors

    markdown('''where $\\alpha$ and $\\beta$ are complex numbers that describe the probability amplitudes of the qubit being in the ground or excited state. The sum of the squares of these amplitudes must equal 1:''')

    latex(r'|\alpha|^2 + |\beta|^2 = 1')

    markdown('''
    There are certain tools that we can use to understand the state of a qubit. In particular, we use the concept 
    of the expectation value of a qubit to view it's excitation, as well as it's phase. We do this with the
    pauli operators:
    ''')

    latex(r'\sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}, \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}')

    markdown('''
    The concept of an expectation value applied to these operators will tell us where the state is on the bloch sphere:
    ''')

    latex(r'\braket{\sigma_x} = \bra{\psi}\sigma_x\ket{\psi} = \begin{bmatrix} \alpha^* & \beta^*\end{bmatrix}\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} \alpha \\ \beta \end{bmatrix} = \beta^*\alpha+\alpha^*\beta')
    latex(r'\braket{\sigma_y} = \bra{\psi}\sigma_y\ket{\psi} = \begin{bmatrix} \alpha^* & \beta^*\end{bmatrix}\begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}\begin{bmatrix} \alpha \\ \beta \end{bmatrix} = i\beta^*\alpha-i\alpha^*\beta')
    latex(r'\braket{\sigma_z} = \bra{\psi}\sigma_z\ket{\psi} = \begin{bmatrix} \alpha^* & \beta^*\end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}\begin{bmatrix} \alpha \\ \beta \end{bmatrix} = \alpha^*\alpha-\beta^*\beta')

    markdown('''
    Using a phase $\phi = \\arctan{\\frac{\\braket{\sigma_y}}{\\braket{\sigma_x}}}$ and $\\braket{\sigma_z}$, we can visualize any qubit state on the bloch sphere:)
    ''')


    header('Visualizing a State on the Bloch Sphere')
    exp = slider('$<\sigma_z>$', -1.0 ,1.0 , 1.0, 0.01)
    phi = slider('$\phi$ (rad)', 0.0,2*np.pi, 0.0, 0.01)
    rho = (exp-1)*np.pi/2
    #s = gates.rz(rho)*gates.rx(phi)*basis(2,0)
    u, v = np.mgrid[0:2*np.pi:20j, np.pi:0:10j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    x_exp = -np.cos(phi) * np.sin(rho)
    y_exp = -np.sin(phi) * np.sin(rho)
    z_exp = np.cos(rho)

    fig_bloch = go.Figure(data=[
    go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.3, colorscale= 'RdBu'),
    go.Scatter3d(
        x=[0,x_exp],  # ⟨σ_x⟩ values
        y=[0,y_exp],  # ⟨σ_y⟩ values
        z=[0,z_exp],  # ⟨σ_z⟩ values
        mode = 'lines',
        line=dict(
        color='black',
        width=5
    )),
    go.Scatter3d(
        x=[x_exp],  # ⟨σ_x⟩ values
        y=[y_exp],  # ⟨σ_y⟩ values
        z=[z_exp],  # ⟨σ_z⟩ values
        mode = 'markers',
            marker=dict(
                size=4,
                opacity=1,
                color='black',
                symbol='x'
            )  # Use time-based colors
        )
    ])


    fig_bloch.update_layout(
        title='State Vector on the Bloch Sphere',
        scene=dict(
            xaxis_title=dict(text="σ_x"),
            yaxis_title=dict(text="σ_y"),
            zaxis_title=r'σ_z',
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1])
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig_bloch.add_trace(go.Scatter3d(
        x=[0, 0], 
        y=[0, 0], 
        z=[1.0, -1.0], 
        mode='text',
        text=["|0⟩", "|1⟩"],
        textposition=["middle center","middle center"],
        textfont=dict(
            color=["black", "black"],
            size=20
        )
    ))

    #fig_bloch.update_layout(scene=dict(zaxis=dict(autorange="reversed")))
    plotly_chart(fig_bloch)
    markdown('''
        While this interactive excersize shows that you can represent a qubit's state on a bloch sphere at
        one moment, the coefficients $\\alpha$ and $\\beta$ are almost always changing over time. In fact,
        quantum physicists use Schrödinger's equation to find these coefficients as a function of time:
        '''
    )

    latex(r'\hat{H}\ket{\psi} = i\hbar\frac{d}{dt}\ket{\psi}')

    markdown('''
    Where H is the Hamiltonian term which contains the total energy of the system. This setup results in
    a set of first order differential equations, and we will see in the following section of this tutorial
    that the solutions of these equations will behave like an oscillator with real and imaginary parts.
    ''')
    header('Quantum Mechanics in Practice: Experimental Setup')
    markdown('''
    In quantum mechanics, measurement typically collapses the qubit's state. 
    To minimize this disturbance, a **quantum non-demolition** approach is preferred.
    A resonator helps perform a QND measurement by coupling indirectly to the qubit, 
    allowing the qubit state to influence the properties of the resonator without 
    directly collapsing the qubit state immediately.
    

    We can use a cavity with an infinite, discrete number of photons (energy levels) like the one shown below
    as a resonator:
    ''')

    image("images/Zurich_Cavity.png",caption = "Quantum Technology & Computing. IBM Research - Zurich, Quantum technology & computing. (2023). https://www.zurich.ibm.com/st/quantum/index.html" )

    markdown('''
            This resonator, coupled to the qubit with a Josephson Junction, can read out the qubit's state if we time
            evolve the whole system. We can show a schematic of the equivalent circuit with the qubit in black, 
            resonator in light blue, and the Josephson Junction in red:''')
    
    image("images/QRC.png",caption = "Kerman, Andrew. (2013). Quantum information processing using quasiclassical electromagnetic interactions between qubits and electrical resonators. New Journal of Physics. 15. 10.1088/1367-2630/15/12/123011.")

    markdown('''
            This circuit is used to map the discrete infinite energy levels of a resonator to the lowest two energy levels of a Qubit, or a
            **Two Level System (TLS)**. Since the both the Qubit and the Resonator
            has multiple energy storing elements (Capacitance and Inductance), we can think of them
            as harmonic oscillators with a cosine potential:
             ''')
    
    image("images/cospotential.png",caption = "Aerts, Diederik & Beltran, Lester. (2020). Quantum Structure in Cognition: Human Language as a Boson Gas of Entangled Words. Foundations of Science. 25. 10.1007/s10699-019-09633-4. ")

    markdown(''' 
    In the measurement of Superconducting Qubits, the oscillator is actually anharmonic (commonly referred to as Morse), which means that the potential
    is not perfectly quadratic and the discrete energy levels are not evenly spaced. The potential is similar to a damped spring with some separation **x** and energy levels $T_n$:
''')
    
    image("images/anharmonic.png",caption = "Jang, S. (2023). Harmonic Oscillator and Vibrational Spectroscopy. In S. J. Jang (Ed.), Quantum Mechanics for Chemistry. Springer International Publishing. https://doi.org/10.1007/978-3-031-30218-3_3")

    markdown(''' 
    Because the superconducting qubit has infinite discrete energies, we must take some extra measures to keep the qubit is kept in the first two energy levels, which we can call
    $\\ket{0}$ and $\\ket{1}$. The anharmonicity in a Qubit actually makes it easier to isolate these two energy levels, and tuning certain parameters will
    reduce the probability of the Qubit transitioning to a state with two (or more) photons, $\\ket{2}$. There are also optimally shaped pulses, such as **Gaussian** or **DRAG (Derivative Removal by Adiabatic Gate)** pulses, which help reduce transitions to higher energy levels. 
    These shaped pulses minimize the off-resonant driving that can excite the qubit to unwanted energy levels.
''')
    
    header('Quantum Operators and Linear Algebra')
    markdown('''
             We must take into account the following terms which describe the Hamiltonian. 
             These terms are described using the raising and lowering operators for a qubit:
             ''')
    
    latex(r'\sigma_+ = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \text{, and } \sigma_- = \begin{bmatrix}0 & 0 \\ 1 & 0 \end{bmatrix}')

    markdown('The raising ($a^{\dagger}$) and lowering ($a$) operators for a resonator with **N** possible levels are as follows:')

    latex(r'''\text{for} \ket{N-1} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1\end{bmatrix},
    a^{\dagger} = \begin{bmatrix} 
    0 & 0 & 0 & 0 & \dots & 0 & 0\\
    1 & 0 & 0 & 0 & \dots & 0 & 0\\
    0 & \sqrt{2} & 0 & 0 & \dots & 0 & 0\\
    0 & 0 & \sqrt{3} & 0 & \dots & 0 & 0\\
    \vdots & \vdots & \vdots & \dots & \ddots & \vdots & \vdots\\
    0 & 0 & 0 & 0 & \dots & 0 & 0\\
    0 & 0 & 0 & 0 & \dots & \sqrt{N-1} & 0
    \end{bmatrix} \text{ and }''')
    
    latex(r'''
    a = \begin{bmatrix}
    0 & 1 & 0 & 0 & \dots & 0 & 0\\
    0 & 0 & \sqrt{2} & 0 & \dots & 0 & 0\\
    0 & 0 & 0 & \sqrt{3} & \dots & 0 & 0\\
    0 & 0 & 0 & 0 & \dots & 0 & 0\\
    \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
    0 & 0 & 0 & 0 & \dots & 0 & \sqrt{N-1}\\
    0 & 0 & 0 & 0 & \dots & 0 & 0\\
    \end{bmatrix}''')

    markdown('''
              The creation operator increases the energy level of your state by 1,
              and the annihilation operator conversely decreases your energy level by 1. For any photon number n:
              ''')
    
    latex(r'a^{\dagger}\ket{n}=\sqrt{n+1}\ket{n+1} \text{, and } a\ket{n} = \sqrt{n}\ket{n-1}')

    markdown('''
            One last thing before we begin modelling our system - because we are working with a two-level
            qubit coupled with an N-level resonator, we cannot describe our system willy nilly with these operators,
            because the dimensions do not match. Therefore, we must use tensor products to describe our system so
            that the math works out. Here's an example where we make an annihilation operator for the resonator:
             ''')
    
    latex(r'''a = \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23} \\
    a_{31} & a_{32} & a_{33}
    \end{bmatrix}, \quad
    I = \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} = \begin{bmatrix} 1 & 0\\0 & 1\end{bmatrix}''')
    latex(r'''a_{qubres} = a \otimes I = \begin{bmatrix}
    a_{11} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} & a_{12} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} & a_{13} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} \\
    a_{21} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} & a_{22} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} & a_{23} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} \\
    a_{31} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} & a_{32} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix} & a_{33} \begin{bmatrix}
    I_{11} & I_{12} \\
    I_{21} & I_{22}
    \end{bmatrix}
    \end{bmatrix}
    \text{. This results in a 6 by 6 matrix.}''')

    markdown('''
    As we can see, the dimensions get really large really fast, and especially considering we need a photon
    level of 15 or higher for the resonator to see reliable results, this means we are solving a system
    of at least 30 (complicated) differential equations when we use qutip to find a solution.
             ''')
    
    header('Theoretical Qubit-Resonator Model')

    markdown('Starting off, we have Hamiltonian terms which describe the dynamics of the qubit and resonator by themselves:')

    latex(r'H_q = \hbar\omega_q\frac{\sigma_z}{2}\text{ , and } H_r = \hbar\omega_ra^{\dagger}{a}')

    markdown('''Where $\omega_q$ and $\omega_r$ are the frequencies which the qubit and the resonator are built to oscillate at.
             Next, we have a term which we call the "Jaynes-Cummings Hamiltonian". This term describes
             the interaction between the qubit and resonator created by the coupling with two superconductors are placed in proximity and with some barrier or restriction between them,
             commonly referred to as a **Josephson Junction**:''')
    
    latex(r'H_{JC} = g\hbar(a^{\dagger}\sigma_-+a\sigma_+)')

    markdown(''' Where g is the coupling strength between the qubit and the resonator.
             We also have a drive term which can be described as follows:''')

    latex(r'\text{For a resonator, } H_{dr} = \hbar\epsilon(a^{\dagger}e^{-i\omega_dt}+ae^{i\omega_dt})')
    latex(r'\text{For a qubit, } H_{dq} = \hbar\epsilon(\sigma_+e^{-i\omega_qt}+\sigma_-e^{i\omega_qt})')
    latex(r'\text{Where } \epsilon = I+iQ')

    markdown('''
             Where $\omega_d$ is a drive frequency that we use to apply to the resonator, and $\epsilon$
             is some complex drive with a real part **I** and an imaginary part **Q**. We use these Hamiltonian terms to 
             solve the Schrödinger equation:
             ''')

    latex(r'H\ket{\psi} = i\hbar\frac{d}{dt}\ket{\psi}')
    latex(r'H = H_q + H_r + H_{JC} + H_{dq} \text{ if driving the qubit}')
    latex(r'H = H_q + H_r + H_{JC} + H_{dr} \text{ if driving the resonator}')
    latex(r'H = H_q + H_r + H_{JC} \text{ with no drive}')

    markdown('''
            In order for us to change the state of a qubit, effectively changing the information we are encoding it with,
            we must drive the qubit. This means that we solve the Schrödinger Equation with the qubit drive term in the Hamiltonian
            in order to find the state after a certain time. Immediately after this, we *readout* the state of the Qubit.
            To tell what the qubit's state is, we need to probe the resonator by driving around the
            resonator frequency, $\omega_d \\approx \omega_r$. Doing this will yield some shift, **$\chi = \\frac{g^2}{\delta} = \\frac{g^2}{\omega_r-\omega_q}$** in the frequency of the
            complex output signal through the resonator, $I^2+Q^2$, which will look like a Lorenzian when we drive the resonator long enough for the transmission to
            reach steady state.
             ''')
    
    header('Resonator Drive Sweep Simulation')
    
    markdown('''
            The following is a simulation where you can alter the parameters in the Hamiltonian and see what the output signal of the
            resonator will look like:
             ''')

    col1,col2 = columns(2)
    with col1:
        g = slider('$g (MHz)$', 10.0 ,100.0 , 50.0, 0.1)
        kappa = slider('$\kappa_r (MHz)$', 3.0 ,8.0 , 3.8, 0.01)
    with col2:
        wr = slider('$\\frac{\omega_r}{2\pi} (GHz)$', 6.1 ,8.0 ,7.0, 0.01)
        wq = slider('$\\frac{\omega_q}{2\pi} (GHz)$', 2.5 ,6.0 , 4.0, 0.01)

    g = g*10**-3
    kappa=kappa*10**-3
    wd = np.linspace(-0.02,0.02,1000)
    chi = (g**2)/(wr-wq)
    dispp = 0.5*kappa/((wd-chi)**2 + (0.5*kappa)**2)
    dispm = 0.5*kappa/((wd+chi)**2 + (0.5*kappa)**2)

    fig_disp = go.Figure(data=[
    go.Scatter(x=wd, y=dispp, mode = 'lines', name = '|g>', marker = dict(color = 'blue')),
    go.Scatter(x=wd, y=dispm, mode = 'lines', name = '|e>', marker = dict(color = 'red'))])

    fig_disp.add_scatter(x=[0],
                y=[dispp[int(len(dispp)/2)]],
                marker=dict(
                    color='black',
                    size=10
                ),
               mode = 'markers',
               name='Output at ωd=ωr')
    fig_disp.update_xaxes(range=[-0.02, 0.02])
    fig_disp.add_vline(x=0, line_width=3, line_dash="dash", line_color="green")
    # add annotation
    fig_disp.update_layout(title='Transmission Through Resonator',xaxis_title='(ωd-ωr)/2π (GHz)',yaxis_title='Amplitude [arb.]',
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

    phig = np.arctan((5/kappa)*((wd-chi)))
    phie = np.arctan((5/kappa)*((wd+chi)))
    fig_phase = go.Figure(data=[
    go.Scatter(x=wd, y=phig, mode = 'lines', name = '|g>', marker = dict(color = 'blue')),
    go.Scatter(x=wd, y=phie, mode = 'lines', name = '|e>', marker = dict(color = 'red'))])
    
    fig_phase.update_xaxes(range=[-0.02, 0.02])
    fig_phase.add_vline(x=0, line_width=3, line_dash="dash", line_color="green")
    fig_phase.add_scatter(x=[0,0],
                y=[phig[int(len(phig)/2)],phie[int(len(phie)/2)]],
                marker=dict(
                    color='black',
                    size=10
                ),
                mode = 'markers',
               name='Phase at ωd=ωr')
    # add annotation
    fig_phase.update_layout(title='Phase of Resonator Output',xaxis_title='(ωd-ωr)/2π (GHz)',yaxis_title='Φ/π (rad)',
                            legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
    with col1:
        plotly_chart(fig_disp)
    with col2:
        plotly_chart(fig_phase)

    markdown('''
        As we can see, the resonator's output takes the shape of a lorenzian around $\omega_r+\chi$ if the qubit is in the ground
        state, and $\omega_r-\chi$ If the qubit is excited. This is how we can use the resonator to tell what state
        the qubit is in! However, sweeping the drive around the resonator can take a very long time, and it is much faster
        if we only drive the resonator at the bare frequency to tell the state of the qubit. The problem with this is that
        the magnitude of the output signal of the resonator is exactly the same at the bare frequency whether the qubit
        is in the ground state or the excited state. This is why we need to take into account the phase $\\arctan{\\frac{Q}{I}}$ of this
        output signal:
    ''')

    markdown('''
        From the phase plot, we can see that driving the resonator at it's bare frequency will get us different phases
        based on the state of the qubit! Therefore, when we are measuring the qubit, we should
        look at the phase of the complex signal through the resonator when we drive it to do qubit readout.
    ''')

if __name__ == "__main__":
    main()
