"""

quget.py
---------

Version 1.0.0\n
Author: Max Weiner\n
PI: Mattias Fitzpatrick\n
Institution: Thayer School of Engineering at Dartmouth College\n

This module provides plots for simulations using mesolve.
"""

# Import Packages
from qutip import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.fft import fftshift,fft,ifft
from time import *
from scipy.optimize import curve_fit

#######################################################################

# Constants
hbar=1.05e-34

#########################################################################

def dispget(
omega_q: float,  # Qubit frequency (Grad/s)
g: float,       # Coupling strength (Grad/s)
Omega_f: float,  # Drive amplitude
omega_r: float,  # Resonator frequency (Grad/s)
gamma_q: float,    # Qubit decay rate (GHz)
kappa_r: float,      # Resonator decay rate (GHz)
tlist: np.array,   # List of time
olist: np.array,   # List of frequencies
mode: str, # Qubit is either excited or ground
N: int, # Maximum number of fock states
psi0 = None, # Initial state
plot_drive = True, # Turn on if you want to see the plot used to calculate dshift

):
    """
    Inputs
    -------
    omega_q: Qubit frequency (Grad/s)\n
    g: Coupling strength (Grad/s)\n
    Omega_f: Drive amplitude\n
    omega_r: Resonator frequency (Grad/s)\n
    gamma_q: Qubit decay rate (GHz)\n
    kappa_r: Resonator decay rate (GHz)\n
    tlist: List of time (s)\n
    olist: List of frequencies (Hz)\n
    N: Maximum fock state\n
    psi0: Initial state\n
    plot_drive: Boolean, determines whether a plot is shown\n

    Outputs
    -------
    Outputs list with the following elements:\n

    dr: Resonator frequency, shifted from bare frequency as a result of the drive\n
    re: 2D array showing resonator transmission\n
    t:  Outputs amount of time each mesolve call takes

    """
    # Creation and annihilation operators for qubit.
    # NOTE: convention is flipped so the ground state has an expectation
    # value of 0 and excited state has expectation value of 1
    sigma_minus=tensor(sigmap(),qeye(N))
    sigma_plus=tensor(sigmam(),qeye(N))

    # Creation and annihilation operators for the resonator
    bos_mode_d=tensor(qeye(2),destroy(N))
    bos_mode_c=tensor(qeye(2),create(N))

    # Annihilation operator for the resonator
    a = tensor(qeye(2), destroy(N))

    # Array of time, will fill up with times it takes for each mesolve to run
    t=[]

    # Will be a 2D array of resonator transmission
    re=[]

    # Counter to keep track of simulation progress
    i=0

    # This loop will simulate a resonator drive sweep
    for omega_d in olist:
        # time it takes to start simulation
        ti=time()

        # converts drive from dBm to Hz
        Omega_d=np.sqrt((kappa_r*10**((Omega_f-30)/10))/(hbar*omega_q/2*np.pi))

        # Drive terms
        def epsilon_t(t, args):
            return Omega_d*np.exp(-1j*omega_d*t)
        def epsilon_t_2(t, args):
            return Omega_d*np.exp(1j*omega_d*t)
        
        H_qubres = omega_q * sigma_minus*sigma_plus + omega_r * a * a.dag() # Qubit and Resonator hamiltonian
        H_int = g * (sigma_plus * a.dag() + sigma_minus * a) # Interaction hamiltonian

        # Total Hamiltonian with driving term
        H = [H_qubres+H_int,[a.dag(),epsilon_t],[a,epsilon_t_2]]

        # Initial state
        if mode=='excited':
            psi0 = tensor(basis(2,1),basis(N,0))
        elif mode=='ground':
            psi0 = tensor(basis(2,0),basis(N,0))

        # Collapse operators
        c_ops = [np.sqrt(gamma_q) * sigma_minus, 
                 np.sqrt(kappa_r) * a]

        # Limits amoutnt of steps and allows for states to be stored
        o=Options(store_states=True,nsteps=5000)

        # Simulate system and store states and expectation values
        result = mesolve(H=H, rho0=psi0, tlist=tlist, c_ops=c_ops, e_ops=[a.dag()* a],options=o)

        # append resonator transmission to resonator expectation value
        re.append(result.expect[0])

        # Print progress
        i+=1
        if len(olist) > 1:
            print(f"{i}/{len(olist)} simulations run")
        
        # convert resonator expectation list to 2D array for plotting
        ex=np.array(re)

        # calculate time taken for simulation and store value
        tf=time()
        t.append(tf-ti)

    # We only plot transmission curve if specified in the arguments
    if plot_drive == True:

        # on the top, plot the resonator transmission curve over time
        fig=plt.figure(figsize=(12, 6))
        X,Y=np.meshgrid(np.array(tlist),np.array(olist))
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        surf = ax.plot_surface(X,Y/(2*np.pi),
                            ex, 
                            cmap="coolwarm",
                            linewidth=0, 
                            antialiased=False, 
                            label="Transmission [arb.]")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$\omega_r$ (Hz)")
        ax.set_zlabel("Transmission [arb.]")

        # on the bottom, plot the dispersive shift curve after
        # drive has been applied for specified amount of time.
        # In other words, this is the trace @ t = tlist[-1]

        # Final dispersive shift plot
        dsweep = np.swapaxes(ex,0,1)[-1]

        # Frequency at omega_r-chi
        omega_shift = float(np.array(olist)[dsweep==max(dsweep)][0])

        # Dispersive Shift (MHz)
        chi = (omega_shift - omega_r)/(2*np.pi*10**6)
        # Plot
        ax = fig.add_subplot(2,1,2)
        ax.plot(np.array(olist)/(2*np.pi),dsweep)
        ax.set_xlabel("$\omega_r$ (Hz)")
        ax.set_ylabel("Transmission [arb.]")
        ax.axvline(x = omega_r/(2*np.pi), color = 'r', linestyle = '--')
        ax.axvline(x = omega_shift/(2*np.pi), color = 'k', linestyle = '--')
        ax.axhline(y = float(max(dsweep)), color = 'g', linestyle = '--', label = f'$\chi$ = {round(chi,3)} MHz')
        ax.legend()
        plt.show()

    # find dispersive shift
    vals=np.swapaxes(ex,0,1)
    dr = olist[np.where(vals[-1]==max(vals[-1]))[0][0]]

    # Print expected steady state
    print(Omega_d/kappa_r)
    
    # return list of number, 2D list and 1D list
    return [dr,re,t]

#########################################################################

def time_evolution(
omega_q: float,  # Qubit frequency (Grad/s)
g: float,       # Coupling strength (Grad/s)
Omega_f: float,  # Drive amplitude
omega_r: float,  # Resonator frequency (Grad/s)
gamma_q: float,    # Qubit decay rate (GHz)
kappa_r: float,      # Resonator decay rate (GHz)
tlist: np.array,   # List of time
omega_d: float,   # List of frequencies
N: int,
psi0:Qobj,
):
    """
    This function uses mesolve to simulate a time evolution for a specific drive.

    Inputs
    -------
    omega_q: Qubit frequency (Grad/s)\n
    g: Coupling strength (Grad/s)\n
    Omega_f: Drive amplitude\n
    omega_r: Resonator frequency (Grad/s)\n
    gamma_q: Qubit decay rate (GHz)\n
    kappa_r: Resonator decay rate (GHz)\n
    tlist: List of time (s)\n
    omega_d: frequency of drive we want to test\n
    N: Maximum fock state\n
    psi0: Initial state\n
    plot_drive: Boolean, determines whether a plot is shown\n

    Outputs
    --------
    resshift: fin at desired drive frequency [arb.]

    """
    y=dispget(omega_q,g,Omega_f,omega_r,gamma_q,kappa_r,tlist,[omega_d],mode='custom',N=N,psi0=psi0,plot_drive=False)


    resshift=y[1][0]

    # Uncomment to see time evolution at drive frequency
    # plt.figure(figsize=(12, 6))
    # plt.plot(tlist,resshift,color="r")
    # plt.xlabel("Qubit Drive Time")
    # plt.ylabel("Excited State Magnitude")
    # plt.show()

    return resshift

#########################################################################

def rabical(
omega_q: float,  # Qubit frequency (Grad/s)
g: float,       # Coupling strength (Grad/s)
Omega_q: float,  # Drive amplitude
Omega_r: float,  # Drive amplitude
omega_r: float,  # Resonator frequency (Grad/s)
gamma_q: float,    # Qubit decay rate (GHz)
kappa_r: float,      # Resonator decay rate (GHz)
tlist: np.array,   # List of time
omega_d: float,   # List of frequencies
ts: np.array,   # Array of qubit drive times
N: int,     # Maximum number of fock states
):
    """
    This function drives the qubit for many different times, then looks at the 
    dispersive shift plots to see the state of the qubit. The maximum
    value of the peak representing the excited state is at the time required
    to do a pi pulse, so this is effectively a pi pulse calibration.
    
    Inputs
    -------
    omega_q: Qubit frequency (Grad/s)\n
    g: Coupling strength (Grad/s)\n
    Omega_q: Qubit drive amplitude (dBm)\n
    Omega_r: Resonator drive amplitude\n
    omega_r: Resonator frequency (Grad/s)\n
    gamma_q: Qubit decay rate (GHz)\n
    kappa_r: Resonator decay rate (GHz)\n
    tlist: List of time (s)\n
    omega_d: frequency of drive we want to test\n
    ts: Array of times which we drive the qubit for\n
    N: Maximum fock state\n

    Outputs
    --------
    tpi: Time it takes to drive the qubit in order to do the pi pulse\n
    f: fit for qubit expectation value
    """

    # list of peak resonator values at each qubit drive time
    respow=[]

    # loop through each qubit drive
    for ttest in ts:

        # tells user how long they are driving the qubit for
        print(f'testing qubit drive time #{ttest}')

        # Converting drive amplitudes from dBm to Hz
        Omega_dq=np.sqrt((gamma_q*10**((Omega_q-30)/10))/(hbar*omega_q/2*np.pi))
        Omega_dr=np.sqrt((kappa_r*10**((Omega_r-30)/10))/(hbar*omega_r/2*np.pi))

        # create and destroy operators for the qubit
        sigma_minus=tensor(sigmap(),qeye(N))
        sigma_plus=tensor(sigmam(),qeye(N))

        # destroy operator for the resonator
        a = tensor(qeye(2), destroy(N))

        # time dependent drive terms
        def epsilon_t(t, args):
            return Omega_dq*np.exp(-1j*omega_q*t)
        def epsilon_t_2(t, args):
            return Omega_dq*np.exp(1j*omega_q*t)
        
        H_qubres = omega_q * sigma_plus * sigma_minus + omega_r * a.dag() * a # Qubit and Resonator hamiltonian
        H_int = g * (sigma_plus * a + sigma_minus * a.dag()) # Interaction hamiltonian

        # Total hamiltonian
        H = [H_qubres+H_int,[sigma_plus,epsilon_t],[sigma_minus,epsilon_t_2]]

        # initialize qubit to ground state
        psi0 = tensor(basis(2,0),basis(N,0))

        # Lindblad operators
        c_ops = [np.sqrt(gamma_q) * sigma_minus, np.sqrt(kappa_r) * a]

        # Qubit simulation time array
        tlist2=np.linspace(0, ttest, 1000)

        # Enable looking at resonator states
        o=Options(store_states=True)

        # Simulate qubit drive
        result = mesolve(H=H, rho0=psi0, tlist=tlist2, c_ops=c_ops, e_ops=[],options=o)

        # Simulate resonator drive and append peak resonator values
        respow.append(time_evolution(omega_q,g,Omega_r,omega_r,gamma_q,kappa_r,tlist,omega_d,N,psi0=result.states[-1])[-1])

    # plot drive time vs dispersive shift peak. The maximum value
    # of this plot tells us how long we need to drive the qubit in
    # order to do a pi pulse
    plt.figure(figsize=(12, 6))
    plt.plot(ts,respow,color="r")
    plt.xlabel("Qubit Drive Time")
    plt.ylabel("Excited State Magnitude")
    plt.show()

    
    # Return time required to do a pi pulse
    excited=max(respow)
    tpi=ts[np.where(respow==excited)[0][0]]
    return [tpi,excited]

#########################################################################

def powercal(
    omega_q: float,  # Qubit frequency (Grad/s)
    g: float,       # Coupling strength (Grad/s)
    Omega_q: float,  # Qubit drive amplitude
    Omega_r: float,  # Resonator drive amplitude
    omega_r: float,  # Resonator frequency (Grad/s)
    gamma_q: float,    # Qubit decay rate (GHz)
    omega_d: float,   # Drive Frequency
    kappa_r: float,      # Resonator decay rate (GHz)
    tqubit: float,         # time required for pi pulse
    tres: float,         # time to simulate resonator
    N: int,             # Maximum fock number
):
    """
    Calculates the state of the qubit based on the state of the resonator,
    drives the qubit for the amount of time required for a pi pulse and drives
    the resonator to do readout. Then, lets system precess and finds T1.

    Inputs
    -------
    omega_q: Qubit frequency (Grad/s)\n
    g: Coupling strength (Grad/s)\n
    Omega_q: Qubit drive amplitude (dBm)\n
    Omega_r: Resonator drive amplitude (dBm)\n
    omega_r: Resonator frequency (Grad/s)\n
    gamma_q: Qubit decay rate (GHz)\n
    kappa_r: Resonator decay rate (GHz)\n
    tlist: List of time (s)\n
    omega_d: frequency of drive we want to test (Hz)\n
    tqubit: Amount of time which we drive the qubit for (s)\n
    tres: Amount of time which we drive the resonator for (s)\n
    N: Maximum fock state\n

    Outputs
    --------
    ss: steady state value for excited state of qubit, tells us if we need to increase N\n

    """

    # Qubit drive times
    Omega_dq=np.sqrt((gamma_q*10**((Omega_q-30)/10))/(hbar*omega_q/2*np.pi))
    Omega_dr=np.sqrt((kappa_r*10**((Omega_r-30)/10))/(hbar*omega_d/2*np.pi))

    # Create and Destroy operators for the qubit, with conventions flipped
    sigma_minus=tensor(sigmap(),qeye(N))
    sigma_plus=tensor(sigmam(),qeye(N))

    # Annihilation operator for qubit
    a = tensor(qeye(2), destroy(N))

    ########### STEP 1: DRIVE THE QUBIT #################
    # time dependent qubit drive
    def epsilon_t(t, args):
        return Omega_dq*np.exp(-1j*omega_q*t)
    def epsilon_t_2(t, args):
        return Omega_dq*np.exp(1j*omega_q*t)
    

    H_qubres = omega_q * sigma_plus * sigma_minus + omega_r * a.dag() * a # Qubit and Resonator hamiltonian
    H_int = g * (sigma_plus * a + sigma_minus * a.dag()) # Interaction hamiltonian

    # Total Hamiltonian
    H = [H_qubres+H_int,[sigma_plus,epsilon_t],[sigma_minus,epsilon_t_2]]

    # Always initialize qubit to ground state
    psi0 = tensor(basis(2,0),basis(N,0))

    # Lindblad operators
    c_ops = [np.sqrt(gamma_q) * sigma_minus, np.sqrt(kappa_r) * a]

    # time it takes to do pi pulse
    tlist=np.linspace(0, tqubit, 1000)

    # array representative of qubit drive
    fd = np.ones(len(tlist))

    # enable storing states
    o=Options(store_states=True)

    # simulate qubit drive
    result = mesolve(H=H, rho0=psi0, tlist=tlist, c_ops=c_ops, e_ops=[sigma_plus*sigma_minus,a.dag()* a],options=o)

    ############### STEP 2: DRIVE THE RESONATOR ##############

    # time for responator drive
    tlist2=np.linspace(0, tres, int(len(tlist)*tres/tqubit))

    # time dependent drive term
    def epsilon_t(t, args):
        return Omega_dr*np.exp(-1j*omega_d*t)
    def epsilon_t_2(t, args):
        return Omega_dr*np.exp(1j*omega_d*t)
    
    # full hamiltonian
    H = [H_qubres+H_int,[a.dag(),epsilon_t],[a,epsilon_t_2]]

    # array which tells us the resonator drive is on
    td = np.ones(len(tlist2))

    # pick up where we left off
    psi0=result.states[-1]

    # Simulate drive with resonator
    resultt = mesolve(H, psi0, tlist2, c_ops, [sigma_plus*sigma_minus,a.dag()* a],options=o)

    # turn drives off, let qubit and resonator dissipate
    H = [H_qubres+H_int]

    ########## STEP 3: TURN DRIVES OFF #################

    # Time to let qubit-resonator system dissipate
    tlist3=np.linspace(0, 10*tqubit, 10000)

    # qubit/resonator is not driven
    fod = np.zeros(len(tlist3))

    # qubit drive list
    qdrive=list(np.ones(len(fd)))+list(np.zeros(len(td)))+list(fod)

    # resonator drive list
    rdrive=list(np.zeros(len(fd)))+list(td)+list(fod)

    # pick up where we left off
    psi0=resultt.states[-1]

    # simulate system with no drive
    resulttt = mesolve(H, psi0, tlist3, c_ops, [sigma_plus*sigma_minus,a.dag()* a],options=o)

    # resonator fock number
    r=list(result.expect[1])+list(resultt.expect[1])+list(resulttt.expect[1])

    # qubit fock number
    q=list(result.expect[0])+list(resultt.expect[0])+list(resulttt.expect[0])

    # time array, made up from previous steps
    t = np.linspace(0, tlist[-1]+tlist2[-1]+tlist3[-1], len(tlist)+len(tlist2)+len(tlist3))

    # plot whole system
    plt.plot(t,qdrive,color="blue")
    plt.plot(t,rdrive, color="green")
    plt.plot(t,(q*np.ones(len(r))),color="black")
    plt.plot(t,r*np.ones(len(r)),color="red")
    plt.xlabel('Time (s)')
    plt.ylabel("Expectation Value")
    plt.title("Resonator Response")
    plt.legend(["Qubit Drive","Resonator Drive","Qubit Excitation Value", "Resonator Fock Number"])
    plt.show()

    ############### T1 FIT
    plt.scatter(tlist3,resulttt.expect[1],color="blue",s=2)
    plt.xlabel('Time (s)')
    plt.ylabel("Transmission [arb.]")
    plt.title("Decaying Resonator")

    # Do polyfit of log data, then undo to get T1 parameters.
    p = np.polyfit(tlist3, np.log(resulttt.expect[1]), 1)
    a = np.exp(p[1])
    b = p[0]
    x_fitted = np.linspace(np.min(tlist3), np.max(tlist3), 100)
    y_fitted = a * np.exp(b * x_fitted)
    plt.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
    plt.show()

    # show parameters in correct form
    print(f'{a}e^({b}x)')

    # Uncomment to find approximate rabi rate via fft
    # resf = np.abs(fftshift(fft(resultt.expect[1])))
    # nsteps=len(tlist2)
    # d = max(tlist2)/nsteps
    # f = np.linspace(-1/(2*d),1/(2*d),nsteps)

    # Calculate expected steady state value of resonator in excited state
    ss = Omega_dr/kappa_r

    return ss

#########################################################################

def phaseplot(
    omega_q: float,  # Qubit frequency (Grad/s)
    g: float,       # Coupling strength (Grad/s)
    Omega_q: float,  # Qubit drive amplitude
    Re: float,       # Real component of drive
    Im: float,       # Imaginary component of drive
    Omega_r: float,  # Resonator drive amplitude
    omega_r: float,  # Resonator frequency (Grad/s)
    gamma_q: float,    # Qubit decay rate (GHz)
    olist: float,   # List of frequencies
    kappa_r: float,      # Resonator decay rate (GHz)
    ts: float,         # time required for pi pulse
    N: int
):
    """
    Finds phase through resonator as a function of frequency.

    Inputs
    -------
    omega_q: Qubit frequency (Grad/s)\n
    g: Coupling strength (Grad/s)\n
    Omega_q: Qubit drive amplitude (dBm)\n
    Re: Real component of drive\n
    Im: Imaginary component of drive\n
    Omega_r: Resonator drive amplitude\n
    omega_r: Resonator frequency (Grad/s)\n
    gamma_q: Qubit decay rate (GHz)\n
    kappa_r: Resonator decay rate (GHz)\n
    olist: List of frequencies(Hz)\n
    ts: Array of time to drive the qubit for (s)\n
    N: Maximum fock state\n

    Outputs
    --------
    None. Prints plot of phase as a function of resonator drive frequency.
    """
    # Qubit drive times
    Omega_dq=np.sqrt((gamma_q*10**((Omega_q-30)/10))/(hbar*omega_q/2*np.pi))
    Omega_dr=np.sqrt((kappa_r*10**((Omega_r-30)/10))/(hbar*omega_r/2*np.pi))

    # Create and Destroy operators for the qubit, with conventions flipped
    sigma_minus=tensor(sigmap(),qeye(N))
    sigma_plus=tensor(sigmam(),qeye(N))

    # Annihilation operator for qubit
    a = tensor(qeye(2), destroy(N))

    # I and Q portions of resonator readout
    I = []
    Q = []
    mag = []

    # iterator, keeps of track of how many simulations have been run
    ii=0

    ############# STEP 1A: DRIVE ALONG I ############

    # Time dependent drive term
    for omega_d in olist:
        def epsilon_t(t, args):
            return Omega_dq*(Re*np.exp(-1j*omega_q*t))
        def epsilon_t_2(t, args):
            return Omega_dq*(Re*np.exp(1j*omega_q*t))

        H_qubres = omega_q * sigma_plus * sigma_minus + omega_r * a.dag() * a # Qubit and Resonator hamiltonian
        H_int = g * (sigma_plus * a + sigma_minus * a.dag()) # Interaction hamiltonian

        # Total Hamiltonian
        H = [H_qubres+H_int,[sigma_plus,epsilon_t],[sigma_minus,epsilon_t_2]]

        # Initialize Qubit to ground state
        psi0 = tensor(basis(2,0),basis(N,0))

        # Lindblad Operators
        c_ops = [np.sqrt(gamma_q) * sigma_minus, np.sqrt(kappa_r) * a]

        # Simulation time array:
        tlist=np.linspace(0, ts, 1000) # I Drive
        tlist2=np.linspace(0, ts, 1000) # Q Drive
        tlist3=np.linspace(0, ts, 1000) # Resonator Drive
        # Store states
        o=Options(store_states=True)

        # Simulate qubit-resonator system with drive along I quadrature
        result = mesolve(H=H, rho0=psi0, tlist=tlist, c_ops=c_ops, e_ops=[tensor(sigmax(),qeye(N)),tensor(sigmay(),qeye(N)),tensor(sigmaz(),qeye(N))],options=o)
        #b = Bloch()
        #b.vector_color=['#FF0000','#A0FF00','#0052FF']
        #b.point_color=hexgradient(b=b.vector_color[1],e=b.vector_color[2],npoints=len(tlist3))
        #bpc=hexgradient('#FFFFFF',b.vector_color[0],len(tlist))+hexgradient(b.vector_color[0],b.vector_color[1],len(tlist2))+hexgradient(b.vector_color[1],b.vector_color[2],len(tlist3))

        ############# STEP 1B: DRIVE ALONG Q ############
        def epsilon_t(t, args):
            return Omega_dq*(Im*np.exp(-1j*(omega_q*t+np.pi/2)))
        def epsilon_t_2(t, args):
            return Omega_dq*(Im*np.exp(1j*(omega_q*t*np.pi/2)))
        
        H_qubres = omega_q * sigma_plus * sigma_minus + omega_r * a.dag() * a # Qubit and Resonator hamiltonian
        H_int = g * (sigma_plus * a + sigma_minus * a.dag()) # Interaction hamiltonian

        # Total Hamiltonian
        H = [H_qubres+H_int,[sigma_plus,epsilon_t],[sigma_minus,epsilon_t_2]]

        # Initialize Qubit to ground state
        psi1 = result.states[-1]

        #b.add_states(psi1.ptrace(0))

        # Store states
        o=Options(store_states=True)

        # Simulate qubit-resonator system with drive along I quadrature
        result2 = mesolve(H=H, rho0=psi1, tlist=tlist2, c_ops=c_ops,e_ops = [tensor(sigmax(),qeye(N)),tensor(sigmay(),qeye(N)),tensor(sigmaz(),qeye(N))],options=o)

        ################# STEP 2: DRIVE RESONATOR, FIND I TERM #################

        # Phi 
        rphi = np.arctan(Im/Re)

        # Drive Term
        def epsilon_t(t, args):
            return Omega_dr*(Im*np.exp(-1j*(omega_d*t+rphi)))
        def epsilon_t_2(t, args):
            return Omega_dr*(Im*np.exp(-1j*(omega_d*t+rphi)))
        
        # Total Hamiltonian
        H = [H_qubres+H_int,[a.dag(),epsilon_t],[a,epsilon_t_2]]

        # Pick up where we left off
        psi2=result2.states[-1]
        #b.add_states(psi2.ptrace(0))

        # Simulate with resonator drive
        result3 = mesolve(H=H, rho0=psi2, tlist=tlist3, c_ops=c_ops,e_ops= [sigma_plus*sigma_minus,a,a.dag(),a.dag()*a,
                                                                            tensor(sigmax(),qeye(N)),tensor(sigmay(),qeye(N)),tensor(sigmaz(),qeye(N))],options=o)
        #b.add_states(result3.states[-1].ptrace(0))
        #b.add_points([list(result3.expect[4]),list(result3.expect[5]),list(result3.expect[6])])
        # b.add_points([list(result.expect[0])+list(result2.expect[0])+list(result3.expect[4]),
        #               list(result.expect[1])+list(result2.expect[1])+list(result3.expect[5]),
        #               list(result.expect[2])+list(result2.expect[2])+list(result3.expect[6])])
        # b.point_color=list(reversed(bpc))
        # b.show()
        #q = input("quit? y for yes or n for no")

        #if q == 'y':
        #    exit()

        # Readout I portion of transmission
        I.append(result3.expect[1][-1])
        Q.append(result3.expect[2][-1])
        mag.append(result3.expect[3][-1])
        # ############# STEP 3: SIMULATE Q DRIVE ################

        # # Drive term with pi/2 added to complex exponential
        # def epsilon_t(t, args):
        #     return Omega_dq*np.exp(-1j*omega_q*t-np.pi*0.5)
        # def epsilon_t_2(t, args):
        #     return Omega_dq*np.exp(1j*omega_q*t-np.pi*0.5)
        
        # # Total Hamiltonian
        # H = [H_qubres+H_int,[sigma_plus,epsilon_t],[sigma_minus,epsilon_t_2]]

        # # Time list for Q qubit drive
        # tlist3=np.linspace(0, ts/2, 5000)

        # # Pick up where we left off
        # psi0=result2.states[-1]

        # # Simulate with Q drive
        # result3 = mesolve(H, psi0, tlist3, c_ops, [sigma_plus*sigma_minus,a.dag()* a],options=o)

        # ########### STEP 4: DRIVE RESONATOR TO FIND Q TERM #######################

        # Time to simulate resonator
        # tlist3=np.linspace(0, 5*ts, 10000) # Drive term

        # # Drive Term
        # def epsilon_t_3(t, args):
        #     return Omega_dr*np.exp(-1j*omega_d*t)
        # def epsilon_t_4(t, args):
        #     return Omega_dr*np.exp(1j*omega_d*t)
        
        # # Total Hamiltonian
        # H = [H_qubres+H_int,[a.dag(),epsilon_t_3],[a,epsilon_t_4]]

        # # Pick up where we left off
        # psi0=result3.states[-1]

        # # Simulate with resonator drive
        # result4 = mesolve(H, psi0, tlist3, c_ops, [sigma_plus*sigma_minus,a.dag()* a],options=o)

        # # Record Q
        # Q.append(result4.expect[1][-1])

        # Record progress
        print(f'{ii}/{len(olist)}')

        # Iterate
        ii+=1
    
    # Plot phase over drive frequency to see dispersive shift
    fig=plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(olist/(2*np.pi),np.array(I),color="b")
    ax1.plot(olist/(2*np.pi),np.array(Q), color = "k")
    ax1.plot(olist/(2*np.pi),np.array(mag), color = "r")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Transmission [arb.]")
    ax1.legend(["I","Q","Fock Number"])
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(olist/(2*np.pi),np.arctan(np.array(Q)/np.array(I))/np.pi,color="r")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase/$\pi$")

    plt.show()


    ##############################################################

#########################################################################

def plot_wigner(W, xvec, ax, title):
    """
    This function plots a wigner function for a given state
    -------
    Inputs
    -------
    W: Wigner state
    xvec: X axes of wigner function
    title: Title of plot

    -------
    Outputs
    -------
    None. Shows a wigner function
    """
    # Contour levels
    levels = np.linspace(W.min(), W.max(), 100)
    line_levels = np.linspace(W.min(), W.max(), 5)  # Fewer levels for clarity in contour lines
    
    # Plot the filled contour
    cf = ax.contourf(xvec, xvec, W, levels=levels, cmap='RdBu_r')
    
    # Contour lines with fading for positive values and dashed for negatives
    # Split the levels for positive and negative
    pos_levels = [level for level in line_levels if level >= 0]
    neg_levels = [level for level in line_levels if level < 0]
    
    # Add contour lines for negative values with dashed style
    ax.contour(xvec, xvec, W, levels=neg_levels, colors='black', linestyles='dashed', linewidths=0.5)
    # Add contour lines for positive values with reduced opacity
    ax.contour(xvec, xvec, W, levels=pos_levels, colors='black', alpha=0.2, linewidths=0.5)
    
    # Set aspect of the plot to be equal
    ax.set_aspect('equal')
    
    # Adding color bar
    norm = Normalize(vmin=W.min(), vmax=W.max())
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, format="%.2f", fraction=0.046, pad=0.04)
    cbar.set_label(r'$W(\beta)$')
    
    # Labeling and styling
    ax.set_xlabel(r'$\rm{Re}(\beta)$', fontsize=20)
    ax.set_ylabel(r'$\rm{Im}(\beta)$', fontsize=20)
    ax.set_title(title, fontsize=22)

#########################################################################

def hexgradient(
    b:str, # beginning color hex
    e:str, # end color hex
    npoints: int # number of points in array
):
    """
    This function plots a wigner function for a given state

    Inputs
    -------
    b: beginning color for hex gradient array
    e: end color for hex gradient array
    npoints: number of elements in hex gradient array

    Outputs
    -------
    hg: Hex gradient
    """
    hex_colorb = b.lstrip('#')

    # Convert to integer values
    rb = int(hex_colorb[0:2], 16)
    gb = int(hex_colorb[2:4], 16)
    bb = int(hex_colorb[4:6], 16)

    hex_colore = e.lstrip('#')

    # Convert to integer values
    re = int(hex_colore[0:2], 16)
    ge = int(hex_colore[2:4], 16)
    be = int(hex_colore[4:6], 16)

    ra = np.linspace(rb,re,npoints).astype(int)
    ga = np.linspace(gb,ge,npoints).astype(int)
    ba = np.linspace(bb,be,npoints).astype(int)

    hg=[]

    for i in range(npoints):
        hg.append('#%02x%02x%02x' % tuple([ra[i],ga[i],ba[i]]))

    return hg

#########################################################################