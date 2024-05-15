# QuBlitz Virtual Qubit Simulator
## High Level Summary 
### Overview 
Qublitz is a simulation tool aimed to lower the barrier to entry for students in high school and college trying to learn about Quantum Engineering. All simulations on Qublitz demonstrate the behavior of a single, two level qubit that occupies states between 1 and 0. Qublitz contains two simulation modes, a free play mode and a guided challenge mode. 
### Free Play Mode 
Free play mode grants users access to most parameters necessary to simulate any qubit behavior they want to. There are two simulation modes in Free Play, the time domain and frequency domain. 
In the time domain, the user can determine both the qubit's properties and send a micorwave square pulse to see how it reacts. The user will first determine the frequency of the qubit and the frequency of the driving pulse, then the simulation duration, Rabi Rate, T1, T2, and finally the number of shots. Then, the user can add microwave pulses to influence the qubit by selecting a target channel to send the pulse, the pulse amplitude, and determine the duration through the start and stop time. The user can add multiple square pulses to the simulation, then run the simulation. The pulse shape and duration is displayed on the "Time-Dependent Amplitudes" graph. Once "Run Simulation" is clicked, the response of the qubit to its subjected square pulses (over time) is displayed in 3 graphs: a grpah for the expected values of the qubit state, a graph for the measured probability of the qubit state, and finally a graph of the qubit on the bloch sphere. See bloch sphere explanation for a tldr. 

Find the website here:

https://qublitz-qubit-lab.streamlit.app/

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)
