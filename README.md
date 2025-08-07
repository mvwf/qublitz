# QuBlitz Virtual Qubit Simulator
Welcome! 
## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview
Qublitz is a simulation tool aimed to lower the barrier to entry for students in high school and college trying to learn about Quantum Engineering. All simulations on Qublitz demonstrate the behavior of a single, two-level qubit that occupies states between 1 and 0. The engine behind every mode in Qublitz is `quantum_simulator.py`, which takes in given parameters to simulate the behavior of a single, two-level qubit.

Qublitz currently includes the following apps:

### Home
The landing page with a welcome message and navigation to all available apps.

### Sonify Images
This app allows you to turn images into sound by mapping image data to audio signals. Upload your own image or use a premade one, and explore the sonification process interactively.

### Qubit Simulator (Free Play)
The Qubit Simulator is a sandbox that grants users access to most parameters necessary to simulate any qubit behavior they want. There are two simulation modes: time domain and frequency domain. In the time domain, you can set qubit properties and send microwave square pulses to see how the qubit reacts on the Bloch sphere and in probability graphs. In the frequency domain, you can observe the responsiveness of a qubit to a range of different driving frequencies.

### Custom Qubit Query
This app allows users to select from predefined qubit parameters (e.g., for classroom or assignment use) and run simulations with those settings.

### IQ Mixing
This app demonstrates IQ mixing, a key concept in quantum control and signal processing, allowing users to explore how in-phase (I) and quadrature (Q) signals combine to drive a qubit.

### Exceptional Point and Transmission Peak Degeneracy Exploration
This app provides interactive tools to explore exceptional points and transmission peak degeneracy in quantum systems, with visualizations and parameter controls.

## Installation 

### Libraries
Qublitz is hosted as a Streamlit app. The relevant libraries for installation are as follows:
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `Pillow`
- `soundfile`
- `qutip` (for the quantum simulator)
Each page will also need to import `run_quantum_simulation` from `quantum_simulator`.

## Usage
Find the current website here:
https://qublitz-qubit-lab.streamlit.app/
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

## Contributing 
### General Steps 
To start contributing, fork the repository and clone the fork so you can run your version of Qublitz locally. Then, create a new branch `git checkout -b feature-branch-name` name the branch based on the feature you are working on. Eg. `git checkout -b gates-challenge`. Once your changes are made, commit them and push to branch. Finally open the pull request. 
### Specific Projects 
For the Gates Challenge, you should take the existing `2_gates_challenge.py` page and modify it. You may rewrite all of the code if you like but the gates page contains starter code for displaying the Bloch Sphere and showing how X and Y-Gates work. 
For the Step Pulse Challenge, you should take the existing `3_gates_challenge.py` page and modify it.  You may rewrite all of the code if you like but the gates page contains starter code for displaying the Bloch Sphere and input for creating a Step Pulse. 

## Contact 
For help or advice email: 
neo.y.cai.25@dartmouth.edu 
mattias.w.fitzpatrick@dartmouth.edu

