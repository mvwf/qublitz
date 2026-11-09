/* concepts.js: the fixed vocabulary of the Concept Map (web/concepts.html), tied to the crash-course modules (course.html).
 * A map is a set of links between these ids, each with one relation from RELATIONS, and nothing else: students never type,
 * so a shared map cannot carry personal information. tools/outreach-collector/Code.gs keeps the same id list (CONCEPT_IDS)
 * and refuses anything else; web/shared/tests/outreach.test.js checks the two lists match.
 *
 * Never rename or reuse an id (stored maps refer to it); add new concepts at the end of their module.
 * module = course module 1–10; depth = who meets it (see the Depth section of web/shared/outreach.js).
 */
(function (root) {
  'use strict';
  const CONCEPTS = [
    // 1 · Waves and modes
    { id: 'mode', label: 'Mode', module: 1, def: 'A standing-wave pattern that vibrates at one frequency of its own, like one harmonic of a guitar string.' },
    { id: 'frequency', label: 'Frequency', module: 1, def: 'How many times per second something repeats, in hertz (Hz).' },
    { id: 'resonance', label: 'Resonance', module: 1, def: 'A big response when something is driven at one of its own natural frequencies.' },
    { id: 'fourier', label: 'Fourier decomposition', module: 1, depth: 'student+', def: 'Writing any signal as a sum of sine waves of different frequencies.' },
    { id: 'spectrogram', label: 'Spectrogram', module: 1, def: 'A picture of sound: time across, frequency up, loudness as colour.' },
    { id: 'doppler', label: 'Doppler effect', module: 1, def: 'A moving source sounds higher-pitched as it approaches and lower as it moves away.' },
    // 2 · Quantization
    { id: 'quantization', label: 'Quantization', module: 2, def: 'A confined wave can only have certain energies, not any value in between.' },
    { id: 'energy-level', label: 'Energy level', module: 2, def: 'One of the allowed energies of an atom or circuit.' },
    { id: 'photon', label: 'Photon', module: 2, def: 'One packet of light or microwave energy, E = hf.' },
    { id: 'spectral-line', label: 'Spectral line', module: 2, def: 'Light at one sharp frequency, given off or absorbed when a system jumps between two energy levels.' },
    { id: 'schrodinger', label: 'Schrödinger equation', module: 2, depth: 'student+', def: 'The wave equation of quantum mechanics: how a quantum state changes in time.' },
    // 3 · The qubit
    { id: 'qubit', label: 'Qubit', module: 3, def: 'A quantum bit: a system with two states, |0⟩ and |1⟩, that can also be in a superposition of both.' },
    { id: 'superposition', label: 'Superposition', module: 3, def: 'A state that is a combination α|0⟩ + β|1⟩ of two states at once.' },
    { id: 'bloch-sphere', label: 'Bloch sphere', module: 3, def: 'A sphere on which every state of one qubit is a point: |0⟩ at the top, |1⟩ at the bottom.' },
    { id: 'measurement', label: 'Measurement', module: 3, def: 'Reading a qubit: it gives 0 or 1 at random, with probabilities set by the state, and leaves it in the answer.' },
    { id: 'gate', label: 'Quantum gate', module: 3, def: 'An operation on qubits; on one qubit it is a rotation of the Bloch sphere.' },
    { id: 'phase', label: 'Phase', module: 3, depth: 'student+', def: 'The angle around the Bloch sphere’s equator: the relative sign or angle between α and β.' },
    // 4 · Control and coherence
    { id: 'rabi', label: 'Rabi oscillation', module: 4, def: 'A qubit driven at its frequency swings back and forth between |0⟩ and |1⟩.' },
    { id: 'detuning', label: 'Detuning', module: 4, depth: 'student+', def: 'How far the drive frequency is from the qubit frequency.' },
    { id: 'pi-pulse', label: 'π pulse', module: 4, def: 'A drive pulse just long enough to flip |0⟩ to |1⟩.' },
    { id: 't1', label: 'T₁ (energy relaxation)', module: 4, def: 'How long an excited qubit takes to lose its energy and fall back to |0⟩.' },
    { id: 't2', label: 'T₂ (dephasing)', module: 4, depth: 'student+', def: 'How long a superposition keeps a definite phase; at most 2T₁.' },
    { id: 'decoherence', label: 'Decoherence', module: 4, def: 'The loss of quantum behaviour as a system leaks information to its surroundings.' },
    { id: 'ramsey', label: 'Ramsey experiment', module: 4, depth: 'student+', def: 'Two π/2 pulses with a wait between them: the fringes measure detuning and T₂* (free-induction decay); an echo gives T₂.' },
    // 5 · The hardware
    { id: 'superconductivity', label: 'Superconductivity', module: 5, def: 'Below a critical temperature some metals carry current with no resistance.' },
    { id: 'josephson', label: 'Josephson junction', module: 5, def: 'Two superconductors separated by a thin insulator: a lossless, nonlinear circuit element.' },
    { id: 'transmon', label: 'Transmon', module: 5, depth: 'student+', def: 'A superconducting qubit made from a Josephson junction and a capacitor.' },
    { id: 'anharmonicity', label: 'Anharmonicity', module: 5, depth: 'engineer+', def: 'Unequal spacing of energy levels, which lets you address |0⟩ ↔ |1⟩ without exciting |2⟩.' },
    { id: 'resonator', label: 'Resonator', module: 5, def: 'A circuit that rings at one frequency, used to store microwave photons and to read qubits out.' },
    { id: 'quality-factor', label: 'Quality factor (Q)', module: 5, def: 'How slowly a resonator loses energy: its frequency divided by its linewidth. A higher Q rings for longer.' },
    { id: 'dilution-fridge', label: 'Dilution refrigerator', module: 5, def: 'A cryostat that cools chips to about 10 millikelvin.' },
    { id: 'thermal-noise', label: 'Thermal noise', module: 5, def: 'Random photons from anything warm, which can disturb a qubit.' },
    { id: 'attenuator', label: 'Attenuator', module: 5, depth: 'student+', def: 'A component that weakens a signal, and with it the noise coming down from warmer stages.' },
    // 6 · Readout
    { id: 'dispersive-readout', label: 'Dispersive readout', module: 6, depth: 'student+', def: 'Reading a qubit through a resonator whose frequency shifts slightly depending on the qubit’s state.' },
    // 7 · Two qubits
    { id: 'entanglement', label: 'Entanglement', module: 7, def: 'A shared state of two or more qubits that cannot be described one qubit at a time.' },
    // 8 · Error correction
    { id: 'no-cloning', label: 'No-cloning theorem', module: 8, def: 'An unknown quantum state cannot be copied.' },
    { id: 'parity-check', label: 'Parity check', module: 8, def: 'A test of whether a group of bits or qubits agrees, without reading their values.' },
    { id: 'syndrome', label: 'Syndrome', module: 8, def: 'The pattern of failed parity checks, which points to the error.' },
    { id: 'logical-qubit', label: 'Logical qubit', module: 8, def: 'One protected qubit stored across many physical qubits.' },
    { id: 'code-distance', label: 'Code distance', module: 8, depth: 'student+', def: 'The fewest errors that can change a code’s stored information without being noticed.' },
    { id: 'decoder', label: 'Decoder', module: 8, depth: 'student+', def: 'The algorithm that guesses the most likely error from a syndrome.' },
    { id: 'threshold', label: 'Threshold', module: 8, def: 'The error rate below which making a code bigger makes it better.' },
    // 9 · Bosonic qubits
    { id: 'harmonic-oscillator', label: 'Harmonic oscillator', module: 9, def: 'A system with equally spaced energy levels, like a mass on a spring or an LC circuit.' },
    { id: 'cat-state', label: 'Cat state', module: 9, depth: 'engineer+', def: 'A superposition of two distinct oscillator states, used to store a protected qubit.' },
    { id: 'gkp', label: 'GKP code', module: 9, depth: 'researcher', def: 'A qubit stored in a grid of oscillator states, so small shifts can be measured and corrected.' },
  ];
  const RELATIONS = [
    { id: 'related', label: 'is related to' },
    { id: 'causes', label: 'causes' },
    { id: 'part-of', label: 'is part of' },
    { id: 'measured-by', label: 'is measured by' },
    { id: 'example-of', label: 'is an example of' },
    { id: 'limits', label: 'limits' },
    { id: 'enables', label: 'makes possible' },
  ];
  const MODULE_NAMES = ['', 'Waves and modes', 'Quantization', 'The qubit', 'Control and coherence', 'The hardware', 'Reading out a qubit',
    'Two qubits', 'Error correction', 'Bosonic qubits', 'Capstone'];
  const api = { CONCEPTS, RELATIONS, MODULE_NAMES };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QL_CONCEPTS = api;
})(typeof window !== 'undefined' ? window : globalThis);
