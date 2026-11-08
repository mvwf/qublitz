/* qubit-lab.js: learning material for the Qublitz Virtual Qubit Lab (qublitz/qubit-simulator.html).
 * Every answer below was checked against qubit-engine.js: drive term 2π·Ω_R·σx·cos(2π·ω_d·t), so the
 * on-resonance Rabi frequency is Ω_R/2π and t_π = 1/(2·Ω_R/2π); P(|1⟩) after a π pulse decays as e^(−t/T₁);
 * Ramsey fringes oscillate at |ω_d − ω_q| with envelope e^(−τ/T₂); mystery qubits have T₂ = 2T₁. */
if (window.QL) QL.learning({
  id: 'qubit-lab',
  title: 'Qublitz Virtual Qubit Lab',
  tagline: 'You get a qubit whose frequency, Rabi rate and lifetime are hidden. Measure them the way a ' +
    'superconducting-qubit lab does: spectroscopy, a Rabi oscillation, a T₁ decay and a Ramsey experiment.',
  fun: 'it is a detective game with a real simulator behind it: every qubit code is a new mystery, the data are noisy ' +
    'like a real measurement, and the checklist tells you when you have cracked it.',
  objectives: [
    { id: 'spectroscopy', text: 'Find a qubit frequency ω<sub>q</sub> from the peak of a drive-frequency sweep, and explain why off-resonant Rabi oscillations are faster and shallower.' },
    { id: 'pi_pulse', text: 'Compute the π and π/2 pulse lengths from a fitted Rabi rate Ω<sub>R</sub>.' },
    { id: 't1', text: 'Extract the energy-relaxation time T₁ from the decay of P(|1⟩) after a π pulse.' },
    { id: 'ramsey', text: 'Use the Ramsey fringe frequency to correct ω<sub>q</sub>, and read the coherence time T₂ from the fringe envelope (T₂ ≤ 2T₁). This model has no slow noise, so the Ramsey envelope gives T₂; in a real lab it gives T₂*, which is shorter, and a spin echo recovers T₂.' },
  ],
  howTo: [
    'Keep <b>Mystery qubit</b> selected (or open the link your instructor gave you). The code names your qubit.',
    '<b>Frequency Domain</b> tab: press <b>Run frequency sweep</b>. The green curve peaks at ω<sub>d</sub> ≈ ω<sub>q</sub>. Press <b>Zoom in on the peak</b> until the trace below swings all the way to P(|1⟩) ≈ 1, and note the fitted Rabi rate and π-pulse time.',
    '<b>Time Domain</b> tab: press <b>Use sweep peak</b>, then <b>+ π pulse</b>. Watch the Bloch vector flip from |0⟩ to |1⟩ and decay back. The red dotted curve is the T₁ fit.',
    '<b>Ramsey Sequence</b> tab: choose <b>Fitted Rabi rate</b> for the π/2 length and press <b>Run Ramsey chevron</b>. Click the chevron to pick a vertical cut and read the fringe frequency and T₂.',
    'Check yourself with <b>Reveal parameters</b> at the bottom of the Ramsey tab (hidden on links your instructor assigned), or try <b>Free play</b> to set the qubit yourself.',
  ],
  goals: [
    'Tick all four items in the lab checklist: find the resonance, flip the qubit with a π pulse, measure T₁ within 20%, and pin ω<sub>q</sub> to within 1 MHz with Ramsey.',
    'Then reveal the parameters and compare them with your fits.',
  ],
  quiz: [
    { id: 'offres', objective: 'spectroscopy',
      q: 'You drive the qubit a little off resonance (detuning Δ ≠ 0) instead of exactly at ω<sub>q</sub>. Compared with driving on resonance, the Rabi oscillation of P(|1⟩) is…',
      choices: ['Slower, but it still reaches P(|1⟩) = 1', 'Faster, and it no longer reaches P(|1⟩) = 1', 'The same speed, just with a smaller swing', 'Gone: nothing happens unless ω<sub>d</sub> = ω<sub>q</sub> exactly'],
      answer: 1,
      explain: 'Off resonance the qubit oscillates at the generalized Rabi frequency √(Ω<sub>R</sub>² + Δ²), which is faster, and P(|1⟩) only reaches Ω<sub>R</sub>²/(Ω<sub>R</sub>² + Δ²). That is why the sweep peaks at ω<sub>q</sub>, and why you zoom in until the peak trace reaches 1 before trusting the fitted Ω<sub>R</sub>.' },
    { id: 'tpi', objective: 'pi_pulse',
      q: 'Your Rabi fit gives Ω<sub>R</sub>/2π = 250 MHz. How long is a π pulse (|0⟩ → |1⟩)?',
      choices: ['1 ns', '2 ns', '4 ns', '250 ns'],
      answer: 1,
      explain: 'One full Rabi period is 1/(250 MHz) = 4 ns and takes |0⟩ → |1⟩ → |0⟩. A π pulse is half a period: t<sub>π</sub> = 1/(2·Ω<sub>R</sub>/2π) = 2 ns. A π/2 pulse is 1 ns.' },
    { id: 't1read', objective: 't1',
      q: 'After a π pulse, P(|1⟩) starts near 1 and decays. What is P(|1⟩) one T₁ after the pulse ends?',
      choices: ['About 0.5, since T₁ is the half-life', 'About 0.37 (that is, 1/e)', 'About 0.63', 'Exactly 0'],
      answer: 1,
      explain: 'Energy relaxation is exponential: P(|1⟩) ≈ e<sup>−t/T₁</sup>, so at t = T₁ it is e<sup>−1</sup> ≈ 0.37. The half-life would be T₁·ln 2 ≈ 0.69 T₁.' },
    { id: 'sign', objective: 'ramsey',
      q: 'At one detuning cut of the chevron the drive is at 5.0000 GHz and the Ramsey fringes oscillate at 1.5 MHz. What do you learn about ω<sub>q</sub>/2π?',
      choices: ['It is 5.0015 GHz', 'It is 4.9985 GHz', 'It is either 4.9985 or 5.0015 GHz: the fringe frequency is the size of the detuning, not its sign', 'It is 5.0000 GHz: Ramsey fringes only appear on resonance'],
      answer: 2,
      explain: 'During the idle time the qubit phase drifts at |ω<sub>d</sub> − ω<sub>q</sub>|, and cos is the same for ±Δ. The page lists both candidates. The chevron settles it: its center (no fringes) sits at the true ω<sub>q</sub>.' },
    { id: 't2', objective: 'ramsey',
      q: 'A mystery qubit has T₁ = 150 ns and no extra (pure) dephasing. What envelope decay time T₂ should the Ramsey fit give?',
      choices: ['75 ns', '150 ns', '300 ns', 'Any value: T₂ is unrelated to T₁'],
      answer: 2,
      explain: 'Energy relaxation alone destroys the coherence at half its rate, so T₂ ≤ 2T₁, with T₂ = 2T₁ = 300 ns when there is no pure dephasing (the model the mystery qubits use). In Free play you can set T₂ lower to add pure dephasing.' },
  ],
});
