/* ep-tpd.js: learning material for Exceptional Points & TPDs (qublitz/ep-tpd.html).
 * Checked against ep-engine.js (units of J): λ± = λ0 ± Δλ/2 with Δλ² = −Δf² + 2iΔfΔκ + Δκ² − 4e^{iφ}, so the EP
 * (Δλ = 0) sits at Δκ = ∓2cos(φ/2), Δf = ∓2sin(φ/2) whatever κc is; Petermann K₂ = (Δf² + Δκ² + |Δλ|² + 4)/(2|Δλ|²)
 * is 1 at Δf = Δκ = 0 and diverges at the EP; at φ = 0, κc = 0.68 the primary TPD is at Δκ ≈ −1.03, and at
 * Δκ = −1.5 the transmission shows one peak while Im λ± = ±0.66; the primary EP-TPD distance shrinks with κc and is
 * 0.002 at φ = π, κc = 0.1; instability = Re λ+ · 2 > 0 means a growing mode. */
if (window.QL) QL.learning({
  id: 'ep-tpd',
  title: 'Exceptional Points & TPDs',
  tagline: 'Two coupled resonators (a magnon mode and a cavity photon) with loss and gain. Find where their two ' +
    'eigenmodes merge into one (an exceptional point, EP), and compare it with where the two peaks you would ' +
    'measure in transmission merge (a transmission-peak degeneracy, TPD). They are not the same place.',
  fun: 'it is a treasure hunt on a map from a 2026 research paper: two kinds of markers move as you turn two dials, ' +
    'and the challenge is to make them meet.',
  objectives: [
    { id: 'ep_def', text: 'Identify an exceptional point as the place where the two complex eigenvalues, and their modes, coalesce.' },
    { id: 'tpd', text: 'Distinguish a TPD (the transmission peaks merge) from an EP (the eigenvalues merge) on the splitting plot.' },
    { id: 'kc', text: 'Predict how changing the cavity damping κ̃<sub>c</sub> moves the TPD but leaves the EP where it is.' },
    { id: 'petermann', text: 'Relate the Petermann factor to how close the system is to an EP, and read the stable and unstable regions.' },
  ],
  howTo: [
    'The big map is drawn over two knobs of the dimer: the frequency detuning Δ̃<sub>f</sub> (up) and the damping difference Δ̃<sub>κ</sub> (across), both in units of the coupling J.',
    'Red × is the primary EP and red ○ is the primary TPD. Color is the Petermann factor (1 means two independent, orthogonal modes); gray means unstable, where one mode grows instead of decaying.',
    'Move the <b>Cavity damping κ̃<sub>c</sub></b> and <b>Coupling phase φ</b> sliders on the right and watch the markers move. The table lists their exact positions and the distance between the primary EP and TPD.',
    'The lower plot cuts through the primary EP: purple lines are the transmission peaks, black dashed lines are the eigenfrequencies |Im λ±|.',
    'Work through the challenges above the map. Each one ticks itself.',
  ],
  goals: [
    'Complete the three challenges: see which marker κ̃<sub>c</sub> moves, put both markers on the Δ̃<sub>f</sub> axis, and bring the TPD within 0.05 of the EP.',
    'Explain why a sensor that reads transmission peaks is not automatically sitting at the EP.',
  ],
  quiz: [
    { id: 'ep_def', objective: 'ep_def',
      q: 'What happens at an exceptional point (EP)?',
      choices: ['The two transmission peaks merge into one', 'The two complex eigenvalues become equal and the two modes merge into one', 'The two resonators stop being coupled', 'The system becomes unstable and starts to oscillate'],
      answer: 1,
      explain: 'At an EP, Δλ = 0: λ+ = λ−, frequency and damping both, and the eigenvectors coalesce as well. Merging transmission peaks is a different condition, the TPD.' },
    { id: 'kc_move', objective: 'kc',
      q: 'Keep φ fixed and raise the cavity damping κ̃<sub>c</sub>. What do the red markers do?',
      choices: ['Both stay put', 'The EP moves and the TPD stays put', 'The TPD moves and the EP stays put', 'Both move together, so they stay on top of each other'],
      answer: 2,
      explain: 'κ̃<sub>c</sub> shifts both eigenvalues equally (it is in λ<sub>0</sub>), so it drops out of Δλ and the EP stays at Δ̃<sub>κ</sub> = ∓2cos(φ/2), Δ̃<sub>f</sub> = ∓2sin(φ/2). The transmission peaks depend on κ̃<sub>c</sub>, so the TPD moves (away from the EP as κ̃<sub>c</sub> grows).' },
    { id: 'split', objective: 'tpd',
      q: 'At φ = 0 and κ̃<sub>c</sub> = 0.68, the EP is at Δ̃<sub>κ</sub> = −2 and the TPD at Δ̃<sub>κ</sub> ≈ −1.03. Between them (say Δ̃<sub>κ</sub> = −1.5), what does the splitting plot show?',
      choices: ['Two eigenfrequencies, but only one transmission peak', 'Two transmission peaks, but only one eigenfrequency', 'Two of each', 'One of each'],
      answer: 0,
      explain: 'Past the EP the eigenfrequencies split (±0.66 at Δ̃<sub>κ</sub> = −1.5), but the peaks are broad enough that transmission still shows a single peak until the TPD. A measured peak splitting therefore starts at the TPD, not at the EP.' },
    { id: 'petermann', objective: 'petermann',
      q: 'How does the Petermann factor K₂ behave as you approach the EP?',
      choices: ['It drops to 0', 'It stays at 1 everywhere', 'It grows without bound', 'It equals 1 exactly at the EP'],
      answer: 2,
      explain: 'K₂ = (Δ̃<sub>f</sub>² + Δ̃<sub>κ</sub>² + |Δλ|² + 4)/(2|Δλ|²) diverges as Δλ → 0, because the two modes become parallel. Its minimum, 1, is where the modes are orthogonal (for example Δ̃<sub>f</sub> = Δ̃<sub>κ</sub> = 0).' },
    { id: 'meet', objective: 'kc',
      q: 'Among the settings on this page, where do the primary EP and TPD (nearly) coincide?',
      choices: ['At φ = 0 with κ̃<sub>c</sub> as large as possible', 'At φ = π with κ̃<sub>c</sub> close to 0', 'At every φ once κ̃<sub>c</sub> = 1', 'Nowhere: they are always at least 1 apart'],
      answer: 1,
      explain: 'Their distance shrinks as κ̃<sub>c</sub> → 0 and as φ → π. At φ = π the EP is at (0, 2) and the TPD at (0, √(κ̃<sub>c</sub>² + 4)), which meet as κ̃<sub>c</sub> → 0. At φ = 0 they stay at least 0.59 apart.' },
  ],
});
