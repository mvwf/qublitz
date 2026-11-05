/* arena.js: learning material for QuBlitz Arena (Photon Duel). The game is vendored byte-for-byte
 * (pages/_assets/quantum_chess.html); .github/workflows/pages.yml appends the outreach kit and this file to the
 * site's copy (qublitz/arena.html). The game's hub reads QL.spec to show "WHAT YOU'LL LEARN", the debrief
 * button and a lab link; the duel reports start / win / lose / level_complete / mode through QL.event, sets
 * QL.context (menu, mission n or free round n, brief / playing / paused / debrief), and between rounds asks one
 * `game` item at a time as a checkpoint: the short choices float in the arena as targets you shoot (QL.answer).
 * Each item's `game` block is the short in-game form; its choices keep the same order as `choices`.
 *
 * Physics notes (checked against the game's DuelCore): P(1) = (1 - z)/2 with z = +1 at |0>; beams are exact
 * rotations; T1/T2 are the exact Lindblad step (relaxation toward |0>); the coupler is a CNOT on a 4x4 density
 * matrix. Game rules, not physics: the 10 degree eigenstate tolerance, the 20 s window in mission 2 and the
 * 2.5 s Zeno freeze after two duds.
 */
if (window.QL) QL.learning({
  id: 'arena',
  title: 'QuBlitz Arena',
  tagline: 'A top-down laser duel where your opponent\'s health bar is their qubit.',
  fun: 'It plays like an arcade laser duel, but every hit is a real rotation of a qubit and every win is a ' +
       'Born-rule roll, so the quantum rules are the tactics.',
  objectives: [
    { id: 'born', text: '<b>Born rule:</b> predict your MEASURE win chance from P(1), and what a miss does to the qubit.' },
    { id: 'rotate', text: '<b>Rotations:</b> choose the charge (pulse area θ) that gives the best odds from a given state.' },
    { id: 'decohere', text: '<b>Decoherence:</b> explain why a mixed state cannot be rotated past P(1) = (1 + |r|)/2, and why T₁ pulls a charged qubit back to |0⟩.' },
    { id: 'eigen', text: '<b>Eigenstates:</b> identify when a beam does nothing (a state on the beam\'s own axis) and what moves it instead.' },
    { id: 'entangle', text: '<b>Entanglement:</b> explain why the CNOT coupler needs a control in superposition, how one MEASURE collapses both qubits, and why the link cannot send a signal.' },
    { id: 'zeno', text: '<b>Zeno effect:</b> relate frequent measurement to a qubit that stays near |0⟩, and tell the physics apart from the game\'s freeze rule.' },
  ],
  howTo: [
    'Press any key to skip the intro. New players: choose <b>DUEL TRAINING</b> and fly mission 1. For a best-of-3 match, choose <b>PHOTON DUEL</b>.',
    'Move with <b>WASD</b>. Aim with the mouse (or the arrow keys).',
    'Hold <b>SHIFT</b> or the left mouse button to charge a beam, then release to fire. A full charge is a π-pulse; the green mark on the charge bar shows the best charge for Sage\'s current state.',
    'Press <b>SPACE</b> to MEASURE Sage: you win the round with probability P(1). A miss resets Sage to |0⟩, and MEASURE needs 6 s to recharge.',
    'Defend yourself: hold <b>F</b> or the right mouse button to pulse your own qubit, or fly into a hotspot to shed charge. <b>1 / 2 / 3</b> pick the X, Y or T beam, <b>E</b> fires the entangling coupler, <b>ESC</b> pauses.',
    'Between rounds and after a mission clear you may get one checkpoint question: shoot the answer you think is right (or press 1 to 4), or press <b>ESC</b> to skip.',
  ],
  goals: [
    'Clear the seven training missions: each refuses the win until you use its concept. Stars: one for a clear, two under par time, three under par time and par shots.',
    'Then win a best-of-3 free duel against Sage (Cadet, Duelist or Decoherent Menace).',
  ],
  quiz: [
    { id: 'born-70', objective: 'born',
      q: 'The HUD says your win chance on Sage is 70%. You press MEASURE. What happens?',
      choices: [
        'You win the round 70% of the time; otherwise Sage collapses to |0⟩ and you have to rotate it up again',
        'You always win, because 70% is more than half',
        'Sage loses 70% of its health, and a second MEASURE finishes it',
        'If you miss, Sage is left at 30%, so you measure again right away',
      ],
      answer: 0,
      game: { q: 'Your win chance on Sage reads 70%. You press MEASURE. What happens?',
              choices: ['You win 70% of the time; a miss resets Sage to |0⟩', 'You always win: 70% is more than half', 'Sage loses 70% of its health', 'A miss leaves Sage at 30%'],
              line: 'Born rule: |1⟩ comes up with probability 0.7, and a miss collapses Sage to |0⟩.' },
      explain: 'Born rule: the outcome is |1⟩ with probability P(1) = 0.7. A measurement also collapses the state, so a miss leaves Sage at |0⟩ (P(1) = 0), not at 30%.' },
    { id: 'rotate-equator', objective: 'rotate',
      q: 'Sage starts at |0⟩. Your X half-charge (π/2) puts it on the equator at P(1) = 50%. Which next X charge gives the best MEASURE odds?',
      choices: [
        'A full charge (π): a bigger pulse is always better',
        'Another half-charge (π/2), which finishes the turn to |1⟩',
        'A quarter charge (π/4): P(1) grows by 25% per quarter charge',
        'It does not matter: from 50% every charge lands at the same odds',
      ],
      answer: 1,
      game: { q: 'Your X half-charge (π/2) put Sage on the equator at 50%. Which next X charge gives the best odds?',
              choices: ['A full charge (π)', 'Another half-charge (π/2)', 'A quarter charge: +25% each', 'Any charge: all the same'],
              line: 'π/2 + π/2 = π reaches |1⟩. A full π from the equator swings Sage back to 50%.' },
      explain: 'Pulses about the same axis add their angles, and from |0⟩ a total angle θ gives P(1) = sin²(θ/2), not a straight line. π/2 + π/2 = π gives 100%. A full π from the equator swings it to the other side of the equator (3π/2 in total), back at 50%.' },
    { id: 'decohere-ceiling', objective: 'decohere',
      q: 'Sage drifted through a hotspot, and its Bloch vector shrank to length |r| = 0.6. What is the best MEASURE chance any X or Y pulse can give you now?',
      choices: [
        '100%: a rotation can always reach |1⟩',
        '80%',
        '60%, the length of the vector',
        '30%',
      ],
      answer: 1,
      game: { q: 'A hotspot shrank Sage\'s Bloch vector to length |r| = 0.6. What is the best MEASURE chance any pulse can give?',
              choices: ['100%', '80%', '60%', '30%'],
              line: 'A rotation cannot lengthen the vector, so the ceiling is (1 + |r|)/2 = 80%.' },
      explain: 'A rotation turns the vector but cannot make it longer. P(1) = (1 − z)/2 and the lowest z it can reach is −|r|, so the ceiling is (1 + 0.6)/2 = 80%. Hotspots shrink the vector fast (short T₂), and their short T₁ also pulls it toward |0⟩, which is why they are a hiding place when you are charged.' },
    { id: 'eigen-plus', objective: 'eigen',
      q: 'Sage parks on the +x axis of its Bloch sphere (the state |+⟩), and your X-beam bounces off with EIGENSTATE! Why, and what works instead?',
      choices: [
        'A rotation about x leaves a state on the x axis where it is; a Y-beam or a T-phase moves it off',
        '|+⟩ is already all |1⟩, so there is nothing left to rotate',
        '|+⟩ is a fully mixed state, so no pulse can move it',
        'Sage picked up a shield power-up; wait for it to wear off',
      ],
      answer: 0,
      game: { q: 'Sage parks at |+⟩ (the +x axis) and your X-beam bounces off. Why?',
              choices: ['An X turn leaves the x axis in place: use Y or T', '|+⟩ is already all |1⟩', '|+⟩ is fully mixed', 'Sage has a shield power-up'],
              line: '|+⟩ is an eigenstate of X. A Y-beam or a T-phase moves it off the axis.' },
      explain: '|+⟩ is an eigenstate of X: an X rotation only changes its overall phase, so P(1) stays at 50%. A rotation about another axis (Y) or the T-phase (π/4 about z) moves it. The game blocks anything within 10° of the axis; a real state that close would move a little.' },
    { id: 'entangle-cnot', objective: 'entangle',
      q: 'You sit at |0⟩ and fire the coupler (E, a CNOT with your qubit as the control) at Sage, which is also at |0⟩. What happens?',
      choices: [
        'Nothing: a control at |0⟩ leaves the target alone. Get into superposition first (about π/2), then the CNOT makes an entangled pair',
        'You become entangled, and both qubits read 50%',
        'Sage\'s qubit flips to |1⟩',
        'Your qubit flips to |1⟩',
      ],
      answer: 0,
      game: { q: 'You and Sage are both at |0⟩. You fire the CNOT coupler (E). What happens?',
              choices: ['Nothing: your control is at |0⟩', 'You are entangled, both at 50%', 'Sage flips to |1⟩', 'You flip to |1⟩'],
              line: 'A CNOT acts only on the control\'s |1⟩ part: get into superposition first, then it entangles.' },
      explain: 'A CNOT flips the target only on the control\'s |1⟩ part. From |1⟩ it simply flips Sage (no entanglement). From |+⟩|0⟩ it makes (|00⟩ + |11⟩)/√2: one MEASURE then collapses both to the same result. Pulsing one half still cannot change the other half\'s odds (no signalling); it only swaps MATCH and OPPOSITE.' },
    { id: 'zeno-cage', objective: 'zeno',
      q: 'In THE ZENO CAGE, Sage is driven so its P(1) climbs as sin²(Ωt/2). Why does measuring it early and often keep it near |0⟩?',
      choices: [
        'Each miss collapses it to |0⟩, and from |0⟩ the climb starts slowly (P(1) ≈ (Ωt/2)² at first), so frequent resets keep it low',
        'Each MEASURE adds extra T₁ decay to Sage, which pulls it down',
        'Two measurements freeze any real qubit for 2.5 s',
        'It does not: P(1) climbs at a fixed rate whatever you do',
      ],
      answer: 0,
      game: { q: 'Sage is driven up as sin²(Ωt/2). Why does measuring it early and often keep it near |0⟩?',
              choices: ['Each miss resets it, and the climb starts slowly (∝ t²)', 'Each MEASURE adds T₁ decay', 'Real qubits freeze for 2.5 s', 'It does not help at all'],
              line: 'That slow t² start is the real Zeno effect. The game\'s 2.5 s freeze is only a game rule.' },
      explain: 'That slow, quadratic start is the real quantum Zeno effect: measure N times within a fixed time and the chance of ever finding |1⟩ shrinks toward zero as N grows. The 2.5 s freeze after two misses in 4 s is a game rule standing in for measuring nonstop; a real qubit is not frozen for a set time.' },
  ],
});
