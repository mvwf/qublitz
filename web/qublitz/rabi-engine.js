/* rabi-engine.js: physics and course for Rabi Rider (rabi-rider.html), plus the Bloch-vector helpers that
 * copter-engine.js shares. No dependencies. Loads as a classic <script> (window.RabiEngine) or from Node
 * (require), so tests/rabi-engine.test.js checks the same code the game runs.
 *
 * The qubit is a pure state, a unit Bloch vector r = (sinθ cosφ, sinθ sinφ, cosθ), with |0⟩ at θ = 0 (z = +1)
 * and |1⟩ at θ = π. It is driven exactly on resonance, so in the frame rotating with the drive
 *
 *   H = (ħΩ/2) a σx,   dr/dt = ω × r,   ω = (Ω a, 0, 0),
 *
 * and there is no precession: the drive is the only thing that turns the state. A drive about x keeps r in the
 * y–z plane, so the state is fixed by θ alone and the game draws it as a height while the course scrolls past
 * in time; the page puts |0⟩ (the ground state) at the bottom and |1⟩ (excited) at the top, so height is energy.
 * You ask for |1⟩ or |0⟩ (input { toOne, toZero }); the game picks the drive's sign (+X or −X) that turns the
 * state that way, which is a choice of drive phase (0 or π), and eases the drive off as the state reaches a pole.
 * Holding a drive on is a Rabi oscillation: P(1) = sin²(Ωt/2). The drive amplitude follows its target with time
 * constant TAU, like a bandwidth-limited pulse.
 *
 * The game flies an ensemble of N runs of the same experiment (P.RUNS = 240 by default), all driven by the same pulse, and draws a
 * few of them (opts.shown, P.GHOSTS = 12) as ghosts. The mean is the average over all N, so one run's jump moves it by only 1/N
 * and it follows the smooth curve an experiment would measure. A target counts
 * when the mean points through it and the ensemble's fidelity with it, F̄ = (1 + r̄·t̂)/2 (the share of runs that would pass a
 * test for the target), is at least F_PASS. F̄ is linear in r̄, so aiming the mean is the best strategy, and decoherence
 * (runs that jumped) shortens r̄ and caps F̄. The ensemble's average Bloch vector r̄ is the density matrix, ρ = (1 + r̄·σ)/2.
 *
 * Decoherence is T₁ only (energy loss, amplitude damping; no extra dephasing), always on: T₁ sets the difficulty, and
 * opts.T1 = Infinity turns it off, for tests. Each run is the standard photon-counting unravelling of amplitude damping:
 *   a jump: at the rate P₁/T₁ (P₁ = sin²(θ/2)) the run emits a photon and is left in |0⟩;
 *   between photons: seeing no photon is evidence the run is lower, so tan(θ/2) shrinks as e^(−t/2T₁), i.e.
 *     dθ/dt = −sin θ/(2T₁). |1⟩ and |0⟩ do not move (sin θ = 0): |1⟩ waits and then jumps; superpositions slide down.
 * Averaged over runs this is exactly the T₁ master equation: P₁ decays as e^(−t/T₁) and the coherence (x, y) as e^(−t/2T₁).
 * T₁+ tokens (power-ups) multiply T₁ by T1_UP, up to T1_MAX times its starting value: each is a real way labs lengthen T₁.
 * T₁− chips (hazards, type 'drain') divide it by T1_UP, down to T1_MIN times the start: each is a real way a qubit loses T₁.
 *
 * Hitting noise costs a heart, and the noise measures the qubit: stray coupling to the environment acts as an
 * unwanted measurement of σz, so the state collapses to |1⟩ with probability P₁ and to |0⟩ otherwise (decoherence
 * as a measurement nobody asked for). A measurement gate is the same projective measurement
 * of σz: the state goes to |1⟩ with probability P₁ = sin²(θ/2) and to |0⟩ otherwise (the Born rule).
 * An X token applies the X gate, a π turn about x, which takes θ to π − θ.
 */
(function (root) {
  'use strict';
  const TAU2 = 2 * Math.PI, D2R = Math.PI / 180;

  // ---------------- vectors (shared with copter-engine.js) ----------------
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => Math.hypot(a[0], a[1], a[2]);
  const unit = a => { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
  const fromAngles = (th, ph) => [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
  const toAngles = r => ({ th: Math.acos(Math.max(-1, Math.min(1, r[2]))), ph: Math.atan2(r[1], r[0]) });
  const wrap = a => { a = (a + Math.PI) % TAU2; return (a < 0 ? a + TAU2 : a) - Math.PI; };   // into [−π, π)
  // Rotate r about ω by |ω|·dt (Rodrigues), the exact solution of dr/dt = ω × r for constant ω.
  function rotate(r, w, dt) {
    const wn = norm(w); if (wn < 1e-12) return r.slice();
    const k = [w[0] / wn, w[1] / wn, w[2] / wn], a = wn * dt, c = Math.cos(a), s = Math.sin(a);
    const kxr = cross(k, r), kr = dot(k, r);
    return [0, 1, 2].map(i => r[i] * c + kxr[i] * s + k[i] * kr * (1 - c));
  }
  // State fidelity |⟨t|ψ⟩|² between two pure states, from their Bloch vectors.
  const fidelity = (r, t) => (1 + dot(r, t)) / 2;
  // Angle on the sphere between two Bloch vectors.
  const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot(a, b))));
  // Seeded generator (mulberry32 on a string hash), so a class can share a course.
  function rng(seed) {
    let h = 1779033703 ^ String(seed).length;
    for (const ch of String(seed)) { h = Math.imul(h ^ ch.charCodeAt(0), 3432918353); h = (h << 13) | (h >>> 19); }
    let a = h >>> 0;
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }

  // ---------------- game constants ----------------
  const P = {
    OMEGA: 1.6,          // Rabi rate Ω [rad/s]: a π pulse takes about 2 s (slow enough to aim with human reflexes)
    TAU: 0.1,            // drive amplitude time constant [s]
    EASE: 10 * D2R,      // the drive eases off this close to the pole it is pushing toward
    T1: { easy: 12, normal: 6, hard: 3, expert: 1.5 },   // T₁ [s] for each difficulty
        RUNS: 240,           // runs of the experiment flown at once, all by the same drive; the mean is their average
    GHOSTS: 12,          // of which this many are drawn (the ghosts)
    T1_UP: 1.35,         // a T₁+ token multiplies T₁ by this
    T1_MAX: 4,           // up to this many times the starting T₁
    POWER: 13 * D2R,     // pickup half-height of a T₁+ token (and of a T₁− hazard)
    T1_MIN: 0.5,         // a T₁− hazard can take T₁ down to half where it started, no lower
    F_PASS: 0.7,        // a target counts when the ensemble's fidelity with it is at least this (and the mean points through it)
    F_PERFECT: 0.95,     // and is a perfect above this
    SHOT: 8,             // s: the endless run resets every run to |0⟩ about this often, starting a new shot of the experiment
    HEARTS: 3,           // in a mission, where only noise costs a heart
    ENDLESS_HEARTS: 5,   // in the endless run, where a missed target costs one too
    STREAK_HEAL: 3,      // and every 3 targets in a row restores one
    GRACE: 1.2,          // s of invulnerability after a jump
    HIT_GRACE: 1.6,      // and after a hit, which also collapses the qubit to |0⟩ or |1⟩
    MISSION_RING: 16 * D2R,
    RING0: 20 * D2R, RING_MIN: 9 * D2R,   // the endless run's rings shrink from RING0 to RING_MIN over LEVEL_TIME
    TOKEN: 10 * D2R,     // pickup half-height of a gate token
    LEVEL_TIME: 150,     // s for the endless course to reach full difficulty
    AHEAD: 10,           // s of course generated ahead
    CLOSE: 4 * D2R,      // passing a cloud this close (without touching it) is a close call
  };

  // A Bloch vector on the y–z circle at polar angle th, on the −y side (where +X pushes θ up) or the +y side.
  const onCircle = (th, side = -1) => [0, side * Math.sin(th), Math.cos(th)];
  const thetaOf = r => Math.acos(Math.max(-1, Math.min(1, r[2])));

  /* T₁+ tokens, in the order they are collected: each is a real way to make a qubit lose its energy more slowly. */
  const UPGRADES = [
    { name: 'Purcell filter', fact: 'The readout resonator is a window to the outside: a qubit can leak its photon through it. A Purcell filter blocks the qubit\'s frequency and keeps the photon in.' },
    { name: 'Infrared shielding', fact: 'Stray infrared light breaks Cooper pairs in the superconductor, and the broken pairs (quasiparticles) soak up the qubit\'s energy. Light-tight, blackened shields stop it.' },
    { name: 'Cleaner surfaces', fact: 'Tiny defects in the oxides on the chip (two-level systems) absorb energy at the qubit\'s frequency. Better cleaning and etching removes many of them.' },
    { name: 'Tantalum films', fact: 'Changing the superconductor from niobium or aluminium to tantalum, whose oxide holds fewer lossy defects, pushed transmon T₁ from about 100 µs to about 300 µs.' },
    { name: 'Magnetic shielding', fact: 'Trapped magnetic flux makes vortices in the film, and vortices dissipate energy. A mu-metal can around the chip keeps Earth\'s field out while it cools down.' },
    { name: 'Better packaging', fact: 'The box holding the chip has resonances of its own. Moving them far from the qubit\'s frequency removes one more place for its photon to go.' },
  ];

  /* T₁− hazards: real ways a qubit's T₁ gets worse, cycled in order. */
  const HAZARDS = [
    { name: 'TLS defect', fact: 'A defect in the chip\'s oxide (a two-level system) drifted into resonance with the qubit and is soaking up its energy. These defects wander in frequency, which is why a real qubit\'s T₁ changes from hour to hour.' },
    { name: 'Light leak', fact: 'Infrared light got past the shields. It breaks Cooper pairs, and the quasiparticles it leaves behind absorb the qubit\'s energy.' },
    { name: 'Trapped vortex', fact: 'A magnetic field was trapped in the film as it cooled. Each vortex has a core of normal metal that dissipates energy as current flows past it.' },
    { name: 'Lossy seam', fact: 'A gap in the lid of the box holding the chip lets microwave energy leak out, so the qubit\'s photon has one more way to escape.' },
  ];

  /* Quiz gates: two rings at the same moment, and the prompt describes one of them. Each maker takes a seeded
   * generator and returns the right θ, a tempting wrong θ, and the explanation shown afterward. */
  const fmtDeg = th => `${Math.round(th / D2R)}°`;
  const fmtNum = x => String(+x.toFixed(2)).replace('-', '−');
  const pick = (R, xs) => xs[Math.floor(R() * xs.length)];
  const QUIZ = [
    R => { const p = pick(R, [0.1, 0.25, 0.75, 0.9]), th = 2 * Math.asin(Math.sqrt(p));
      return { prompt: `P(1) = ${p}`, th, wrong: Math.PI - th,
        why: `P(1) = sin²(θ/2), so P(1) = ${p} sits at θ = ${fmtDeg(th)}. The other ring had P(1) = ${fmtNum(1 - p)}.` }; },
    R => { const p = pick(R, [0.1, 0.25, 0.75, 0.9]), th = 2 * Math.acos(Math.sqrt(p));
      return { prompt: `P(0) = ${p}`, th, wrong: Math.PI - th,
        why: `P(0) = cos²(θ/2), so P(0) = ${p} sits at θ = ${fmtDeg(th)}. The closer to |0⟩ at the bottom, the likelier a 0.` }; },
    R => { const z = pick(R, [0.8, 0.5, -0.5, -0.8]), th = Math.acos(z);
      return { prompt: `⟨σz⟩ = ${fmtNum(z)}`, th, wrong: Math.PI - th,
        why: `⟨σz⟩ = cos θ = P(0) − P(1): +1 at |0⟩ (the bottom), −1 at |1⟩ (the top). So ⟨σz⟩ = ${fmtNum(z)} is at θ = ${fmtDeg(th)}.` }; },
    R => { const [deg, name] = pick(R, [[60, 'π/3'], [90, 'π/2'], [120, '2π/3'], [135, '3π/4']]), th = deg * D2R;
      return { prompt: `|0⟩ after a ${name} pulse`, th, wrong: th / 2,
        why: `A pulse of area ${name} turns the Bloch vector by ${deg}°, so θ = ${deg}°. Half of that, ${deg / 2}°, is the angle in the ket cos(θ/2)|0⟩ − i sin(θ/2)|1⟩.` }; },
  ];

  /* Missions: short scripted courses, each teaching one idea. Items are placed by time t [s] and polar angle θ [°]:
   *   ['r', t, θ]            ring (a target state)        ['c', t, θ, half-height°, half-width s]   noise cloud
   *   ['q', t, k]            quiz gate QUIZ[k]             ['m', t, want]   measurement gate paying for outcome want (0 or 1)
   *   ['x', t, θ]            X gate token                  ['z', t]         reset every run to |0⟩ (a new shot)
   *   ['u', t, θ]            T₁+ token (a power-up)        ['d', t, θ]      T₁− hazard (dodge it)
   * Stars: 3 for 90 % of the rings (or the mission's `three` share) with no hits, 2 for 70 %, 1 for finishing. Decay is always
 * on, so a quantum jump can cost a ring: that is luck the stars allow for. */
  const MISSIONS = [
    { id: 'flight', title: 'Flight school', start: 90, ghosts: 1, T1: 30,
      brief: 'One qubit, one run. Hold <b>▲</b> to drive it up toward |1⟩ and <b>▼</b> down toward |0⟩; let go and the drive is off. Fly through every ring.',
      learn: 'Each ring was a target state, and you reached it by rotating the Bloch vector with a resonant drive. With the drive off nothing turns it: in the frame rotating with a resonant drive there is no precession (in the lab the qubit still turns at GHz, but so does the frame). The qubit did not stay put forever, though: that is the next mission.',
      items: [['r', 3, 90], ['r', 5, 60], ['r', 7, 40], ['r', 9, 70], ['r', 11, 110], ['r', 13, 140], ['r', 15, 120], ['r', 17, 80], ['r', 19, 45], ['r', 21, 90]] },
    { id: 'photon', title: 'One qubit, one photon', start: 0, ghosts: 1, T1: 4, three: 0.6,
      brief: 'Still one run, now with a short T₁ (4 s). Climb to |1⟩ and stay near the top. At a random moment the qubit will give up its energy as a <b>photon</b> (the yellow wave) and <b>jump</b> straight down to |0⟩. That is a quantum jump: not a slow slide but all at once. Climb back and keep flying.',
      learn: 'A qubit in |1⟩ does not decay a little at a time. It stays put and then, at a random moment you cannot predict, emits one photon and is in |0⟩. The rate is fixed, 1/T₁ (times P(1), the chance it is found excited), so a jump in the next second is as likely now as it was a minute ago. T₁ is the average wait. Next: what many runs of this look like together.',
      items: [['r', 2.5, 160], ['r', 5, 170], ['r', 7.5, 160], ['r', 10, 170], ['r', 12.5, 155], ['r', 15, 170], ['r', 17.5, 160], ['r', 20, 170]] },
    { id: 'ensemble', title: 'Many runs', start: 90,
      brief: 'An experiment repeats the same pulses many times, and each run jumps at its own moment. Now you fly 240 runs at once, with one drive for all of them; the faint ghosts are a few of them. The big marker is their mean, the density matrix. Aim the mean through each ring: it counts when the share of runs that would pass a test for the ring\'s state, the fidelity, is at least 70 %.',
      learn: 'Every run felt the same pulse, but each jumped at its own random moment, so the runs drifted apart and the mean shrank. You cannot steer one run at a time, and you cannot see them in a real experiment: what you measure is the average. Aiming the mean is the best a single pulse can do, and it gets the most runs through.',
      items: [['r', 3, 90], ['r', 5, 60], ['r', 7, 40], ['r', 9, 70], ['r', 11, 110], ['r', 13, 140], ['r', 15, 120], ['r', 17, 80], ['r', 19, 45], ['r', 21, 90]] },
    { id: 'pi', title: 'The π pulse', start: 0,
      brief: 'A π pulse turns |0⟩ all the way to |1⟩. Hold <b>▲</b> to climb to the ring at the top, then <b>▼</b> back down to the bottom.',
      learn: 'At this Rabi rate (Ω = 1.6 rad/s) a π pulse takes π/Ω ≈ 2 s, which is how long each climb took. A constant drive would carry the state through |1⟩ and back to |0⟩, P(1) = sin²(Ωt/2): one Rabi oscillation. The game stops you at |1⟩ and reverses the drive when you press ▼, so your up-and-down traced the same P(1) curve. On a real transmon a π pulse typically takes 10–50 ns.',
      items: [['r', 3, 0], ['r', 6, 180], ['r', 9.5, 0], ['r', 12.5, 180], ['r', 16, 0], ['r', 19, 180], ['r', 22.5, 0]] },
    { id: 'equator', title: 'Superposition', start: 0,
      brief: 'The middle line is the equator: P(1) = ½, an equal superposition. That takes a π/2 pulse, half a climb. Stay on it between the noise clouds.',
      learn: 'On the equator a measurement gives 0 or 1 with even odds, yet the state is not a coin that has already landed: one more π/2 pulse takes it to |1⟩ with certainty.',
      items: [['r', 3.5, 90], ['c', 5, 40, 18, 0.5], ['c', 5, 140, 18, 0.5], ['r', 6.5, 90], ['c', 8, 42, 17, 0.6], ['c', 8, 138, 17, 0.6],
        ['r', 9.5, 90], ['r', 12, 0], ['r', 14.5, 90], ['c', 16, 40, 18, 0.5], ['c', 16, 140, 18, 0.5], ['r', 17.5, 90]] },
    { id: 'born', title: 'Measure up', start: 90,
      brief: 'Question gates: two purple rings, one answer. Read the question and fly through the ring it describes. The labels on the right give P(1) at each height.',
      learn: 'The height here is the angle θ, not the sphere\'s z = cos θ. P(1) = (1 − cos θ)/2 = sin²(θ/2) is a straight line in z but not in θ, so P(1) = ¼ sits at θ = 60°, not 45°.',
      // question gates are spaced ~8 s apart so there is time to read and think (players found 5.5 s too fast)
      items: [['r', 3, 90], ['q', 8, 0], ['r', 11.5, 90], ['q', 16.5, 2], ['r', 20, 90], ['q', 25, 3], ['r', 28.5, 90], ['q', 33.5, 1], ['r', 37, 90]] },
    { id: 'collapse', title: 'Collapse', start: 90,
      brief: 'A measurement gate (the blue curtain) reads every ghost: each lands in |0⟩ or |1⟩ at random, with P(1) = sin²(θ/2). Each gate names the outcome it pays for, and pays for the share of ghosts that read it: fly close to that end first.',
      learn: 'Measurement is not a rotation. Each run gives one definite answer with the Born-rule odds, so the fraction of 1s across the runs is P(1). Each run is left in the state it read; here the game then resets every run to |0⟩, as labs do. (In Free play, Measure skips the reset, so you can see the ghosts split between the poles.)',
      items: [['r', 3, 60], ['m', 5.5, 0], ['r', 8, 90], ['m', 10.5, 1], ['r', 13, 100], ['m', 15.5, 1], ['r', 18, 80], ['m', 20.5, 0], ['r', 23, 90]] },
    { id: 'gates', title: 'The X gate', start: 30,
      brief: 'X tokens apply the X gate at once: a π pulse, done for you. It takes θ to 180° − θ, mirroring you through the equator. Some rings come too fast to fly to: grab the X first. Green resets start each try from |0⟩.',
      learn: 'A gate is a fast, calibrated pulse. X is a π rotation about x, so it swaps |0⟩ and |1⟩ and maps every height θ to 180° − θ.',
      items: [['r', 3, 30], ['x', 4, 30], ['r', 4.6, 150], ['z', 5.3], ['r', 6.5, 45], ['x', 7.5, 45], ['r', 8.1, 135], ['z', 8.8], ['r', 10, 60], ['x', 11, 60], ['r', 11.6, 120],
        ['z', 12.3], ['x', 13.2, 0], ['r', 13.8, 180], ['r', 15.6, 150], ['x', 16.6, 150], ['r', 17.2, 30], ['r', 19, 90]] },
    { id: 'jumps', title: 'Jumps in the crowd', start: 0, T1: 5, three: 0.6,
      brief: 'A short T₁ here (5 s, whatever the difficulty). The green curtains reset every ghost to |0⟩, a fresh shot. Watch the ghosts: each one falls to |0⟩ in a sudden jump at its own random moment, and the higher it was, the likelier a photon flies out. Jumps cost no hearts, but a ghost that jumped has to be brought back up.',
      learn: 'T₁ and the jumps are the same thing seen two ways. Each run emits its photon and drops to |0⟩ at a random moment, at the rate P(1)/T₁; between photons a superposition slides slowly down, because not seeing a photon is itself evidence that the run is lower. Averaged over the runs, the fraction in |1⟩ falls smoothly as e^(−t/T₁): that average is what an experiment measures, and the big marker shows it.',
      items: [['r', 2, 100], ['r', 3.5, 125], ['z', 4.5], ['r', 6.5, 110], ['r', 8, 130], ['z', 9], ['r', 11, 120], ['r', 12.5, 100], ['z', 13.5],
        ['r', 15.5, 130], ['r', 17, 110]] },
    { id: 'upgrade', title: 'Build a better qubit', start: 0, T1: 2.5, three: 0.6,
      brief: 'This chip is bad: T₁ = 2.5 s, barely longer than a π pulse, so runs near |1⟩ keep jumping down. Fly the mean through the cyan <b>T₁+</b> chips: each one is a real fix that labs use, and each makes T₁ 35 % longer. Dodge the red <b>T₁−</b> chips: each is a real way T₁ gets worse. The rings high up get easier with every upgrade.',
      learn: 'T₁ is not a law of nature: it is set by everywhere the qubit\'s energy can go, and engineers close those exits one by one. Superconducting qubits went from nanoseconds (the first Cooper-pair box, 1999) to hundreds of microseconds today. A longer T₁ lets the runs stay together for longer, so the mean stays long and more experiments succeed.',
      items: [['u', 1.6, 60], ['r', 3, 100], ['u', 4.3, 120], ['z', 5.2], ['u', 6.6, 90], ['r', 8, 130], ['z', 9], ['u', 10.4, 100], ['r', 12, 145],
        ['u', 13.3, 155], ['r', 14.8, 165], ['z', 15.8], ['d', 17, 70], ['r', 18, 150], ['r', 20, 165]] },
    { id: 'exam', title: 'Final exam', start: 90,
      brief: 'Everything at once: rings, noise, quiz gates, a measurement, a T₁+ chip and X tokens.',
      learn: 'You drove a qubit with real resonant pulses, read probabilities off the Bloch sphere, used a gate, and watched a measurement collapse the state. That is the toolkit behind every single-qubit experiment.',
      items: [['r', 3, 90], ['c', 4.5, 40, 20, 0.5], ['r', 6, 130], ['q', 9, 0], ['r', 11.5, 90], ['m', 14, 1], ['r', 16.5, 90], ['c', 18, 140, 22, 0.5],
        ['u', 19.5, 60], ['r', 21, 60], ['x', 22, 60], ['r', 22.6, 120], ['q', 25.5, 2], ['r', 28, 90]] },
  ];
  const isGood = o => o.type === 'ring' || o.type === 'token';   // things to collect, which clouds never cover

  /* ---------------- the world ----------------
   * opts: { free (free play: no course, never over), seed, T1 [s] (default P.T1.normal; Infinity for no decay),
   *         ghosts (runs simulated; default P.RUNS; 1 for a single run), shown (runs drawn as ghosts; default P.GHOSTS), mission: id };
   * a mission's own T1 overrides opts.T1.
   * Missions start with HEARTS and only noise costs one; the endless run starts with ENDLESS_HEARTS, a missed target
   * (or a wrong quiz answer) costs one, and every STREAK_HEAL targets in a row restores one.
   * step(input, dt): input = { toOne, toZero } (booleans). Each step returns a list of events: ring, miss, quiz, quizmiss,
   * token, power (a T₁+ token), measure, close, heal, hit, jump, clear (a mission finished) and over.
   * Objects carry a time t; the course scrolls in time, and an object is met when the clock reaches it. */
  class World {
    constructor(opts = {}) {
      this.T1 = opts.T1 ?? P.T1.normal;
      this.upgrades = 0; this.powerPlanned = 0; this.drains = 0;         // T₁+ tokens collected, and placed on the endless course
      this.noReset = !!opts.noReset;
      this.free = !!opts.free;                          // free play: no course, no hearts, no end
      this.photons = 0;
      this.N = Math.max(1, Math.round(opts.ghosts ?? P.RUNS));
      this.rand = rng('course:' + (opts.seed ?? 'random'));
      this.t = 0; this.amp = 0;                         // amp: −1 drives toward |0⟩, +1 toward |1⟩
      this.objects = []; this.nextT = 2.5; this.walk = 90 * D2R; this.lastReset = 0;
      this.score = 0; this.combo = 0; this.rings = 0; this.misses = 0; this.perfects = 0; this.jumps = 0; this.hits = 0; this.closes = 0;
      this.quizRight = 0; this.quizWrong = 0; this.measures = 0; this.measWins = 0; this.gates = 0;
      this.hearts = P.ENDLESS_HEARTS; this.streak = 0; this.grace = 0; this.over = false;
      this.mission = null; this.cleared = false; this.total = 0;
      const m = opts.mission && MISSIONS.find(x => x.id === opts.mission);
      if (m && m.ghosts) this.N = m.ghosts;              // a mission can fix the number of runs
      this.shown = Math.min(this.N, Math.max(1, Math.round(opts.shown ?? P.GHOSTS)));
      // each ghost is one run of the experiment, with its own photon detections and measurement outcomes
      const tag = m ? 'mission:' + m.id : String(opts.seed ?? 'random');
      this.ghosts = Array.from({ length: this.N }, (_, i) => ({
        r: onCircle(90 * D2R), jump: rng(`jumps:${tag}:${i}`), meas: rng(`measure:${tag}:${i}`), noise: rng(`noise:${tag}:${i}`) }));
      // Photon times, stratified: each run waits until its integrated jump rate ∫P₁/T₁ dt passes −ln u, a fresh uniform u for
      // each photon. Any one run's u is uniform and independent of its last one, so every run is a correct random trajectory;
      // across runs the k-th u's are spread evenly over (0, 1) (strata, offset by a shift shared at each k), so the share of
      // runs that have jumped follows e^(−t/T₁) closely and the mean does not wander off the master equation.
      const perm = this.ghosts.map((_, i) => i), PR = rng(`strata:${tag}`);
      for (let i = perm.length - 1; i > 0; i--) { const j = Math.floor(PR() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      this.shifts = [0]; this.shiftR = rng(`shift:${tag}`);
      this.ghosts.forEach((gh, i) => { gh.u0 = (perm[i] + gh.jump()) / this.N; gh.k = 0; gh.acc = 0; gh.thr = -Math.log(1 - gh.u0); });
      if (m) {
        this.mission = m; if (m.T1 && this.T1 < Infinity) this.T1 = m.T1; this.hearts = P.HEARTS; this.r = onCircle(Math.max(1e-4, Math.min(Math.PI - 1e-4, m.start * D2R)));
        this.script(m);
      } else if (!this.free) this.generate();
      this.T1start = this.T1;
    }
    get decay() { return this.T1 < Infinity; }
    get level() { return Math.min(1, this.t / P.LEVEL_TIME); }
    get ringRadius() { return P.RING0 + (P.RING_MIN - P.RING0) * this.level; }
    // The ensemble's Bloch vector, the average over the runs: the density matrix is ρ = (1 + r̄·σ)/2, and |r̄| < 1 once
    // the runs disagree (a mixed state). Setting it puts every run in that (pure) state.
    get r() { const m = [0, 0, 0]; for (const g of this.ghosts) for (let k = 0; k < 3; k++) m[k] += g.r[k] / this.N; return m; }
    set r(v) { for (const g of this.ghosts) g.r = v.slice(); }
    get p1() { return Math.max(0, Math.min(1, (1 - this.r[2]) / 2)); }   // the fraction of runs that would read 1
    // the height with the ensemble's P(1): θ̄ = 2 arcsin √P̄(1) (for one run, its own θ)
    get meanTheta() { return 2 * Math.asin(Math.sqrt(this.p1)); }
    // "You" are the ensemble mean, the direction of r̄ (the density matrix's Bloch vector): the drive is one pulse for every
    // run, so the best you can do is aim the mean, and the runs scatter around it as decoherence throws individual runs off.
    // Its length |r̄| (1 when every run agrees) is how far they have spread. When r̄ ≈ 0 the direction is kept from before.
    get purity() { return norm(this.r); }
    get theta() {
      const m = this.r, n = norm(m);
      if (n > 1e-6) this._dir = Math.acos(Math.max(-1, Math.min(1, m[2] / n)));
      return this._dir ?? Math.PI / 2;
    }
    // the heights of the runs drawn as ghosts
    thetas() { const out = []; for (let i = 0; i < this.shown; i++) out.push(thetaOf(this.ghosts[i].r)); return out; }
    get driving() { return Math.abs(this.omega()[0]) > 0.02 * P.OMEGA; }
    // The drive about x, one pulse for every run: its sign follows the side of the circle the ensemble mean is on, so +amp
    // pushes the mean toward |1⟩, and it eases off within EASE of the pole it is pushing toward. Once the runs disagree the
    // mean is shorter than 1 (a mixed state) and a pulse moves it less; split evenly between |0⟩ and |1⟩, it is zero and no
    // pulse can move it at all. Only T₁, by bringing the runs back to |0⟩, restores it.
    omega() {
      const th = this.meanTheta, side = this.r[1] <= 0 ? 1 : -1;   // ease by P̄(1), so a nearly mixed ensemble can still be driven
      const ease = Math.min(1, (this.amp > 0 ? Math.PI - th : th) / P.EASE);
      return [P.OMEGA * this.amp * side * ease, 0, 0];
    }
    // +1 for a +X drive, −1 for −X, 0 when off: the pulse the game is playing for you.
    get driveSign() { const w = this.omega()[0]; return Math.abs(w) < 0.02 * P.OMEGA ? 0 : Math.sign(w); }

    step(input, dt) {
      const ev = [];
      if (this.over) return ev;
      const want = (input.toOne ? 1 : 0) - (input.toZero ? 1 : 0);
      this.amp += (want - this.amp) * (1 - Math.exp(-dt / P.TAU));
      if (Math.abs(this.amp) < 1e-3 && !want) this.amp = 0;
      const w = this.omega(), photons = [], shrink = this.decay ? Math.exp(-dt / (2 * this.T1)) : 1;
      this.ghosts.forEach((gh, i) => {
        let r = rotate(gh.r, w, dt);
        if (this.decay) {
          gh.acc += (1 - r[2]) / 2 * dt / this.T1;
          if (gh.acc >= gh.thr) {
            // a photon: at the rate P₁/T₁ the run gives up its energy and is left in |0⟩
            photons.push({ i, th: thetaOf(r), shown: i < this.shown }); r = [0, 0, 1];
            gh.k++; gh.acc = 0; gh.thr = -Math.log(1 - this.nextU(gh));
          } else {
            // no photon: tan(θ/2) shrinks by e^(−dt/2T₁) at fixed φ (|0⟩ and |1⟩ stay put)
            const th = thetaOf(r), rho = Math.hypot(r[0], r[1]);
            if (rho > 1e-12) {
              const t2 = 2 * Math.atan2(Math.sin(th / 2) * shrink, Math.cos(th / 2)), s2 = Math.sin(t2) / rho;
              r = [r[0] * s2, r[1] * s2, Math.cos(t2)];
            }
          }
        }
        gh.r = unit(r);
      });
      if (photons.length) {
        this.jumps += photons.length; this.photons += photons.length; ev.push({ type: 'jump', n: photons.length, photons });
        if (this.N === 1) { this.combo = 0; this.grace = Math.max(this.grace, P.GRACE); }
      }
      const t0 = this.t; this.t += dt;
      this.score += 2 * dt;
      this.grace = Math.max(0, this.grace - dt);
      if (!this.mission && !this.free) this.generate();
      this.meet(t0, ev);
      this.objects = this.objects.filter(o => o.t + (o.w || 0) > this.t - 3);
      if (this.hearts <= 0 && !this.free) { this.over = true; ev.push({ type: 'over' }); }
      else if (this.mission && this.t >= this.endT) { this.over = true; this.cleared = true; ev.push({ type: 'clear', stars: this.stars() }); }
      return ev;
    }

    // The uniform u for a run's next photon: its stratum, moved by a shift shared by every run at that photon count.
    nextU(gh) { while (this.shifts.length <= gh.k) this.shifts.push(this.shiftR()); const u = (gh.u0 + this.shifts[gh.k]) % 1; return Math.min(u, 1 - 1e-12); }
    // How many runs are within rad of θ (shown on the page; the score uses the fidelity below).
    count(th, rad) { let n = 0; for (const g of this.ghosts) if (Math.abs(thetaOf(g.r) - th) < rad) n++; return n; }
    // The ensemble's fidelity with the target state at height th: F̄ = ⟨t|ρ|t⟩ = (1 + r̄·t̂)/2, the average of the runs'
    // fidelities, i.e. the share of runs that would pass a test for |t⟩. It is linear in r̄, so aiming the mean is the
    // best you can do, and decoherence caps it at (1 + |r̄|)/2. The target is the state at the ring's height on the mean's own
    // azimuth φ̄ (the closest state at that height), so a phase error common to every run does not count against you, but
    // runs that disagree in phase do.
    fidelity(th) { const m = this.r, ph = Math.hypot(m[0], m[1]) > 1e-9 ? Math.atan2(m[1], m[0]) : -Math.PI / 2;
      return (1 + dot(m, fromAngles(th, ph))) / 2; }
    // A target is met when the mean points through it and F̄ ≥ F_PASS.
    passes(o) { return Math.abs(this.theta - o.th) < o.rad && this.fidelity(o.th) >= P.F_PASS; }
    // Everything the clock reached in (t0, t].
    meet(t0, ev) {
      const th = this.theta, half = this.N / 2;
      for (const o of this.objects) {
        if (o.done) continue;
        if (o.type === 'cloud') {
          // an ellipse in (time, θ). Noise measures each run that touches it (Born rule); a heart is lost when it reaches half of them
          const u = (this.t - o.t) / o.w, reach = o.rad * Math.sqrt(Math.max(0, 1 - u * u));
          if (Math.abs(u) <= 1) {
            o.near = Math.min(o.near ?? Infinity, Math.abs(th - o.th) - reach);
            if (this.grace > 0) continue;
            o.touched = o.touched || new Set(); let fresh = 0, ones = 0, p1 = this.p1;
            this.ghosts.forEach((g, i) => {
              if (o.touched.has(i) || Math.abs(thetaOf(g.r) - o.th) >= reach) return;
              o.touched.add(i); fresh++;
              const one = g.noise() < (1 - g.r[2]) / 2; g.r = one ? [0, 0, -1] : [0, 0, 1]; if (one) ones++;
            });
            if (fresh) {
              const heart = !o.struck && o.touched.size >= half;
              if (heart) { o.struck = true; this.hearts--; this.hits++; this.combo = 0; this.grace = P.HIT_GRACE; this.amp = 0; }
              ev.push({ type: 'hit', o, n: fresh, ones, p1, heart, one: ones > 0 });
            }
          } else if (u > 1) {
            o.done = true;
            if (!o.struck && !o.touched && o.near >= 0 && o.near < P.CLOSE) { const pts = 25 * Math.max(1, this.combo); this.score += pts; this.closes++; ev.push({ type: 'close', o, pts }); }
          }
          continue;
        }
        if (o.t > this.t || o.t <= t0 - 1e-9) continue;
        o.done = true;
        if (o.type === 'gate') this.measure(o, ev);
        else if (o.type === 'drain') { if (Math.abs(th - o.th) < P.POWER) this.drain(o, ev); else { o.dodged = true; this.score += 30; ev.push({ type: 'dodge', o }); } }
        else if (o.type === 'reset') this.reset(o, ev);
        else if (o.type === 'token') { if (o.kind === 'T1') { if (Math.abs(th - o.th) < P.POWER) this.power(o, ev); } else if (Math.abs(th - o.th) < P.TOKEN) this.applyX(o, ev); }
        else if (o.quiz) this.answer(o, ev);
        else {
          const n = this.count(o.th, o.rad), off = Math.abs(th - o.th), F = this.fidelity(o.th);
          if (this.passes(o)) {
            this.combo = Math.min(10, this.combo + 1);
            const perfect = F >= P.F_PERFECT && off < o.rad / 3; if (perfect) this.perfects++;
            const pts = Math.round(100 * this.combo * F * (perfect ? 1.5 : 1)); this.score += pts;
            ev.push({ type: 'ring', o, pts, perfect, combo: this.combo, F, n, N: this.N });
            this.counted(ev);
          } else { this.combo = 0; this.misses++; o.missed = true; ev.push({ type: 'miss', o, off, F, n, N: this.N, heart: this.lose() }); }
        }
      }
    }
    // A target reached. In the endless run every STREAK_HEAL in a row restores a heart.
    counted(ev) {
      this.rings++;
      if (!this.mission && ++this.streak % P.STREAK_HEAL === 0 && this.hearts < P.ENDLESS_HEARTS) { this.hearts++; ev.push({ type: 'heal' }); }
    }
    // A target missed: in the endless run it costs a heart (true if it did).
    lose() { this.streak = 0; if (this.mission) return false; this.hearts--; return true; }
    // Both rings of a quiz arrive together: the one the ensemble passes (if either) answers it.
    answer(o, ev) {
      const q = o.quiz; if (q.answered) return;
      const pair = this.objects.filter(x => x.quiz === q), right = pair.find(x => x.correct), wrong = pair.find(x => !x.correct);
      q.answered = true; pair.forEach(x => { x.done = true; });
      const pr = this.passes(right), pw = this.passes(wrong), F = this.fidelity(right.th);
      if (!pr && !pw) { this.combo = 0; this.misses++; ev.push({ type: 'quizmiss', q, F, heart: this.lose() }); return; }
      const correct = pr, mine = correct ? right : wrong;
      let pts = 0;
      if (correct) { this.quizRight++; this.combo = Math.min(10, this.combo + 1); pts = Math.round(300 * this.combo * F); this.score += pts; this.counted(ev); }
      else { this.quizWrong++; this.combo = 0; }
      ev.push({ type: 'quiz', o: mine, q, correct, pts, combo: this.combo, F, heart: !correct && this.lose() });
    }
    // A projective measurement of σz in every run: each reads 1 with probability P₁ = sin²(θ/2), otherwise 0 (the Born rule).
    // Then measure-and-reset, as labs do: each run that read 1 gets a π pulse conditioned on its own outcome (real-time
    // feedback), so every run starts again in |0⟩ and the ensemble is pure again. (With one run and no reset, opts.noReset,
    // the state stays where the measurement left it.)
    measure(o, ev) {
      const p1 = this.p1; let ones = 0;
      for (const g of this.ghosts) { const one = g.meas() < (1 - g.r[2]) / 2; g.r = one && this.noReset ? [0, 0, -1] : [0, 0, 1]; if (one) ones++; }
      this.amp = 0;   // the readout interrupts the drive
      this.measures++;
      const got = o.want ? ones : this.N - ones, win = got >= this.N / 2, pts = win ? Math.round(400 * Math.max(1, this.combo) * got / this.N) : 0;
      if (win) { this.measWins++; this.score += pts; }
      ev.push({ type: 'measure', o, ones, N: this.N, one: ones >= this.N / 2, p1, win, pts });
    }
    // Active reset: every run back to |0⟩, a new shot of the experiment; the ensemble is pure again.
    reset(o, ev) { const p = this.purity; this.r = [0, 0, 1]; this.amp = 0; ev.push({ type: 'reset', o, purity: p }); }
    // The X gate on every run: a π turn about x, so θ → π − θ.
    applyX(o, ev) {
      const from = this.theta; for (const g of this.ghosts) g.r = unit(rotate(g.r, [Math.PI, 0, 0], 1)); this.gates++; this.score += 50;
      ev.push({ type: 'token', o, from, to: this.theta });
    }
    // A T₁+ token: T₁ grows by T1_UP (up to T1_MAX times where it started), so the runs stay together longer.
    power(o, ev) {
      const from = this.T1, up = UPGRADES[this.upgrades % UPGRADES.length];
      if (this.decay) this.T1 = Math.min(this.T1start * P.T1_MAX, Math.round(this.T1 * P.T1_UP * 10) / 10);
      this.upgrades++; const pts = 150 * Math.max(1, this.combo); this.score += pts;
      ev.push({ type: 'power', o, from, to: this.T1, up, pts, maxed: this.T1 >= this.T1start * P.T1_MAX - 1e-9 });
    }
    // A T₁− hazard: T₁ shrinks by T1_UP (down to T1_MIN times where it started), and the combo is lost.
    drain(o, ev) {
      const from = this.T1, hz = HAZARDS[this.drains % HAZARDS.length];
      if (this.decay) this.T1 = Math.max(Math.round(this.T1start * P.T1_MIN * 10) / 10, Math.round(this.T1 / P.T1_UP * 10) / 10);
      this.drains++; this.combo = 0;
      ev.push({ type: 'drain', o, from, to: this.T1, hz });
    }
    stars() {
      const f = this.total ? this.rings / this.total : 1;
      return f >= (this.mission.three ?? 0.9) - 1e-9 && this.hits === 0 ? 3 : f >= 0.7 ? 2 : 1;
    }

    /* ---------------- course ---------------- */
    add(o) { this.objects.push(o); if (isGood(o)) this.objects = this.objects.filter(c => c.type !== 'cloud' || !this.covers(c, o)); return o; }
    // Would cloud c sit on top of good object o (ring or token)?
    covers(c, o) { return Math.abs(o.t - c.t) < c.w + 0.15 && Math.abs(o.th - c.th) < c.rad + (o.rad || P.TOKEN) * 0.6 + 3 * D2R; }
    script(m) {
      const R = rng('mission:' + m.id), rr = P.MISSION_RING;
      let last = 0;
      for (const it of m.items) {
        const [k, t] = it, d = v => v * D2R;
        if (k === 'r') { this.add({ type: 'ring', t, th: d(it[2]), rad: rr }); this.total++; }
        else if (k === 'c') this.add({ type: 'cloud', t, th: d(it[2]), rad: d(it[3]), w: it[4] });
        else if (k === 'q') { this.quiz(t, QUIZ[it[2]](R), rr); this.total++; }
        else if (k === 'm') this.add({ type: 'gate', t, want: it[2] });
        else if (k === 'x') this.add({ type: 'token', t, th: d(it[2]), kind: 'X' });
        else if (k === 'z') this.add({ type: 'reset', t });
        else if (k === 'u') this.add({ type: 'token', t, th: d(it[2]), kind: 'T1' });
        else if (k === 'd') this.add({ type: 'drain', t, th: d(it[2]) });
        last = Math.max(last, t);
      }
      this.endT = last + 1.5;
      this.objects.sort((a, b) => a.t - b.t);
    }
    quiz(t, q, rr) {
      const rad = Math.min(rr, 11 * D2R);
      this.add({ type: 'ring', t, th: q.th, rad, quiz: q, correct: true });
      this.add({ type: 'ring', t, th: q.wrong, rad, quiz: q, correct: false });
    }
    // Time to fly from θ a to θ b with a reaction margin (the drive runs at about Ω; the margin shrinks with the level).
    reach(a, b, lvl) { return (0.75 - 0.3 * lvl) + Math.abs(b - a) / (P.OMEGA * 0.85); }
    // Keep AHEAD seconds of the endless course scheduled. Every ring is reachable from the one before, clouds stay off
    // the band between consecutive rings, and after a measurement the next ring is reachable from either outcome.
    generate() {
      const R = this.rand;
      while (this.nextT < this.t + P.AHEAD) {
        const lvl = Math.min(1, this.nextT / P.LEVEL_TIME), rr = P.RING0 + (P.RING_MIN - P.RING0) * lvl, roll = R();
        const clamp = th => Math.max(12 * D2R, Math.min(168 * D2R, th));
        if (this.N > 1 && this.nextT - this.lastReset > P.SHOT) {
          // a reset: a new shot starts from |0⟩, and the next ring is reachable from there
          const t = this.nextT + 0.4; this.add({ type: 'reset', t }); this.lastReset = t;
          this.walk = 0; this.nextT = t + 0.2;
          continue;
        }
        if (this.nextT > 8 && roll < 0.1 + 0.05 * lvl) {
          // a measurement gate, then a ring on the equator
          const t = this.nextT + 0.4; this.add({ type: 'gate', t, want: R() < 0.5 ? 1 : 0 });
          this.walk = Math.PI / 2; this.nextT = t + this.reach(0, Math.PI / 2, lvl) + 0.2;
          this.add({ type: 'ring', t: this.nextT, th: this.walk, rad: rr });
        } else if (this.nextT > 8 && roll < 0.22 + 0.06 * lvl) {
          // a quiz gate: reachable whichever ring is right
          // extra lead (2 s) so the question can be read before the rings arrive
          const q = QUIZ[Math.floor(R() * QUIZ.length)](R), t = this.nextT + this.reach(this.walk, q.th, lvl) + 2;
          this.quiz(t, q, rr); this.walk = q.th; this.nextT = t;
        } else if (this.nextT > 8 && roll < 0.3 + 0.05 * lvl) {
          // an X token on the way, and a ring where it sends you, sooner than you could fly there
          const t = this.nextT + this.reach(0, 0, lvl) + 0.3, th = clamp(this.walk), to = Math.PI - th;
          this.add({ type: 'token', t, th, kind: 'X' }); this.walk = to; this.nextT = t + 0.6;
          this.add({ type: 'ring', t: this.nextT, th: to, rad: rr });
        } else if (this.nextT > 6 && roll < 0.36 + 0.06 * lvl && this.decay && this.powerPlanned < 6) {
          // a T₁+ token near the path (at most six: enough to reach T1_MAX)
          const th = clamp(this.walk + (R() * 2 - 1) * 35 * D2R), t = this.nextT + this.reach(this.walk, th, lvl) + 0.2;
          this.add({ type: 'token', t, th, kind: 'T1' }); this.powerPlanned++; this.walk = th; this.nextT = t;
        } else {
          // a run of rings on a random walk, with clouds beside the path
          const n = 2 + Math.floor(R() * 3);
          for (let i = 0; i < n; i++) {
            const from = this.walk, to = clamp(from + (R() * 2 - 1) * (35 + 50 * lvl) * D2R), t = this.nextT + this.reach(from, to, lvl);
            this.add({ type: 'ring', t, th: to, rad: rr });
            if (R() < 0.35 + 0.4 * lvl && t - this.nextT > 1) this.cloudBeside(this.nextT, t, from, to, R);
            else if (this.decay && this.nextT > 10 && R() < 0.12 + 0.1 * lvl && t - this.nextT > 0.8) this.drainBeside(this.nextT, t, from, to, R);
            this.walk = to; this.nextT = t;
          }
        }
      }
    }
    // A T₁− hazard between times t0 and t1, well clear of the band from θ a to θ b (so it is always possible to dodge).
    drainBeside(t0, t1, a, b, R) {
      const m = P.POWER + 14 * D2R, lo = Math.min(a, b) - m, hi = Math.max(a, b) + m, room = [lo - 8 * D2R, Math.PI - 8 * D2R - hi];
      if (Math.max(...room) <= 0) return;
      const below = room[0] > room[1], th = below ? 8 * D2R + R() * room[0] : hi + R() * room[1];
      this.add({ type: 'drain', t: (t0 + t1) / 2, th });
    }
    // A cloud between times t0 and t1 that keeps clear of the band from θ a to θ b (with a margin) on its far side.
    cloudBeside(t0, t1, a, b, R) {
      const lo = Math.min(a, b) - 18 * D2R, hi = Math.max(a, b) + 18 * D2R, rad = (12 + 8 * R()) * D2R;
      const room = [lo, Math.PI - hi], side = room[0] > room[1] ? -1 : 1;
      if (Math.max(...room) < 2 * rad) return;
      const th = side < 0 ? lo - rad - R() * (lo - 2 * rad) : hi + rad + R() * (Math.PI - hi - 2 * rad);
      this.add({ type: 'cloud', t: (t0 + t1) / 2, th: Math.max(rad, Math.min(Math.PI - rad, th)), rad, w: Math.min(0.6, (t1 - t0) / 2 - 0.1) });
    }
  }

  const api = { P, D2R, dot, cross, norm, unit, fromAngles, toAngles, wrap, rotate, fidelity, angle, rng, onCircle, thetaOf,
    QUIZ, MISSIONS, UPGRADES, HAZARDS, World };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RabiEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
