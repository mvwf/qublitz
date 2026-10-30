/* rabi-engine.js: physics and course for Rabi Rider (rabi-rider.html). No dependencies. Loads as a
 * classic <script> (window.RabiEngine) or from Node (require), so tests/rabi-engine.test.js checks the
 * same code the game runs.
 *
 * The qubit is a pure state, a unit Bloch vector r = (sinθ cosφ, sinθ sinφ, cosθ), with |0⟩ at
 * θ = 0 (z = +1) and |1⟩ at θ = π. In the frame rotating with the drive,
 *
 *   H = (ħ/2) ω·σ,   ω = (Ω a_x, Ω a_y, Δ + Ω_z a_z),   dr/dt = ω × r,
 *
 * so the detuning Δ precesses the state about z (that is the scrolling), and a drive with phase α
 * rotates it about the axis (cos α, sin α, 0) at the Rabi rate Ω. A drive about that axis changes θ
 * at dθ/dt = −Ω sin(φ − α): +X lifts the state toward |1⟩ on the −y side of the sphere and pushes it
 * down on the +y side. The drive amplitudes a follow their targets with time constant TAU, like a
 * real, bandwidth-limited pulse generator.
 *
 * T₁ decay is treated as a watched qubit (a quantum trajectory): while no photon is detected the
 * state stays pure and slides toward |0⟩ at dθ/dt = −(γ/2) sinθ, and a photon is detected with
 * probability γ·P₁·dt per step, which resets the state to |0⟩ (a quantum jump). Averaged over many
 * runs this reproduces ordinary T₁ decay with γ = 1/T₁.
 *
 * Angles inside the engine are radians; the course is laid out in φ measured from the camera.
 */
(function (root) {
  'use strict';
  const TAU2 = 2 * Math.PI, D2R = Math.PI / 180;

  // ---------------- vectors ----------------
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

  // dθ/dt from a drive of phase α: −Ω sin(φ − α). Positive means toward |1⟩.
  const liftRate = (r, alpha, omega = 1) => -omega * Math.sin(toAngles(r).ph - alpha);
  // The drive phase that lifts fastest at the state's longitude (the autopilot's choice).
  const liftAxis = r => toAngles(r).ph + Math.PI / 2;
  // State fidelity |⟨t|ψ⟩|² between two pure states, from their Bloch vectors.
  const fidelity = (r, t) => (1 + dot(r, t)) / 2;
  // Angle on the sphere between two Bloch vectors.
  const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot(a, b))));
  // Angular distance from r to the meridian arc at longitude ph from θ0 to θ1.
  function meridianDist(r, ph, th0, th1) {
    const p = [Math.cos(ph), Math.sin(ph), 0], z = [0, 0, 1];
    // r's latitude along that meridian's great circle, measured from |0⟩ toward the +p side
    const t = Math.atan2(dot(r, p), dot(r, z));
    if (t >= th0 && t <= th1) return Math.asin(Math.min(1, Math.abs(r[0] * -Math.sin(ph) + r[1] * Math.cos(ph))));
    return Math.min(angle(r, fromAngles(th0, ph)), angle(r, fromAngles(th1, ph)));
  }

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
    OMEGA: 2.4,          // Rabi rate Ω [rad/s]: a π pulse takes 1.3 s
    OMEGA_Z: 1.1,        // extra z rotation from the Z control [rad/s]
    DELTA0: 0.55,        // starting detuning Δ [rad/s] (the scroll speed)
    DELTA_MAX: 1.45,
    DELTA_TIME: 110,     // s to approach DELTA_MAX
    TAU: 0.16,           // drive amplitude time constant [s]
    EASE: 12 * D2R,      // the autopilot eases off this close to a pole
    GAMMA: 1 / 14,       // T₁ decay rate γ = 1/T₁ [1/s], when decay is on
    HOME: -40 * D2R,     // where the camera keeps the qubit, relative to screen center
    SPAWN: 165 * D2R,    // objects appear this far ahead of the camera
    DESPAWN: -118 * D2R, // and leave this far behind
    HEARTS: 3,
    GRACE: 1.2,          // s of invulnerability after a hit or a jump
    RING0: 13 * D2R, RING_MIN: 8 * D2R,
    POWER_RAD: 9 * D2R,  // pickup radius of a power-up
    SLOW: 6,             // s of recalibration, which lowers Δ to SLOW_FACTOR of its value
    SLOW_FACTOR: 0.55,
    PURCELL: 10,         // s without T₁ decay after picking up a Purcell filter
    CLOSE: 4 * D2R,      // passing an obstacle this close (without touching it) is a close call
    STAGE: 30,           // s per stage in the endless game
  };

  // Named states for the bonus rings.
  const NAMED = [
    { th: Math.PI / 2, ph: 0, name: '|+⟩', ket: '(|0⟩ + |1⟩)/√2' },
    { th: Math.PI / 2, ph: Math.PI / 2, name: '|+i⟩', ket: '(|0⟩ + i|1⟩)/√2' },
    { th: Math.PI / 2, ph: Math.PI, name: '|−⟩', ket: '(|0⟩ − |1⟩)/√2' },
    { th: Math.PI / 2, ph: -Math.PI / 2, name: '|−i⟩', ket: '(|0⟩ − i|1⟩)/√2' },
    { th: Math.PI, ph: 0, name: '|1⟩', ket: 'the excited state' },
  ];

  /* Quiz gates: two rings at the same longitude, and the prompt describes one of them. Rings at one
   * longitude differ only in θ, so every question is about the polar angle. Each maker takes a seeded
   * generator and returns the right θ, a tempting wrong θ, and the explanation shown afterward. */
  const fmtDeg = th => `${Math.round(th / D2R)}°`;
  const fmtNum = x => String(+x.toFixed(2)).replace('-', '−');
  const pick = (R, xs) => xs[Math.floor(R() * xs.length)];
  const QUIZ = [
    R => { const p = pick(R, [0.1, 0.25, 0.75, 0.9]), th = 2 * Math.asin(Math.sqrt(p));
      return { prompt: `P(|1⟩) = ${p}`, th, wrong: Math.PI - th,
        why: `P(|1⟩) = sin²(θ/2), so P(|1⟩) = ${p} sits at θ = ${fmtDeg(th)}. The other ring had P(|1⟩) = ${fmtNum(1 - p)}.` }; },
    R => { const p = pick(R, [0.1, 0.25, 0.75, 0.9]), th = 2 * Math.acos(Math.sqrt(p));
      return { prompt: `P(|0⟩) = ${p}`, th, wrong: Math.PI - th,
        why: `P(|0⟩) = cos²(θ/2), so P(|0⟩) = ${p} sits at θ = ${fmtDeg(th)}. The closer to |0⟩ at the top, the likelier a 0.` }; },
    R => { const z = pick(R, [0.8, 0.5, -0.5, -0.8]), th = Math.acos(z);
      return { prompt: `⟨σz⟩ = ${fmtNum(z)}`, th, wrong: Math.PI - th,
        why: `⟨σz⟩ = cos θ = P(|0⟩) − P(|1⟩): +1 at the top, −1 at the bottom. So ⟨σz⟩ = ${fmtNum(z)} is at θ = ${fmtDeg(th)}.` }; },
    R => { const [deg, name] = pick(R, [[60, 'π/3'], [90, 'π/2'], [120, '2π/3'], [135, '3π/4']]), th = deg * D2R;
      return { prompt: `|0⟩ after a ${name} pulse`, th, wrong: th / 2,
        why: `A pulse of area ${name} turns the Bloch vector by ${deg}°, so θ = ${deg}°. Half of that, ${deg / 2}°, is the angle in the ket cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.` }; },
  ];

  /* Missions: short scripted courses, each teaching one idea. Items are placed by arrival, the qubit's
   * longitude φ in degrees (counting laps, so 450 is φ = 90° on the second lap) when it reaches them:
   *   ['r', a, θ]                ring                 ['c', a, θ, radius]          noise cloud
   *   ['p', a, θ0, θ1, width]    pillar               ['u', a, θ, kind]            power-up
   *   ['q', a, k]                quiz gate QUIZ[k]    ['n', a, name, life]         named ring
   * A named ring on the equator moves to the first arrival ≥ a where φ matches the state; |1⟩ is a band
   * along the bottom that appears at a and lasts `life` seconds. Stars: 3 for every ring with no hits
   * (or a `three` share of the rings, where luck is involved), 2 for 70 % of the rings, 1 for finishing. */
  const MISSIONS = [
    { id: 'flight', title: 'Flight school', mode: 'auto', delta: 0.45, start: 90,
      brief: 'Hold to rotate your qubit toward |0⟩ (up). Let go and the drive flips phase by π, rotating it toward |1⟩ (down). Fly through every ring.',
      learn: 'Every ring was a target state, and you reached each one by rotating the Bloch vector. That is all a drive does. Hold it on and the state keeps turning, |0⟩ → |1⟩ → |0⟩: Rabi oscillations.',
      items: [['r', 50, 85], ['r', 90, 65], ['r', 130, 48], ['r', 170, 65], ['r', 210, 95], ['r', 250, 125], ['r', 290, 140],
        ['r', 330, 115], ['r', 370, 80], ['r', 410, 50], ['r', 450, 90]] },
    { id: 'pi', title: 'The π pulse', mode: 'auto', delta: 0.5, start: 25,
      brief: 'A π pulse turns |0⟩ all the way into |1⟩. When the glowing |1⟩ band appears at the bottom, let go and dive into it, then climb back for the next ring.',
      learn: 'At this Rabi rate (Ω = 2.4 rad/s) a π pulse takes π/Ω ≈ 1.3 s, and that is how long each dive took. On a real transmon the same pulse takes 20–50 ns.',
      items: [['r', 50, 30], ['n', 90, '|1⟩', 3.5], ['r', 240, 30], ['n', 280, '|1⟩', 3.5], ['r', 430, 30], ['n', 470, '|1⟩', 3.5], ['r', 620, 40]] },
    { id: 'equator', title: 'Superposition', mode: 'auto', delta: 0.45, start: 70,
      brief: 'The equator holds the equal superpositions, where P(|1⟩) = ½. Stay on it between the noise clouds with short taps, and collect the four named states.',
      learn: 'All four named rings had P(|1⟩) = ½. They differ only in their phase φ, and the detuning keeps turning φ. A Hadamard gate takes |0⟩ to |+⟩ and |1⟩ to |−⟩.',
      items: [['r', 40, 80], ['n', 90, '|+i⟩'], ['c', 120, 45, 13], ['c', 150, 135, 13], ['r', 135, 90], ['n', 180, '|−⟩'], ['c', 215, 50, 13],
        ['c', 235, 130, 13], ['r', 225, 95], ['n', 270, '|−i⟩'], ['r', 315, 85], ['n', 360, '|+⟩'], ['r', 405, 90]] },
    { id: 'born', title: 'Measure up', mode: 'auto', delta: 0.42, start: 90,
      brief: 'Quiz gates: two purple rings, one answer. Read the question at the top and fly through the ring it describes. The grid lines are at θ = 45°, 90° and 135°.',
      learn: 'P(|1⟩) = sin²(θ/2) and ⟨σz⟩ = cos θ, so probability is not proportional to height on the sphere: P(|1⟩) = ¼ sits at θ = 60°, not 45°.',
      items: [['r', 40, 90], ['q', 100, 0], ['r', 170, 90], ['q', 230, 2], ['r', 300, 90], ['q', 360, 3], ['r', 430, 90], ['q', 490, 1],
        ['r', 560, 90], ['q', 620, 0]] },
    { id: 'axes', title: 'Pick an axis', mode: 'pilot', delta: 0.38, start: 90,
      brief: 'Pilot mode: you choose the drive axis. +X lifts the state on the +y side of the sphere and pushes it down on the −y side. Watch the arrows on the drive pad as φ turns.',
      learn: 'dθ/dt = −Ω sin(φ − α): the same pulse does different things to states with different phases. That is why gates are defined against a reference phase, and why phase errors matter.',
      items: [['r', 50, 55], ['r', 110, 120], ['r', 170, 60], ['r', 230, 125], ['r', 290, 60], ['r', 350, 120], ['r', 410, 55], ['r', 470, 110],
        ['r', 530, 70], ['r', 590, 90]] },
    { id: 'noise', title: 'Noise', mode: 'auto', delta: 0.5, start: 90,
      brief: 'Noise clouds and pillars each cost one heart of coherence. A pillar that reaches a pole covers the whole pole, so go around it through the gap.',
      learn: 'Each heart stands for coherence leaking away. On real chips, defects in the materials (two-level systems, the flickering clouds) are one of the main sources of loss.',
      items: [['r', 40, 90], ['c', 80, 55, 14], ['r', 100, 120], ['p', 150, 0, 80, 7], ['r', 150, 130], ['c', 195, 145, 13], ['r', 210, 80],
        ['p', 260, 100, 180, 7], ['r', 260, 50], ['c', 300, 40, 13], ['c', 300, 140, 13], ['r', 300, 90], ['p', 350, 0, 70, 7], ['r', 350, 125],
        ['p', 400, 110, 180, 7], ['r', 400, 55], ['r', 440, 90]] },
    { id: 'tools', title: 'Lab tricks', mode: 'auto', delta: 0.55, start: 90, decay: true, hearts: 2,
      brief: 'Three power-ups. π (spin echo) blocks the next hit. Δ↓ (recalibrate) slows the precession for 6 s. T₁↑ (Purcell filter) stops quantum jumps for 10 s, so you can skim near |1⟩.',
      learn: 'A spin echo flips the qubit halfway through so slow noise cancels itself. Recalibrating moves the drive closer to the qubit frequency, which shrinks Δ. A Purcell filter blocks the qubit from leaking energy into its readout line, which lengthens T₁.',
      items: [['u', 50, 90, 'echo'], ['c', 110, 35, 17], ['c', 110, 75, 17], ['c', 110, 115, 17], ['r', 150, 150], ['u', 200, 100, 'calib'],
        ['r', 235, 50], ['r', 260, 130], ['r', 285, 50], ['r', 310, 130], ['u', 360, 120, 'purcell'], ['r', 400, 150], ['r', 440, 155],
        ['r', 480, 150], ['r', 520, 145]] },
    { id: 'jumps', title: 'Quantum jumps', mode: 'auto', delta: 0.5, start: 90, decay: true, gamma: 1 / 6, three: 0.8,
      brief: 'T₁ decay is on, and strong. The closer you fly to |1⟩, the more often a photon escapes and the qubit jumps to |0⟩. Jumps cost no hearts, but they throw you to the top.',
      learn: 'The jump rate is γ·P(|1⟩). Between jumps the state still drifts toward |0⟩, because not seeing a photon is information too. Averaged over many flights this is plain exponential decay, e^(−t/T₁).',
      items: [['r', 50, 120], ['r', 100, 145], ['r', 150, 150], ['r', 200, 140], ['r', 250, 150], ['r', 300, 135], ['r', 350, 150], ['r', 400, 145],
        ['r', 450, 150], ['r', 500, 140]] },
    { id: 'exam', title: 'Final exam', mode: 'pilot', delta: 0.42, start: 90,
      brief: 'Everything at once, in Pilot mode: named states, quiz gates, noise, pillars and a spin echo. Use the arrows on the drive pad.',
      learn: 'You steered a qubit with real drive physics: rotations about X, Y and Z, the phase dependence of a pulse, and the Born rule. That is the toolkit behind every single-qubit gate.',
      items: [['r', 40, 90], ['n', 90, '|+i⟩'], ['c', 130, 50, 13], ['r', 140, 120], ['q', 200, 2], ['p', 260, 0, 75, 7], ['r', 260, 125],
        ['u', 300, 110, 'echo'], ['n', 360, '|+⟩'], ['c', 400, 130, 14], ['r', 410, 70], ['p', 460, 105, 180, 7], ['r', 460, 55], ['q', 530, 0],
        ['r', 590, 90], ['n', 630, '|−i⟩'], ['r', 680, 100], ['n', 720, '|+⟩']] },
  ];
  const isGood = o => o.type === 'ring' || o.type === 'power';   // things to collect, which obstacles never cover

  /* ---------------- the world ----------------
   * opts: { seed, mode: 'auto'|'pilot', decay: bool, mission: id }
   * step(input, dt): input = { hold } in autopilot, { x, y, z } targets in [−1, 1] in pilot mode.
   * In autopilot the drive phase is locked to the state's longitude, so the drive only changes θ:
   * amplitude a = +1 pushes toward |1⟩ at the full Rabi rate, a = −1 toward |0⟩.
   * Each step returns a list of events: ring, quiz, quizmiss, power, block, close, heal, miss, hit, jump,
   * stage, clear (a mission finished) and over.
   * The course is generated ahead as a queue of objects at course positions cs (radians of scroll);
   * each one is placed SPAWN ahead of the camera when the scroll reaches it.
   */
  class World {
    constructor(opts = {}) {
      this.mode = opts.mode === 'pilot' ? 'pilot' : 'auto';
      this.decay = opts.decay !== false;
      this.pace = this.mode === 'pilot' ? 0.8 : 1;     // pilot mode scrolls a little slower
      this.rand = rng('course:' + (opts.seed ?? 'random'));
      this.jumpRand = rng('jumps:' + (opts.seed ?? 'random'));
      this.r = fromAngles(70 * D2R, 0);                 // a little below the equator, at +x
      this.camPh = wrap(-P.HOME);
      this.t = 0; this.s = 0;                           // time, and distance scrolled
      this.amp = { a: 0.2, x: 0, y: 0, z: 0 };
      this.objects = []; this.queue = []; this.nextS = 60 * D2R; this.walk = 75 * D2R; this.segment = 0;
      this.score = 0; this.combo = 0; this.rings = 0; this.perfects = 0; this.jumps = 0; this.hits = 0;
      this.hearts = P.HEARTS; this.grace = 0; this.over = false;
      this.quizRight = 0; this.quizWrong = 0; this.closes = 0; this.powers = 0; this.blocked = 0; this.stage = 0;
      this.shield = false; this.slowT = 0; this.slowK = 1; this.purcellT = 0;
      this.mission = null; this.cleared = false;
      const m = opts.mission && MISSIONS.find(x => x.id === opts.mission);
      if (m) {
        this.mission = m; this.mode = m.mode; this.decay = !!m.decay; this.hearts = m.hearts ?? P.HEARTS;
        this.r = fromAngles(m.start * D2R, 0);
        this.jumpRand = rng('jumps:mission:' + m.id);
        this.script(m);
      }
    }
    get level() { return Math.min(1, this.t / 150); }
    get delta() {
      const base = this.mission ? this.mission.delta : this.pace * (P.DELTA0 + (P.DELTA_MAX - P.DELTA0) * (1 - Math.exp(-this.t / P.DELTA_TIME)));
      return base * this.slowK;   // a recalibration lowers Δ for a while
    }
    get ringRadius() { return P.RING_MIN + (P.RING0 - P.RING_MIN) * Math.exp(-this.t / 120); }
    get angles() { return toAngles(this.r); }
    rel(ph) { return wrap(ph - this.camPh); }             // longitude relative to the camera

    // The rotation vector ω for the current drive amplitudes.
    omega() {
      const D = this.delta, A = this.amp;
      if (this.mode === 'auto') {
        // the autopilot eases off within EASE of the pole it is driving toward, so the state settles there
        // instead of crossing it (crossing a pole flips φ by 180° and would throw the camera around)
        const al = liftAxis(this.r), th = toAngles(this.r).th;
        const a = A.a * Math.min(1, (A.a < 0 ? th : Math.PI - th) / P.EASE);
        return [P.OMEGA * a * Math.cos(al), P.OMEGA * a * Math.sin(al), D];
      }
      return [P.OMEGA * A.x, P.OMEGA * A.y, D + P.OMEGA_Z * A.z];
    }

    step(input, dt) {
      const ev = [];
      if (this.over) return ev;
      // drive amplitudes relax toward their targets
      const f = 1 - Math.exp(-dt / P.TAU), A = this.amp;
      // autopilot: holding drives toward |0⟩ (up on the map), releasing flips the drive phase by π
      if (this.mode === 'auto') A.a += ((input.hold ? -1 : 1) - A.a) * f;
      else for (const k of ['x', 'y', 'z']) A[k] += (Math.max(-1, Math.min(1, input[k] || 0)) - A[k]) * f;
      // unitary part, exact for this step
      this.r = rotate(this.r, this.omega(), dt);
      // watched T₁: a jump to |0⟩ with probability γ P₁ dt, otherwise the no-jump drift toward |0⟩
      if (this.decay && this.purcellT <= 0) {
        const g = this.mission?.gamma ?? P.GAMMA, z = this.r[2];
        if (this.jumpRand() < g * (1 - z) / 2 * dt) {
          this.r = fromAngles(1e-4, toAngles(this.r).ph);   // |0⟩, keeping φ so the camera stays put
          this.jumps++; this.combo = 0;
          this.grace = Math.max(this.grace, P.GRACE);
          ev.push({ type: 'jump' });
        } else {
          const h = (g / 2) * dt;
          this.r = [this.r[0] * (1 - h * z), this.r[1] * (1 - h * z), this.r[2] + h * (1 - z * z)];
        }
      }
      this.r = unit(this.r);
      // camera: scroll with Δ, and pull the qubit back toward its home position
      const D = this.delta, off = wrap(toAngles(this.r).ph - this.camPh - P.HOME);
      const k = Math.abs(off) > Math.PI / 2 ? 9 : 2.2;      // quick pan after a trip over a pole
      this.camPh = wrap(this.camPh + (D + k * off) * dt);
      this.t += dt; this.s += D * dt;
      this.score += 4 * D * dt / (18 * D2R);               // distance: 4 points per 18° of scroll
      this.grace = Math.max(0, this.grace - dt);
      this.slowT = Math.max(0, this.slowT - dt); this.purcellT = Math.max(0, this.purcellT - dt);
      this.slowK += ((this.slowT > 0 ? P.SLOW_FACTOR : 1) - this.slowK) * (1 - Math.exp(-dt / 0.4));
      if (!this.mission && Math.floor(this.t / P.STAGE) > this.stage) { this.stage++; ev.push({ type: 'stage', n: this.stage + 1 }); }
      this.spawn();
      this.collide(ev);
      this.cull(ev);
      if (this.hearts <= 0) { this.over = true; ev.push({ type: 'over' }); }
      else if (this.mission && this.s >= this.endS) { this.over = true; this.cleared = true; ev.push({ type: 'clear', stars: this.stars() }); }
      return ev;
    }

    /* ---------------- course ---------------- */
    // Queue o at course position cs; th is its polar angle, used for the overlap checks.
    push(cs, o) { this.queue.push({ cs, o }); }
    // Polar angle near `want` that keeps a ring clear of the clouds queued near course position cs.
    clear(cs, want, rad) {
      let th = want;
      for (const { cs: c, o } of this.queue) {
        if (o.type !== 'cloud' || Math.abs(c - cs) > 40 * D2R) continue;
        const sep = o.rad + rad + 5 * D2R;
        const d = angle(fromAngles(th, cs), fromAngles(o.th, c));
        if (d < sep) th = o.th + Math.sign(th - o.th || 1) * Math.sqrt(Math.max(0, sep * sep - (c - cs) ** 2 * Math.sin(th) ** 2));
      }
      return Math.max(22 * D2R, Math.min(158 * D2R, th));
    }
    generate() {
      const R = this.rand, lvl = this.level, rr = this.ringRadius, s0 = this.nextS;
      const roll = R(), kind = this.segment++ < 2 ? 'rings' : roll < 0.22 + 0.25 * lvl ? 'pillar' : roll < 0.34 + 0.25 * lvl ? 'named'
        : roll < 0.44 + 0.25 * lvl ? 'quiz' : 'rings';
      if (kind === 'pillar') {
        // a pillar reaching up from |1⟩ or down from |0⟩, with a ring in the gap; it usually grows from the
        // pole the qubit is nearer to, so parking at a pole is not a safe place to wait
        const nearOne = this.angles.th > Math.PI / 2, fromTop = R() < (nearOne ? 0.75 : 0.25), len = (55 + 45 * lvl + 20 * R()) * D2R;
        const th0 = fromTop ? Math.PI - len : 0, th1 = fromTop ? Math.PI : len;
        this.push(s0, { type: 'pillar', th0, th1, th: (th0 + th1) / 2, w: (6 + 3 * lvl) * D2R });
        this.walk = fromTop ? (Math.PI - len) / 2 : (Math.PI + len) / 2;
        this.push(s0, { type: 'ring', th: this.walk, rad: rr });
        this.nextS += (75 - 15 * lvl) * D2R;
      } else if (kind === 'named') {
        // a bonus ring on a named state, which sits at a fixed place on the sphere
        const n = NAMED[Math.floor(R() * NAMED.length)];
        this.push(s0, { type: 'ring', named: n, th: n.th, rad: rr * 1.2, bonus: true, fallback: this.walk });
        this.nextS += 45 * D2R;
      } else if (kind === 'quiz') {
        // a quiz gate, set a little further on so there is time to read the question and reach either ring
        const q = QUIZ[Math.floor(R() * QUIZ.length)](R);
        this.pushQuiz(s0 + 25 * D2R, q, rr);
        this.walk = q.th;
        this.nextS += 85 * D2R;
      } else {
        // a short run of rings on a smooth random walk, with noise clouds beside it
        const n = 2 + Math.floor(R() * 3), stepTh = (20 + 25 * lvl) * D2R, gap = 38 * D2R;
        for (let i = 0; i < n; i++) {
          const cs = s0 + i * gap;
          this.walk = this.clear(cs, Math.max(22 * D2R, Math.min(158 * D2R, this.walk + (R() * 2 - 1) * stepTh)), rr);
          this.push(cs, { type: 'ring', th: this.walk, rad: rr });
          if (R() < 0.35 + 0.4 * lvl) {
            const side = this.walk > Math.PI / 2 ? -1 : 1, rad = (9 + 6 * R()) * D2R;
            const th = Math.max(rad + 6 * D2R, Math.min(Math.PI - rad - 6 * D2R, this.walk + side * (rr + rad + (6 + 14 * R()) * D2R)));
            this.push(cs + gap / 2, { type: 'cloud', th, rad });
          } else if (R() < 0.2) {
            // a power-up on the way to the next ring; Purcell filters only matter when T₁ decay is on
            const k = R(), kind = this.decay && k < 0.25 ? 'purcell' : k < 0.6 ? 'echo' : 'calib';
            this.push(cs + gap / 2, { type: 'power', kind, th: this.walk, rad: P.POWER_RAD });
          }
        }
        this.nextS += (n * 38 + 30 - 10 * lvl) * D2R;
      }
      this.queue.sort((a, b) => a.cs - b.cs);
    }
    // Two rings at one course position: the answer to quiz q and the tempting wrong one.
    pushQuiz(cs, q, rr) {
      const rad = Math.min(rr, 11 * D2R);
      this.push(cs, { type: 'ring', quiz: q, correct: true, th: q.th, rad });
      this.push(cs, { type: 'ring', quiz: q, correct: false, th: q.wrong, rad });
    }
    // Queue a mission's items (see MISSIONS); nothing else is generated.
    script(m) {
      const R = rng('mission:' + m.id), lead = P.SPAWN - P.HOME, rr = (m.ring ?? 12) * D2R;
      let last = 0; this.total = 0;
      for (const it of m.items) {
        const [k, a0] = it, d = v => v * D2R;
        let a = d(a0);
        if (k === 'r') this.push(a - lead, { type: 'ring', th: d(it[2]), rad: rr });
        else if (k === 'c') this.push(a - lead, { type: 'cloud', th: d(it[2]), rad: d(it[3]) });
        else if (k === 'p') this.push(a - lead, { type: 'pillar', th0: d(it[2]), th1: d(it[3]), th: d(it[2] + it[3]) / 2, w: d(it[4]) });
        else if (k === 'u') this.push(a - lead, { type: 'power', kind: it[3], th: d(it[2]), rad: P.POWER_RAD });
        else if (k === 'q') this.pushQuiz(a - lead, QUIZ[it[2]](R), rr);
        else if (k === 'n') {
          const n = NAMED.find(x => x.name === it[2]), o = { type: 'ring', named: n, th: n.th, rad: rr * 1.2, bonus: true, fallback: Math.min(n.th, d(150)) };
          if (n.th >= Math.PI - 1e-9) this.push(a, Object.assign(o, { life: it[3] }));
          else { a += ((n.ph - a) % TAU2 + TAU2) % TAU2; this.push(a - lead, o); }
        }
        if (k === 'r' || k === 'q' || k === 'n') this.total++;
        last = Math.max(last, a);
      }
      this.endS = last + 25 * D2R; this.nextS = Infinity;
      this.queue.sort((x, y) => x.cs - y.cs);
    }
    stars() {
      const f = this.total ? this.rings / this.total : 1;
      return f >= (this.mission.three ?? 1) - 1e-9 && this.hits === 0 ? 3 : f >= 0.7 ? 2 : 1;
    }
    spawn() {
      while (this.nextS < this.s + 1.2) this.generate();
      while (this.queue.length && this.queue[0].cs <= this.s) {
        const { cs, o } = this.queue.shift();
        this.place(o, this.s - cs);
      }
    }
    place(o, late) {
      if (o.named) {
        const n = o.named;
        if (n.th >= Math.PI - 1e-9) {                   // |1⟩ is at the pole: it is "everywhere" along the top
          if (this.objects.some(x => x.pole)) { o = { type: 'ring', th: o.fallback, rad: this.ringRadius }; }
          else { Object.assign(o, { name: n.name, ket: n.ket, ph: 0, v: [0, 0, -1], pole: true, born: this.t }); this.objects.push(o); this.unblock(o); return; }
        } else {
          const ahead = this.rel(n.ph);
          if (ahead > 40 * D2R) { Object.assign(o, { name: n.name, ket: n.ket, ph: n.ph, v: fromAngles(n.th, n.ph) }); this.objects.push(o); this.unblock(o); return; }
          o = { type: 'ring', th: o.fallback, rad: this.ringRadius };   // not ahead of us right now
        }
      }
      o.ph = wrap(this.camPh + P.SPAWN - late);
      o.v = fromAngles(o.th, o.ph);
      this.objects.push(o);
      this.unblock(o);
    }
    // Rings and power-ups always win: an obstacle that would cover one is not placed (or is removed).
    blocks(ring, ob) {
      const m = ring.rad * 0.6 + 3 * D2R;
      return ob.type === 'cloud' ? angle(ring.v, ob.v) < ob.rad + m : meridianDist(ring.v, ob.ph, ob.th0, ob.th1) < ob.w + m;
    }
    unblock(o) {
      if (isGood(o)) this.objects = this.objects.filter(q => isGood(q) || !this.blocks(o, q));
      else if (this.objects.some(q => isGood(q) && !q.done && this.blocks(q, o))) this.objects.pop();
    }
    collide(ev) {
      const me = this.rel(toAngles(this.r).ph);
      for (const o of this.objects) {
        if (o.done) continue;
        if (o.type === 'power') {
          if (angle(this.r, o.v) < o.rad) { o.done = true; this.collect(o, ev); }
        } else if (o.type === 'ring') {
          const a = angle(this.r, o.v);
          if (a < o.rad && o.quiz) this.answer(o, ev);
          else if (a < o.rad) {
            o.done = true; this.combo = Math.min(10, this.combo + 1);
            const perfect = a < o.rad / 3;
            const pts = Math.round((o.bonus ? 250 : 100) * this.combo * (perfect ? 1.5 : 1));
            this.score += pts; if (perfect) this.perfects++;
            ev.push({ type: 'ring', o, pts, perfect, fidelity: fidelity(this.r, o.v), combo: this.combo });
            this.counted(ev);
          }
        } else {
          const d = o.type === 'cloud' ? angle(this.r, o.v) - o.rad : meridianDist(this.r, o.ph, o.th0, o.th1) - o.w;
          if (d < 0 && this.grace <= 0) {
            o.struck = true; this.grace = P.GRACE;
            if (this.shield) { this.shield = false; this.blocked++; ev.push({ type: 'block', o }); }   // the spin echo takes the hit
            else { this.hearts--; this.hits++; this.combo = 0; ev.push({ type: 'hit', o }); }
          }
          o.near = Math.min(o.near ?? Infinity, d);
          // once it is well behind, an obstacle passed closely without a touch is a close call
          if (!o.passed && this.rel(o.ph) < me - 15 * D2R) {
            o.passed = true;
            if (!o.struck && o.near >= 0 && o.near < P.CLOSE) {
              const pts = 25 * Math.max(1, this.combo); this.score += pts; this.closes++; ev.push({ type: 'close', o, pts });
            }
          }
        }
      }
    }
    // A target state reached: every 10th one restores a heart.
    counted(ev) {
      this.rings++;
      if (this.rings % 10 === 0 && this.hearts < P.HEARTS) { this.hearts++; ev.push({ type: 'heal' }); }
    }
    // Flying through either ring of a quiz gate answers it, and both rings go.
    answer(o, ev) {
      const q = o.quiz; q.answered = true;
      for (const x of this.objects) if (x.quiz === q) x.done = true;
      let pts = 0;
      if (o.correct) { this.quizRight++; this.combo = Math.min(10, this.combo + 1); pts = 300 * this.combo; this.score += pts; }
      else { this.quizWrong++; this.combo = 0; }
      ev.push({ type: 'quiz', o, q, correct: o.correct, pts, combo: this.combo });
      if (o.correct) this.counted(ev);
    }
    collect(o, ev) {
      if (o.kind === 'echo') this.shield = true;
      else if (o.kind === 'calib') this.slowT = P.SLOW;
      else this.purcellT = P.PURCELL;
      this.powers++; this.score += 50;
      ev.push({ type: 'power', o, kind: o.kind });
    }
    cull(ev) {
      this.objects = this.objects.filter(o => {
        const rel = this.rel(o.ph);
        const gone = o.pole ? this.t - o.born > (o.life ?? 8) : rel < P.DESPAWN && rel > -Math.PI + 0.2;
        if (gone && o.type === 'ring' && !o.done && o.quiz) { if (o.correct) { this.combo = 0; ev.push({ type: 'quizmiss', q: o.quiz }); } }
        else if (gone && o.type === 'ring' && !o.done && !o.bonus) { this.combo = 0; ev.push({ type: 'miss', o }); }
        return !gone;
      });
    }
  }

  const api = { P, D2R, dot, cross, norm, unit, fromAngles, toAngles, wrap, rotate, liftRate, liftAxis, fidelity, angle,
    meridianDist, rng, NAMED, QUIZ, MISSIONS, World };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RabiEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
