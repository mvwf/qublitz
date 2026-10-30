/* copter-engine.js: physics and course for Qubit Copter (qubit-copter.html). No dependencies beyond
 * rabi-engine.js (for the vector helpers). Loads as a classic <script> (window.CopterEngine)
 * or from Node (require), so tests/copter-engine.test.js checks the same code the game runs.
 *
 * The qubit is driven exactly on resonance, so in the rotating frame there is no free precession:
 *
 *   H = (ħΩ/2)(a_x σx + a_y σy + a_z σz),   dr/dt = ω × r,   ω = Ω (a_x, a_y, a_z),
 *
 * and the state moves only while you drive it. The drive phase α sets the axis (cos α, sin α, 0). The z term
 * is off unless you ask for it: holding Z detunes the drive by Ω for as long as you hold, so the state
 * precesses about z only then. In the game modes the amplitudes follow their targets with time constant TAU,
 * like a bandwidth-limited pulse; in Learn and Free flight they switch instantly, so every move is a square
 * pulse that the Virtual Qubit Lab can replay (labSequence).
 *
 * Time scrolls. Targets are states the qubit must be at when their time comes: on the page each one is a
 * window on the three projections ⟨σx⟩ = x, ⟨σy⟩ = y, ⟨σz⟩ = z and a ring on the sphere. Walls forbid a
 * band of one projection for a while, which on the sphere is a cap cut off by a plane.
 */
(function (root) {
  'use strict';
  const R = typeof module === 'object' && module.exports ? require('./rabi-engine.js') : root.RabiEngine;
  const { D2R, dot, cross, unit, angle, rotate, rng, fromAngles } = R;

  // Single-qubit gates as Bloch rotations (up to a global phase): exp(−iα n·σ/2) turns r by α about n.
  const R2 = Math.SQRT1_2;
  const GATES = {
    X: { label: 'X', axis: [1, 0, 0], angle: Math.PI, what: 'a half turn about x' },
    Y: { label: 'Y', axis: [0, 1, 0], angle: Math.PI, what: 'a half turn about y' },
    Z: { label: 'Z', axis: [0, 0, 1], angle: Math.PI, what: 'a half turn about z' },
    H: { label: 'H', axis: [R2, 0, R2], angle: Math.PI, what: 'a half turn about the axis halfway between x and z' },
    S: { label: 'S', axis: [0, 0, 1], angle: Math.PI / 2, what: 'a quarter turn about z' },
    T: { label: 'T', axis: [0, 0, 1], angle: Math.PI / 4, what: 'an eighth of a turn about z' },
    SX: { label: '√X', axis: [1, 0, 0], angle: Math.PI / 2, what: 'a quarter turn about x' },
    SY: { label: '√Y', axis: [0, 1, 0], angle: Math.PI / 2, what: 'a quarter turn about y' },
    RX4: { label: 'Rx(π/4)', axis: [1, 0, 0], angle: Math.PI / 4, what: 'an eighth of a turn about x' },
    RY3: { label: 'Ry(π/3)', axis: [0, 1, 0], angle: Math.PI / 3, what: 'a sixth of a turn about y' },
  };
  const applyGate = (r, k) => unit(rotate(r, GATES[k].axis.map(c => c * GATES[k].angle), 1));
  const gateArc = (r, k, n = 32) => Array.from({ length: n + 1 }, (_, i) => rotate(r, GATES[k].axis.map(c => c * GATES[k].angle * i / n), 1));
  const KETS = [[[0, 0, 1], '|0⟩'], [[0, 0, -1], '|1⟩'], [[1, 0, 0], '|+⟩'], [[-1, 0, 0], '|−⟩'], [[0, 1, 0], '|+i⟩'], [[0, -1, 0], '|−i⟩']];
  // One of the six cardinal states by name, or the amplitudes cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.
  function stateName(r) {
    for (const [v, n] of KETS) if (dot(r, v) > 0.9995) return n;
    const th = Math.acos(Math.max(-1, Math.min(1, r[2]))), ph = Math.round(((Math.atan2(r[1], r[0]) / D2R) % 360 + 360) % 360);
    return `${Math.cos(th / 2).toFixed(2)}|0⟩ + ${ph ? `e^{i${ph}°}` : ''}${Math.sin(th / 2).toFixed(2)}|1⟩`;
  }

  const P = {
    OMEGA: 1.8,           // Rabi rate at full drive [rad/s]: a π pulse takes 1.75 s
    TAU: 0.12,            // drive amplitude time constant [s] (game modes; Learn and Free flight use square pulses)
    WIN: 8 * D2R,         // a lesson is done when the state is this close to the goal
    SETTLE: 0.35,         // s it has to stay there, not driving
    RAD0: 20 * D2R,       // target radius at the start (angle on the sphere)
    RAD_MIN: 11 * D2R,
    LEVEL_TIME: 150,      // s for the course to reach full difficulty
    AHEAD: 9,             // s of course generated ahead
    HEARTS: 3,
    GRACE: 1.0,           // s of invulnerability after a wall hit
    WALL_WARN: 3,         // walls are visible at least this long before they start
  };

  // The equatorial drive (a_x, a_y) that carries r to p. A rotation about an equatorial axis n keeps r·n fixed,
  // so it can join r and p exactly when (r − p)·n = 0: n along (r − p) × z. (If r and p differ only in z, they are
  // mirror images through the equator, and the axis pointing at their longitude does it.) The drive turns the way
  // that is shorter around that circle and eases off as the turn left shrinks.
  function steerToward(r, p) {
    if (angle(r, p) < 1e-4) return { x: 0, y: 0 };
    let nx = r[1] - p[1], ny = p[0] - r[0];
    if (Math.hypot(nx, ny) < 1e-6) { nx = r[0]; ny = r[1]; if (Math.hypot(nx, ny) < 1e-6) { nx = 1; ny = 0; } }   // pole to pole: any axis
    const m = Math.hypot(nx, ny); nx /= m; ny /= m;
    const n = [nx, ny, 0], rp = [0, 1, 2].map(i => r[i] - dot(r, n) * n[i]), pp = [0, 1, 2].map(i => p[i] - dot(p, n) * n[i]);
    const turn = Math.atan2(dot(n, cross(rp, pp)), dot(rp, pp));   // signed angle left to turn about n
    const k = Math.max(-1, Math.min(1, turn * 1.5));
    return { x: nx * k, y: ny * k };
  }

  /* Learn mode: short tasks, each with a start state, a goal state and the axes you may use. No clock. */
  const K = Object.fromEntries(KETS.map(([v, n]) => [n, v]));
  const LESSONS = [
    { id: 'flip', title: 'Flip the bit', start: '|0⟩', goal: '|1⟩', axes: ['x'],
      brief: 'Hold <b>X</b> (or press <b>π</b>). The drive turns the state about the x axis, so it swings from |0⟩ over to |1⟩. Watch the three strips.',
      learn: 'That half turn is a <b>π pulse</b>, the quantum NOT gate X. ⟨σz⟩ went from +1 to −1 while <b>⟨σx⟩ never moved</b>: a rotation leaves the part of the state along its own axis alone. Keep holding and it comes back: that is a Rabi oscillation.' },
    { id: 'super', title: 'Superposition', start: '|0⟩', goal: '|+⟩', axes: ['y'],
      brief: 'Turn about <b>Y</b> until you reach |+⟩ on the equator. Try the <b>π/2</b> button: a quarter turn.',
      learn: 'A π/2 pulse makes |+⟩ = (|0⟩ + |1⟩)/√2. On the equator ⟨σz⟩ = 0, so P(1) = (1 − ⟨σz⟩)/2 = ½: a measurement gives 0 or 1 with even odds.' },
    { id: 'phase', title: 'Phase is a direction', start: '|+⟩', goal: '|+i⟩', axes: ['z'],
      brief: 'Now turn about <b>Z</b>. Keep an eye on the ⟨σz⟩ strip, which is P(1) in disguise.',
      learn: 'A Z rotation changes only the <b>relative phase</b> φ. P(1) did not budge, yet |+⟩ and |+i⟩ = (|0⟩ + i|1⟩)/√2 are different states: measuring along x or y tells them apart. In a real lab Z is free, a <b>virtual Z</b>: you just shift the phase of every later pulse.' },
    { id: 'sqx', title: 'Same distance, new phase', start: '|0⟩', goal: '|−i⟩', axes: ['x'],
      brief: 'Reach |−i⟩ using only <b>X</b>.',
      learn: 'A quarter turn about x (√X) takes |0⟩ to |−i⟩ = (|0⟩ − i|1⟩)/√2, while a quarter turn about y gave |+⟩. Same P(1), different phase: the <b>drive phase</b> (x or y) sets which superposition you make.' },
    { id: 'had', title: 'Make |−⟩ from |1⟩', start: '|1⟩', goal: '|−⟩', axes: ['x', 'y', 'z'],
      brief: 'Any axis you like. One quarter turn about the right axis does it.',
      learn: 'That is what a <b>Hadamard</b> does to |1⟩. H swaps the z and x axes (|0⟩ ↔ |+⟩, |1⟩ ↔ |−⟩): a half turn about the axis between x and z. Here a quarter turn about y had the same effect on |1⟩.' },
    { id: 'euler', title: 'Anywhere on the sphere', start: '|0⟩', goal: [60, 135], axes: ['x', 'y', 'z'],
      brief: 'The gold ring is off every axis: θ = 60°, φ = 135°. Tip: tilt it down with Y, then swing it around with Z.',
      learn: 'Two axes reach <b>any</b> pure state: a turn about y sets θ (and so P(1) = sin²(θ/2) = ¼ here), a turn about z sets φ. Every single-qubit gate is a rotation like these, which is how a quantum computer builds them out of pulses.' },
  ];
  const lessonVec = g => (typeof g === 'string' ? K[g].slice() : fromAngles(g[0] * D2R, g[1] * D2R));

  /* The moves of a Learn or Free flight run, as the Virtual Qubit Lab's pulse schedule. log is a list of square
   * segments { x, y, z, dur } (game seconds). The lab drives the σx and σy channels on resonance, which turn the
   * state exactly as the game's +X and +Y do; its Rabi rate is 2π f (f = Ω_R/2π in GHz, so angles per ns).
   * It has no σz channel, so Z turns are done the way real labs do them, as a virtual Z: a turn by Φ about z
   * commutes to the end of the sequence if every later drive axis is turned by −Φ, so the lab's P(1) comes out
   * the same and its final state differs only by the leftover Φ about z. A segment that drives z together with
   * x or y is split into short slices (a Trotter step every 3°). */
  function labSequence(log, opts = {}) {
    const f = opts.fGHz ?? 0.05, t0 = opts.start ?? 5, rate = 2 * Math.PI * f, slice = 3 * D2R, max = opts.max ?? 80;
    const out = []; let phi = 0, t = t0;
    const drive = (ax, ay, dur) => {
      const m = Math.hypot(ax, ay); if (m < 1e-9 || dur <= 0) return;
      const al = Math.atan2(ay, ax) - phi, ns = dur * P.OMEGA / rate;   // same turn angle Ω m dur in lab time
      const sx = m * Math.cos(al), sy = m * Math.sin(al), last = out[out.length - 1];
      if (last && Math.abs(last.sx - sx) < 1e-6 && Math.abs(last.sy - sy) < 1e-6 && Math.abs(last.start + last.dur - t) < 1e-9) last.dur += ns;
      else out.push({ sx, sy, start: t, dur: ns });
      t += ns;
    };
    for (const s of log) {
      const xy = Math.hypot(s.x, s.y) > 1e-9, z = Math.abs(s.z) > 1e-9;
      if (xy && z) {
        const n = Math.max(1, Math.ceil(P.OMEGA * Math.hypot(s.x, s.y, s.z) * s.dur / slice)), d = s.dur / n;
        for (let i = 0; i < n; i++) { drive(s.x, s.y, d); phi += P.OMEGA * s.z * d; }
      } else if (xy) drive(s.x, s.y, s.dur);
      else if (z) phi += P.OMEGA * s.z * s.dur;
    }
    const kept = out.slice(-max), cut = out.length - kept.length;
    return { pulses: kept, cut, phi: ((phi % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI, end: t, f };
  }

  class World {
    // opts: { seed, mode: 'targets' | 'circuit' | 'learn' | 'free', lesson: id }
    constructor(opts = {}) {
      this.mode = ['circuit', 'learn', 'free'].includes(opts.mode) ? opts.mode : 'targets';
      this.game = this.mode === 'targets' || this.mode === 'circuit';
      this.tau = this.game ? P.TAU : 0;
      this.rand = rng('copter:' + this.mode + ':' + (opts.seed ?? 'random'));
      this.r = [0, 0, 1]; this.amp = { x: 0, y: 0, z: 0 };
      this.t = 0; this.score = 0; this.combo = 0; this.hits = 0; this.misses = 0; this.perfects = 0; this.wallHits = 0;
      this.hearts = P.HEARTS; this.grace = 0; this.over = false;
      this.targets = []; this.walls = [];
      this.pulses = []; this.log = []; this.moves = 0;
      this.lesson = null; this.goal = null; this.settle = 0; this.done = false;
      if (this.mode === 'learn') {
        this.lesson = LESSONS.find(l => l.id === opts.lesson) || LESSONS[0];
        this.r = lessonVec(this.lesson.start); this.goal = lessonVec(this.lesson.goal);
      }
      this.plan = { t: 1.5, v: [0, 0, 1] };   // the last scheduled target (the start state, to begin with)
      if (this.game) this.generate();
    }
    get level() { return Math.min(1, this.t / P.LEVEL_TIME); }
    get radius() { return P.RAD0 + (P.RAD_MIN - P.RAD0) * this.level; }
    get driving() { return this.pulses.length > 0 || Math.hypot(this.amp.x, this.amp.y, this.amp.z) > 1e-3; }
    omega() { return [P.OMEGA * this.amp.x, P.OMEGA * this.amp.y, P.OMEGA * this.amp.z]; }
    // Queue a square pulse that turns the state by ang (radians, signed) about axis 'x', 'y' or 'z'.
    pulse(axis, ang) { this.pulses.push({ axis, sign: Math.sign(ang), left: Math.abs(ang) }); this.moves++; }
    // Put the state back (Free flight: at |0⟩) and forget the moves.
    reset(r) { this.r = (r || [0, 0, 1]).slice(); this.amp = { x: 0, y: 0, z: 0 }; this.pulses = []; this.log = []; this.settle = 0; }
    record(a, dur) {
      if (this.game) return;
      const last = this.log[this.log.length - 1];
      if (last && last.x === a.x && last.y === a.y && last.z === a.z) last.dur += dur;
      else if (a.x || a.y || a.z) this.log.push({ x: a.x, y: a.y, z: a.z, dur });
    }

    // Keep AHEAD seconds of targets (and the odd wall) scheduled.
    generate() {
      while (this.plan.t < this.t + P.AHEAD) {
        const R = this.rand, lvl = Math.min(1, this.plan.t / P.LEVEL_TIME), prev = this.plan.v;
        let v, label = null, gate = null;
        if (this.mode === 'circuit') {
          // the next gate of a random circuit, skipping gates that would leave the state where it is
          const pool = ['H', 'S', 'T', 'X', 'Y', 'Z', 'SX', 'SY', 'RX4', 'RY3', 'H', 'SX', 'SY'];
          for (let i = 0; i < 12; i++) { gate = pool[Math.floor(R() * pool.length)]; v = applyGate(prev, gate); if (angle(v, prev) > 0.2) break; }
          label = stateName(v);
        } else {
          // a random state 35°–100° from the last one
          const a = (35 + 65 * R() * (0.6 + 0.4 * lvl)) * D2R, ref = Math.abs(prev[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
          const e1 = unit(cross(prev, ref)), e2 = cross(prev, e1), s = 2 * Math.PI * R();
          v = unit([0, 1, 2].map(i => Math.cos(a) * prev[i] + Math.sin(a) * (Math.cos(s) * e1[i] + Math.sin(s) * e2[i])));
        }
        // time to get there: a reaction time plus the turn at about half the Rabi rate (only equatorial axes are
        // available, so the way is often longer than the straight arc), tightening as the level rises
        const gap = (1.2 - 0.4 * lvl) + angle(prev, v) / (P.OMEGA * (0.45 + 0.15 * lvl));
        const t = this.plan.t + gap;
        this.targets.push({ t, v, rad: P.RAD0 + (P.RAD_MIN - P.RAD0) * lvl, label, gate, from: prev });
        this.plan = { t, v };
        // now and then a wall between two targets that keeps both of them (and the way between) clear
        if (R() < 0.25 + 0.35 * lvl && gap > 1.8) this.addWall(t - gap, t, prev, v, R);
      }
    }
    addWall(t0, t1, a, b, R) {
      const axis = Math.floor(R() * 3), lo = Math.min(a[axis], b[axis]), hi = Math.max(a[axis], b[axis]);
      // forbid the far side of whichever end leaves room: values beyond both targets, with a margin
      const m = 0.3, room = [1 - (hi + m), (lo - m) + 1];
      const side = room[0] > room[1] ? 'hi' : 'lo';
      if (Math.max(...room) < 0.25) return;
      const band = side === 'hi' ? [hi + m, 1.01] : [-1.01, lo - m];
      this.walls.push({ t0: t0 + 0.3, t1: t1 - 0.3, axis, lo: band[0], hi: band[1] });
    }

    step(input, dt) {
      const ev = [];
      if (this.over) return ev;
      const A = this.amp;
      if (this.pulses.length) {
        // a queued pulse: full amplitude on one axis, stopped exactly at its angle
        const p = this.pulses[0], d = Math.min(dt, p.left / P.OMEGA);
        for (const k of ['x', 'y', 'z']) A[k] = k === p.axis ? p.sign : 0;
        this.r = unit(rotate(this.r, this.omega(), d)); this.record(A, d);
        p.left -= P.OMEGA * d; if (p.left < 1e-9) { this.pulses.shift(); if (!this.pulses.length) A.x = A.y = A.z = 0; }
      } else {
        const f = this.tau > 0 ? 1 - Math.exp(-dt / this.tau) : 1;
        for (const k of ['x', 'y', 'z']) A[k] += (Math.max(-1, Math.min(1, input[k] || 0)) - A[k]) * f;
        const n = Math.hypot(A.x, A.y); if (n > 1) { A.x /= n; A.y /= n; }   // one drive: its amplitude is at most 1
        this.r = unit(rotate(this.r, this.omega(), dt)); this.record(A, dt);
      }
      this.t += dt; this.grace = Math.max(0, this.grace - dt);
      if (!this.game) {
        // a lesson is done once the state rests within WIN of the goal
        if (this.goal && !this.done) {
          this.settle = angle(this.r, this.goal) < P.WIN && !this.driving ? this.settle + dt : 0;
          if (this.settle >= P.SETTLE) { this.done = true; ev.push({ type: 'goal', fidelity: (1 + dot(this.r, this.goal)) / 2 }); }
        }
        return ev;
      }
      // targets whose moment has come
      while (this.targets.length && this.targets[0].t <= this.t) {
        const o = this.targets.shift(), a = angle(this.r, o.v);
        if (a < o.rad) {
          this.hits++; this.combo = Math.min(10, this.combo + 1);
          const perfect = a < o.rad / 3; if (perfect) this.perfects++;
          const pts = Math.round(100 * this.combo * (perfect ? 1.5 : 1)); this.score += pts;
          ev.push({ type: 'hit', o, pts, perfect, fidelity: (1 + dot(this.r, o.v)) / 2 });
          if (this.hits % 10 === 0 && this.hearts < P.HEARTS) { this.hearts++; ev.push({ type: 'heal' }); }
        } else {
          this.misses++; this.combo = 0; this.hearts--;
          ev.push({ type: 'miss', o, off: a });
        }
      }
      // walls
      for (const w of this.walls) {
        if (this.t < w.t0 || this.t > w.t1 || this.grace > 0) continue;
        const v = this.r[w.axis];
        if (v > w.lo && v < w.hi) { this.hearts--; this.wallHits++; this.combo = 0; this.grace = P.GRACE; ev.push({ type: 'wall', w }); }
      }
      this.walls = this.walls.filter(w => w.t1 > this.t - 3);
      this.score += dt * 2;
      this.generate();
      if (this.hearts <= 0) { this.over = true; ev.push({ type: 'over' }); }
      return ev;
    }
    // Where the state would be after `secs` of holding drive (x, y) from now (amplitude ramp included).
    predict(input, secs, n = 20) {
      let r = this.r.slice(); const a = { ...this.amp }, out = [r], dt = secs / n, f = this.tau > 0 ? 1 - Math.exp(-dt / this.tau) : 1;
      for (let i = 0; i < n; i++) {
        a.x += ((input.x || 0) - a.x) * f; a.y += ((input.y || 0) - a.y) * f;
        a.z = (a.z ?? 0) + ((input.z || 0) - (a.z ?? 0)) * f;
        r = rotate(r, [P.OMEGA * a.x, P.OMEGA * a.y, P.OMEGA * a.z], dt); out.push(r);
      }
      return out;
    }
  }

  const api = { P, steerToward, World, GATES, applyGate, gateArc, stateName, fromAngles, LESSONS, lessonVec, labSequence, KETS };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CopterEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
