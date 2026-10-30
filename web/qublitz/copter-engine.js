/* copter-engine.js: physics and course for Qubit Copter (qubit-copter.html). No dependencies beyond
 * rabi-engine.js (for the vector helpers). Loads as a classic <script> (window.CopterEngine)
 * or from Node (require), so tests/copter-engine.test.js checks the same code the game runs.
 *
 * The qubit is driven exactly on resonance, so in the rotating frame there is no free precession:
 *
 *   H = (ħΩ/2)(a_x σx + a_y σy),   dr/dt = ω × r,   ω = Ω (a_x, a_y, 0),
 *
 * and the state moves only while you drive it. The drive phase α sets the axis (cos α, sin α, 0); the
 * amplitudes follow their targets with time constant TAU, like a bandwidth-limited pulse.
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
    TAU: 0.12,            // drive amplitude time constant [s]
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

  class World {
    // opts: { seed, mode: 'targets' | 'circuit' }
    constructor(opts = {}) {
      this.mode = opts.mode === 'circuit' ? 'circuit' : 'targets';
      this.rand = rng('copter:' + this.mode + ':' + (opts.seed ?? 'random'));
      this.r = [0, 0, 1]; this.amp = { x: 0, y: 0 };
      this.t = 0; this.score = 0; this.combo = 0; this.hits = 0; this.misses = 0; this.perfects = 0; this.wallHits = 0;
      this.hearts = P.HEARTS; this.grace = 0; this.over = false;
      this.targets = []; this.walls = [];
      this.plan = { t: 1.5, v: [0, 0, 1] };   // the last scheduled target (the start state, to begin with)
      this.generate();
    }
    get level() { return Math.min(1, this.t / P.LEVEL_TIME); }
    get radius() { return P.RAD0 + (P.RAD_MIN - P.RAD0) * this.level; }
    omega() { return [P.OMEGA * this.amp.x, P.OMEGA * this.amp.y, 0]; }

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
      const f = 1 - Math.exp(-dt / P.TAU), A = this.amp;
      for (const k of ['x', 'y']) A[k] += (Math.max(-1, Math.min(1, input[k] || 0)) - A[k]) * f;
      const n = Math.hypot(A.x, A.y); if (n > 1) { A.x /= n; A.y /= n; }   // one drive: its amplitude is at most 1
      this.r = unit(rotate(this.r, this.omega(), dt));
      this.t += dt; this.grace = Math.max(0, this.grace - dt);
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
      let r = this.r.slice(); const a = { ...this.amp }, out = [r], dt = secs / n, f = 1 - Math.exp(-dt / P.TAU);
      for (let i = 0; i < n; i++) {
        a.x += ((input.x || 0) - a.x) * f; a.y += ((input.y || 0) - a.y) * f;
        r = rotate(r, [P.OMEGA * a.x, P.OMEGA * a.y, 0], dt); out.push(r);
      }
      return out;
    }
  }

  const api = { P, steerToward, World, GATES, applyGate, gateArc, stateName, fromAngles };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CopterEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
