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
  };

  // Named states for the bonus rings.
  const NAMED = [
    { th: Math.PI / 2, ph: 0, name: '|+⟩', ket: '(|0⟩ + |1⟩)/√2' },
    { th: Math.PI / 2, ph: Math.PI / 2, name: '|+i⟩', ket: '(|0⟩ + i|1⟩)/√2' },
    { th: Math.PI / 2, ph: Math.PI, name: '|−⟩', ket: '(|0⟩ − |1⟩)/√2' },
    { th: Math.PI / 2, ph: -Math.PI / 2, name: '|−i⟩', ket: '(|0⟩ − i|1⟩)/√2' },
    { th: Math.PI, ph: 0, name: '|1⟩', ket: 'the excited state' },
  ];

  /* ---------------- the world ----------------
   * opts: { seed, mode: 'auto'|'pilot', decay: bool }
   * step(input, dt): input = { hold } in autopilot, { x, y, z } targets in [−1, 1] in pilot mode.
   * In autopilot the drive phase is locked to the state's longitude, so the drive only changes θ:
   * amplitude a = +1 pushes toward |1⟩ at the full Rabi rate, a = −1 toward |0⟩.
   * Each step returns a list of events: ring, heal, miss, hit, jump, over.
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
    }
    get level() { return Math.min(1, this.t / 150); }
    get delta() { return this.pace * (P.DELTA0 + (P.DELTA_MAX - P.DELTA0) * (1 - Math.exp(-this.t / P.DELTA_TIME))); }
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
      if (this.decay) {
        const g = P.GAMMA, z = this.r[2];
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
      this.spawn();
      this.collide(ev);
      this.cull(ev);
      if (this.hearts <= 0) { this.over = true; ev.push({ type: 'over' }); }
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
      const roll = R(), kind = this.segment++ < 2 ? 'rings' : roll < 0.22 + 0.25 * lvl ? 'pillar' : roll < 0.34 + 0.25 * lvl ? 'named' : 'rings';
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
          }
        }
        this.nextS += (n * 38 + 30 - 10 * lvl) * D2R;
      }
      this.queue.sort((a, b) => a.cs - b.cs);
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
          if (ahead > 40 * D2R && ahead < P.SPAWN) { Object.assign(o, { name: n.name, ket: n.ket, ph: n.ph, v: fromAngles(n.th, n.ph) }); this.objects.push(o); this.unblock(o); return; }
          o = { type: 'ring', th: o.fallback, rad: this.ringRadius };   // not ahead of us right now
        }
      }
      o.ph = wrap(this.camPh + P.SPAWN - late);
      o.v = fromAngles(o.th, o.ph);
      this.objects.push(o);
      this.unblock(o);
    }
    // Rings always win: an obstacle that would cover a ring is not placed (or is removed).
    blocks(ring, ob) {
      const m = ring.rad * 0.6 + 3 * D2R;
      return ob.type === 'cloud' ? angle(ring.v, ob.v) < ob.rad + m : meridianDist(ring.v, ob.ph, ob.th0, ob.th1) < ob.w + m;
    }
    unblock(o) {
      if (o.type === 'ring') this.objects = this.objects.filter(q => q.type === 'ring' || !this.blocks(o, q));
      else if (this.objects.some(q => q.type === 'ring' && !q.done && this.blocks(q, o))) this.objects.pop();
    }
    collide(ev) {
      for (const o of this.objects) {
        if (o.done) continue;
        if (o.type === 'ring') {
          const a = angle(this.r, o.v);
          if (a < o.rad) {
            o.done = true; this.rings++; this.combo = Math.min(10, this.combo + 1);
            const perfect = a < o.rad / 3;
            const pts = Math.round((o.bonus ? 250 : 100) * this.combo * (perfect ? 1.5 : 1));
            this.score += pts; if (perfect) this.perfects++;
            ev.push({ type: 'ring', o, pts, perfect, fidelity: fidelity(this.r, o.v), combo: this.combo });
            if (this.rings % 10 === 0 && this.hearts < P.HEARTS) { this.hearts++; ev.push({ type: 'heal' }); }   // every 10th ring
          }
        } else if (this.grace <= 0) {
          const d = o.type === 'cloud' ? angle(this.r, o.v) - o.rad : meridianDist(this.r, o.ph, o.th0, o.th1) - o.w;
          if (d < 0) { this.hearts--; this.hits++; this.combo = 0; this.grace = P.GRACE; ev.push({ type: 'hit', o }); }
        }
      }
    }
    cull(ev) {
      this.objects = this.objects.filter(o => {
        const rel = this.rel(o.ph);
        const gone = o.pole ? this.t - o.born > 8 : rel < P.DESPAWN && rel > -Math.PI + 0.2;
        if (gone && o.type === 'ring' && !o.done && !o.bonus) { this.combo = 0; ev.push({ type: 'miss', o }); }
        return !gone;
      });
    }
  }

  const api = { P, D2R, dot, cross, norm, unit, fromAngles, toAngles, wrap, rotate, liftRate, liftAxis, fidelity, angle,
    meridianDist, rng, NAMED, World };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RabiEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
