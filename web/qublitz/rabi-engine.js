/* rabi-engine.js: physics and course for Rabi Rider (rabi-rider.html), plus the Bloch-vector helpers that
 * copter-engine.js shares. No dependencies. Loads as a classic <script> (window.RabiEngine) or from Node
 * (require), so tests/rabi-engine.test.js checks the same code the game runs.
 *
 * The qubit is a pure state, a unit Bloch vector r = (sinθ cosφ, sinθ sinφ, cosθ), with |0⟩ at θ = 0 (z = +1)
 * and |1⟩ at θ = π. It is driven exactly on resonance, so in the frame rotating with the drive
 *
 *   H = (ħΩ/2) a σx,   dr/dt = ω × r,   ω = (Ω a, 0, 0),
 *
 * and there is no precession: the state moves only while you drive it. A drive about x keeps r in the y–z
 * plane, so the state is fixed by θ alone and the game draws it as a height, |0⟩ at the top and |1⟩ at the
 * bottom, while the course scrolls past in time. You ask for up or down; the game picks the drive's sign
 * (+X or −X) that turns the state that way, which is a choice of drive phase (0 or π), and eases the drive
 * off as the state reaches a pole. Holding a drive on is a Rabi oscillation: P(1) = sin²(Ωt/2).
 * The drive amplitude follows its target with time constant TAU, like a bandwidth-limited pulse.
 *
 * T₁ decay (when on) is a watched qubit (a quantum trajectory): while no photon is detected the state stays
 * pure and slides toward |0⟩ at dθ/dt = −(γ/2) sinθ, and a photon is detected with probability γ·P₁·dt per
 * step, which resets the state to |0⟩ (a quantum jump). Averaged over runs this is ordinary T₁ decay.
 *
 * Hitting noise costs a heart and relaxes the qubit to |0⟩. A measurement gate is a projective measurement
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
    GAMMA: 1 / 14,       // T₁ decay rate γ = 1/T₁ [1/s], when decay is on
    HEARTS: 3,           // in a mission, where only noise costs a heart
    ENDLESS_HEARTS: 5,   // in the endless run, where a missed target costs one too
    STREAK_HEAL: 3,      // and every 3 targets in a row restores one
    GRACE: 1.2,          // s of invulnerability after a jump
    HIT_GRACE: 1.6,      // and after a hit, which also resets the qubit to |0⟩
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
        why: `P(0) = cos²(θ/2), so P(0) = ${p} sits at θ = ${fmtDeg(th)}. The closer to |0⟩ at the top, the likelier a 0.` }; },
    R => { const z = pick(R, [0.8, 0.5, -0.5, -0.8]), th = Math.acos(z);
      return { prompt: `⟨σz⟩ = ${fmtNum(z)}`, th, wrong: Math.PI - th,
        why: `⟨σz⟩ = cos θ = P(0) − P(1): +1 at the top, −1 at the bottom. So ⟨σz⟩ = ${fmtNum(z)} is at θ = ${fmtDeg(th)}.` }; },
    R => { const [deg, name] = pick(R, [[60, 'π/3'], [90, 'π/2'], [120, '2π/3'], [135, '3π/4']]), th = deg * D2R;
      return { prompt: `|0⟩ after a ${name} pulse`, th, wrong: th / 2,
        why: `A pulse of area ${name} turns the Bloch vector by ${deg}°, so θ = ${deg}°. Half of that, ${deg / 2}°, is the angle in the ket cos(θ/2)|0⟩ − i sin(θ/2)|1⟩.` }; },
  ];

  /* Missions: short scripted courses, each teaching one idea. Items are placed by time t [s] and polar angle θ [°]:
   *   ['r', t, θ]            ring (a target state)        ['c', t, θ, half-height°, half-width s]   noise cloud
   *   ['q', t, k]            quiz gate QUIZ[k]             ['m', t, want]   measurement gate paying for outcome want (0 or 1)
   *   ['x', t, θ]            X gate token
   * Stars: 3 for every ring with no hits (or a `three` share of the rings, where luck is involved), 2 for 70 %, 1 for finishing. */
  const MISSIONS = [
    { id: 'flight', title: 'Flight school', start: 90,
      brief: 'Hold <b>▼</b> to drive the qubit toward |1⟩ and <b>▲</b> to drive it toward |0⟩. Let go and it stays put: the drive is off. Fly through every ring.',
      learn: 'Each ring was a target state, and you reached it by rotating the Bloch vector with a resonant drive. With the drive off nothing changes: in the frame rotating with a resonant drive there is no precession (in the lab the qubit still turns at GHz, but so does the frame).',
      items: [['r', 3, 90], ['r', 5, 60], ['r', 7, 40], ['r', 9, 70], ['r', 11, 110], ['r', 13, 140], ['r', 15, 120], ['r', 17, 80], ['r', 19, 45], ['r', 21, 90]] },
    { id: 'pi', title: 'The π pulse', start: 0,
      brief: 'A π pulse turns |0⟩ all the way to |1⟩. Hold <b>▼</b> to dive to the ring at the bottom, then <b>▲</b> back to the top.',
      learn: 'At this Rabi rate (Ω = 1.6 rad/s) a π pulse takes π/Ω ≈ 2 s, which is how long each dive took. A constant drive would carry the state through |1⟩ and back to |0⟩, P(1) = sin²(Ωt/2): one Rabi oscillation. The game stops you at |1⟩ and reverses the drive when you press ▲, so your down-and-up traced the same P(1) curve. On a real transmon a π pulse typically takes 10–50 ns.',
      items: [['r', 3, 0], ['r', 6, 180], ['r', 9.5, 0], ['r', 12.5, 180], ['r', 16, 0], ['r', 19, 180], ['r', 22.5, 0]] },
    { id: 'equator', title: 'Superposition', start: 0,
      brief: 'The middle line is the equator: P(1) = ½, an equal superposition. That takes a π/2 pulse, half a dive. Stay on it between the noise clouds.',
      learn: 'On the equator a measurement gives 0 or 1 with even odds, yet the state is not a coin that has already landed: one more π/2 pulse takes it to |1⟩ with certainty.',
      items: [['r', 3.5, 90], ['c', 5, 40, 18, 0.5], ['c', 5, 140, 18, 0.5], ['r', 6.5, 90], ['c', 8, 42, 17, 0.6], ['c', 8, 138, 17, 0.6],
        ['r', 9.5, 90], ['r', 12, 0], ['r', 14.5, 90], ['c', 16, 40, 18, 0.5], ['c', 16, 140, 18, 0.5], ['r', 17.5, 90]] },
    { id: 'born', title: 'Measure up', start: 90,
      brief: 'Quiz gates: two purple rings, one answer. Read the question and fly through the ring it describes. The labels on the right give P(1) at each height.',
      learn: 'The height here is the angle θ, not the sphere\'s z = cos θ. P(1) = (1 − cos θ)/2 = sin²(θ/2) is a straight line in z but not in θ, so P(1) = ¼ sits at θ = 60°, not 45°.',
      items: [['r', 3, 90], ['q', 6, 0], ['r', 8.5, 90], ['q', 11.5, 2], ['r', 14, 90], ['q', 17, 3], ['r', 19.5, 90], ['q', 22.5, 1], ['r', 25, 90]] },
    { id: 'collapse', title: 'Collapse', start: 90,
      brief: 'A measurement gate (the blue curtain) puts your qubit in |0⟩ or |1⟩ at random, with P(1) = sin²(θ/2). Each gate names the outcome it pays for: fly close to that end first.',
      learn: 'Measurement is not a rotation. It picks |0⟩ or |1⟩ with the Born-rule odds and leaves the qubit there. Near |0⟩ or |1⟩ the result is almost certain; on the equator it is a coin flip.',
      items: [['r', 3, 60], ['m', 5.5, 0], ['r', 8, 90], ['m', 10.5, 1], ['r', 13, 100], ['m', 15.5, 1], ['r', 18, 80], ['m', 20.5, 0], ['r', 23, 90]] },
    { id: 'gates', title: 'The X gate', start: 30,
      brief: 'X tokens apply the X gate at once: a π pulse, done for you. It takes θ to 180° − θ, mirroring you through the equator. Some rings come too fast to fly to: grab the X first.',
      learn: 'A gate is a fast, calibrated pulse. X is a π rotation about x, so it swaps |0⟩ and |1⟩ and maps every height θ to 180° − θ.',
      items: [['r', 3, 30], ['x', 4, 30], ['r', 4.6, 150], ['r', 6.5, 150], ['x', 7.5, 150], ['r', 8.1, 30], ['r', 10, 60], ['x', 11, 60], ['r', 11.6, 120],
        ['r', 13.5, 45], ['x', 14.5, 45], ['r', 15.1, 135], ['r', 17, 90]] },
    { id: 'jumps', title: 'Quantum jumps', start: 90, decay: true, gamma: 1 / 6, three: 0.8,
      brief: 'T₁ decay is on, and strong. Now the qubit drifts up toward |0⟩ by itself, and the closer you fly to |1⟩, the more often it jumps straight back to |0⟩. Jumps cost no hearts.',
      learn: 'The jump rate is γ·P(1). Between jumps the state still drifts toward |0⟩, because not seeing a photon is information too. With the drive off, averaging many flights gives plain exponential decay, P(1) ∝ e^(−t/T₁).',
      items: [['r', 3, 120], ['r', 5, 145], ['r', 7, 150], ['r', 9, 140], ['r', 11, 150], ['r', 13, 135], ['r', 15, 150], ['r', 17, 145], ['r', 19, 150], ['r', 21, 140]] },
    { id: 'exam', title: 'Final exam', start: 90,
      brief: 'Everything at once: rings, noise, quiz gates, a measurement and X tokens.',
      learn: 'You drove a qubit with real resonant pulses, read probabilities off the Bloch sphere, used a gate, and watched a measurement collapse the state. That is the toolkit behind every single-qubit experiment.',
      items: [['r', 3, 90], ['c', 4.5, 40, 20, 0.5], ['r', 6, 130], ['q', 9, 0], ['r', 11.5, 90], ['m', 14, 1], ['r', 16.5, 90], ['c', 18, 140, 22, 0.5],
        ['r', 19.5, 60], ['x', 20.5, 60], ['r', 21.1, 120], ['q', 24, 2], ['r', 26.5, 90]] },
  ];
  const isGood = o => o.type === 'ring' || o.type === 'token';   // things to collect, which clouds never cover

  /* ---------------- the world ----------------
   * opts: { seed, decay: bool, mission: id }
   * Missions start with HEARTS and only noise costs one; the endless run starts with ENDLESS_HEARTS, a missed target
   * (or a wrong quiz answer) costs one, and every STREAK_HEAL targets in a row restores one.
   * step(input, dt): input = { up, down } (booleans). Each step returns a list of events: ring, miss, quiz, quizmiss,
   * token, measure, close, heal, hit, jump, clear (a mission finished) and over.
   * Objects carry a time t; the course scrolls in time, and an object is met when the clock reaches it. */
  class World {
    constructor(opts = {}) {
      this.decay = !!opts.decay;
      this.rand = rng('course:' + (opts.seed ?? 'random'));
      this.jumpRand = rng('jumps:' + (opts.seed ?? 'random'));
      this.measRand = rng('measure:' + (opts.seed ?? 'random'));
      this.r = onCircle(90 * D2R);
      this.t = 0; this.amp = 0;                         // amp: −1 drives toward |0⟩, +1 toward |1⟩
      this.objects = []; this.nextT = 2.5; this.walk = 90 * D2R;
      this.score = 0; this.combo = 0; this.rings = 0; this.misses = 0; this.perfects = 0; this.jumps = 0; this.hits = 0; this.closes = 0;
      this.quizRight = 0; this.quizWrong = 0; this.measures = 0; this.measWins = 0; this.gates = 0;
      this.hearts = P.ENDLESS_HEARTS; this.streak = 0; this.grace = 0; this.over = false;
      this.mission = null; this.cleared = false; this.total = 0;
      const m = opts.mission && MISSIONS.find(x => x.id === opts.mission);
      if (m) {
        this.mission = m; this.decay = !!m.decay; this.hearts = P.HEARTS; this.r = onCircle(Math.max(1e-4, Math.min(Math.PI - 1e-4, m.start * D2R)));
        this.jumpRand = rng('jumps:mission:' + m.id); this.measRand = rng('measure:mission:' + m.id);
        this.script(m);
      } else this.generate();
    }
    get level() { return Math.min(1, this.t / P.LEVEL_TIME); }
    get ringRadius() { return P.RING0 + (P.RING_MIN - P.RING0) * this.level; }
    get theta() { return thetaOf(this.r); }
    get p1() { return (1 - this.r[2]) / 2; }
    get driving() { return Math.abs(this.omega()[0]) > 0.02 * P.OMEGA; }
    // The drive about x for the current amplitude: its sign follows the side of the circle the state is on, so +amp
    // always pushes θ up (toward |1⟩), and it eases off within EASE of the pole it is pushing toward.
    omega() {
      const th = this.theta, side = this.r[1] <= 0 ? 1 : -1;
      const ease = Math.min(1, (this.amp > 0 ? Math.PI - th : th) / P.EASE);
      return [P.OMEGA * this.amp * side * ease, 0, 0];
    }
    // +1 for a +X drive, −1 for −X, 0 when off: the pulse the game is playing for you.
    get driveSign() { const w = this.omega()[0]; return Math.abs(w) < 0.02 * P.OMEGA ? 0 : Math.sign(w); }

    step(input, dt) {
      const ev = [];
      if (this.over) return ev;
      const want = (input.down ? 1 : 0) - (input.up ? 1 : 0);
      this.amp += (want - this.amp) * (1 - Math.exp(-dt / P.TAU));
      if (Math.abs(this.amp) < 1e-3 && !want) this.amp = 0;
      this.r = rotate(this.r, this.omega(), dt);
      // watched T₁: a jump to |0⟩ with probability γ P₁ dt, otherwise the no-jump drift toward |0⟩
      if (this.decay) {
        const g = this.mission?.gamma ?? P.GAMMA, z = this.r[2];
        if (this.jumpRand() < g * (1 - z) / 2 * dt) {
          this.r = [0, 0, 1]; this.jumps++; this.combo = 0; this.grace = Math.max(this.grace, P.GRACE);
          ev.push({ type: 'jump' });
        } else {
          const h = (g / 2) * dt;
          this.r = [this.r[0] * (1 - h * z), this.r[1] * (1 - h * z), this.r[2] + h * (1 - z * z)];
        }
      }
      this.r = unit(this.r);
      const t0 = this.t; this.t += dt;
      this.score += 2 * dt;
      this.grace = Math.max(0, this.grace - dt);
      if (!this.mission) this.generate();
      this.meet(t0, ev);
      this.objects = this.objects.filter(o => o.t + (o.w || 0) > this.t - 3);
      if (this.hearts <= 0) { this.over = true; ev.push({ type: 'over' }); }
      else if (this.mission && this.t >= this.endT) { this.over = true; this.cleared = true; ev.push({ type: 'clear', stars: this.stars() }); }
      return ev;
    }

    // Everything the clock reached in (t0, t].
    meet(t0, ev) {
      const th = this.theta;
      for (const o of this.objects) {
        if (o.done) continue;
        if (o.type === 'cloud') {
          // an ellipse in (time, θ); touching it costs a heart and relaxes the qubit to |0⟩
          const u = (this.t - o.t) / o.w, d = Math.abs(th - o.th) - o.rad * Math.sqrt(Math.max(0, 1 - u * u));
          if (Math.abs(u) <= 1) {
            o.near = Math.min(o.near ?? Infinity, d);
            if (d < 0 && this.grace <= 0) {
              o.struck = true; this.hearts--; this.hits++; this.combo = 0; this.grace = P.HIT_GRACE;
              this.r = [0, 0, 1]; this.amp = 0;
              ev.push({ type: 'hit', o });
            }
          } else if (u > 1) {
            o.done = true;
            if (!o.struck && o.near >= 0 && o.near < P.CLOSE) { const pts = 25 * Math.max(1, this.combo); this.score += pts; this.closes++; ev.push({ type: 'close', o, pts }); }
          }
          continue;
        }
        if (o.t > this.t || o.t <= t0 - 1e-9) continue;
        o.done = true;
        if (o.type === 'gate') this.measure(o, ev);
        else if (o.type === 'token') { if (Math.abs(th - o.th) < P.TOKEN) this.applyX(o, ev); }
        else if (o.quiz) this.answer(o, th, ev);
        else {
          const off = Math.abs(th - o.th);
          if (off < o.rad) {
            this.combo = Math.min(10, this.combo + 1);
            const perfect = off < o.rad / 3; if (perfect) this.perfects++;
            const pts = Math.round(100 * this.combo * (perfect ? 1.5 : 1)); this.score += pts;
            ev.push({ type: 'ring', o, pts, perfect, combo: this.combo });
            this.counted(ev);
          } else { this.combo = 0; this.misses++; o.missed = true; ev.push({ type: 'miss', o, off, heart: this.lose() }); }
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
    // Both rings of a quiz arrive together: the one you are in answers it.
    answer(o, th, ev) {
      const q = o.quiz; if (q.answered) return;
      const mine = this.objects.filter(x => x.quiz === q).find(x => Math.abs(th - x.th) < x.rad);
      if (!mine) { if (o.correct) { q.answered = true; this.combo = 0; this.misses++; ev.push({ type: 'quizmiss', q, heart: this.lose() }); } return; }
      q.answered = true; this.objects.forEach(x => { if (x.quiz === q) x.done = true; });
      let pts = 0;
      if (mine.correct) { this.quizRight++; this.combo = Math.min(10, this.combo + 1); pts = 300 * this.combo; this.score += pts; this.counted(ev); }
      else { this.quizWrong++; this.combo = 0; }
      ev.push({ type: 'quiz', o: mine, q, correct: mine.correct, pts, combo: this.combo, heart: !mine.correct && this.lose() });
    }
    // A projective measurement of σz: |1⟩ with probability P₁ = sin²(θ/2), otherwise |0⟩.
    measure(o, ev) {
      const p1 = this.p1, one = this.measRand() < p1;
      this.r = one ? [0, 0, -1] : [0, 0, 1]; this.amp = 0;   // the readout interrupts the drive
      this.measures++;
      const win = (one ? 1 : 0) === o.want, pts = win ? 400 * Math.max(1, this.combo) : 0;
      if (win) { this.measWins++; this.score += pts; }
      ev.push({ type: 'measure', o, one, p1, win, pts });
    }
    // The X gate: a π turn about x, so θ → π − θ.
    applyX(o, ev) {
      const from = this.theta; this.r = unit(rotate(this.r, [Math.PI, 0, 0], 1)); this.gates++; this.score += 50;
      ev.push({ type: 'token', o, from, to: this.theta });
    }
    stars() {
      const f = this.total ? this.rings / this.total : 1;
      return f >= (this.mission.three ?? 1) - 1e-9 && this.hits === 0 ? 3 : f >= 0.7 ? 2 : 1;
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
        if (this.nextT > 8 && roll < 0.1 + 0.05 * lvl) {
          // a measurement gate, then a ring on the equator
          const t = this.nextT + 0.4; this.add({ type: 'gate', t, want: R() < 0.5 ? 1 : 0 });
          this.walk = Math.PI / 2; this.nextT = t + this.reach(0, Math.PI / 2, lvl) + 0.2;
          this.add({ type: 'ring', t: this.nextT, th: this.walk, rad: rr });
        } else if (this.nextT > 8 && roll < 0.22 + 0.06 * lvl) {
          // a quiz gate: reachable whichever ring is right
          const q = QUIZ[Math.floor(R() * QUIZ.length)](R), t = this.nextT + this.reach(this.walk, q.th, lvl) + 0.8;
          this.quiz(t, q, rr); this.walk = q.th; this.nextT = t;
        } else if (this.nextT > 8 && roll < 0.3 + 0.05 * lvl) {
          // an X token on the way, and a ring where it sends you, sooner than you could fly there
          const t = this.nextT + this.reach(0, 0, lvl) + 0.3, th = clamp(this.walk), to = Math.PI - th;
          this.add({ type: 'token', t, th, kind: 'X' }); this.walk = to; this.nextT = t + 0.6;
          this.add({ type: 'ring', t: this.nextT, th: to, rad: rr });
        } else {
          // a run of rings on a random walk, with clouds beside the path
          const n = 2 + Math.floor(R() * 3);
          for (let i = 0; i < n; i++) {
            const from = this.walk, to = clamp(from + (R() * 2 - 1) * (35 + 50 * lvl) * D2R), t = this.nextT + this.reach(from, to, lvl);
            this.add({ type: 'ring', t, th: to, rad: rr });
            if (R() < 0.35 + 0.4 * lvl && t - this.nextT > 1) this.cloudBeside(this.nextT, t, from, to, R);
            this.walk = to; this.nextT = t;
          }
        }
      }
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
    QUIZ, MISSIONS, World };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RabiEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
