// Checks the physics and the course in web/qublitz/copter-engine.js (the Qubit Copter game).
// Usage: node web/qublitz/tests/copter-engine.test.js
'use strict';
const C = require('../copter-engine.js');
const R = require('../rabi-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// ---- no precession: with no drive the state stays put ----
{
  const w = new C.World({ seed: 'still' }); w.r = R.fromAngles(1.1, 0.4); const r0 = w.r.slice();
  for (let i = 0; i < 600; i++) w.step({ x: 0, y: 0 }, 1 / 120);
  check('without a drive the Bloch vector does not move (on resonance)', R.angle(w.r, r0) < 1e-12);
}

// ---- a full drive about x for π/Ω is a π pulse ----
{
  const w = new C.World({ seed: 'pi' }); w.amp.x = 1;
  const T = Math.PI / C.P.OMEGA, n = Math.round(T * 1200);
  for (let i = 0; i < n; i++) { w.amp.x = 1; w.r = R.unit(R.rotate(w.r, w.omega(), T / n)); }
  check('a π pulse about x takes |0⟩ to |1⟩', close(w.r[2], -1, 1e-9));
}

// ---- steering: the chosen equatorial axis joins r and p in one turn ----
{
  const rnd = R.rng('steer'); let worst = 0, slowest = 0;
  for (let i = 0; i < 300; i++) {
    const r = R.fromAngles(Math.acos(2 * rnd() - 1), 2 * Math.PI * rnd()), p = R.fromAngles(Math.acos(2 * rnd() - 1), 2 * Math.PI * rnd());
    const s = C.steerToward(r, p), n = Math.hypot(s.x, s.y);
    if (n < 1e-9) continue;
    const axis = [s.x / n, s.y / n, 0];
    worst = Math.max(worst, Math.abs(R.dot(r, axis) - R.dot(p, axis)));   // r and p on one circle about the axis
    // and following it gets there
    let q = r; for (let k = 0; k < 2000; k++) { const d = C.steerToward(q, p); q = R.unit(R.rotate(q, [C.P.OMEGA * d.x, C.P.OMEGA * d.y, 0], 0.005)); }
    slowest = Math.max(slowest, R.angle(q, p));
  }
  check('the steering axis keeps r·n = p·n (one rotation joins them)', worst < 1e-9, worst.toExponential(1));
  check('following the steering reaches the point', slowest < 1 * Math.PI / 180, `${(slowest * 180 / Math.PI).toFixed(2)}° left`);
}

// ---- circuit targets are the states of the gate sequence ----
{
  const w = new C.World({ seed: 'circ', mode: 'circuit' }); let r = [0, 0, 1], bad = 0;
  for (let k = 0; k < 12; k++) { w.t += 5; w.generate(); }   // schedule a minute of the course
  for (const o of w.targets) { r = C.applyGate(r, o.gate); if (R.angle(r, o.v) > 1e-6) bad++; }
  check('each circuit target is the previous one with its gate applied, from |0⟩', bad === 0 && w.targets.length > 10, `${w.targets.length} targets`);
  check('H|0⟩ = |+⟩ and √X|0⟩ = |−i⟩', C.stateName(C.applyGate([0, 0, 1], 'H')) === '|+⟩' && C.stateName(C.applyGate([0, 0, 1], 'SX')) === '|−i⟩');
}

// ---- the course is fair: walls never cover a target, and a steering bot reaches (almost) every target in time ----
{
  let covered = 0, hits = 0, total = 0;
  for (const mode of ['targets', 'circuit']) for (const seed of ['a', 'b', 'c']) {
    const w = new C.World({ seed, mode }); w.hearts = 1e9;
    for (let i = 0; i < 180 * 120; i++) {
      const tg = w.targets[0];
      for (const wl of w.walls) for (const o of w.targets) if (o.t >= wl.t0 && o.t <= wl.t1 && o.v[wl.axis] > wl.lo && o.v[wl.axis] < wl.hi) covered++;
      for (const e of w.step(tg ? C.steerToward(w.r, tg.v) : {}, 1 / 120)) { if (e.type === 'hit') { hits++; total++; } if (e.type === 'miss') total++; }
    }
  }
  check('no wall covers a target', covered === 0, `${covered}`);
  check('a bot that points at each target reaches at least 98 % of them', hits / total >= 0.98, `${hits}/${total}`);
  const a = new C.World({ seed: 'same' }), b = new C.World({ seed: 'same' });
  check('the same seed gives the same course', JSON.stringify(a.targets.map(o => o.v)) === JSON.stringify(b.targets.map(o => o.v)));
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
