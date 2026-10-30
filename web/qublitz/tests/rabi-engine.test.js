// Checks the physics and the course generator in web/qublitz/rabi-engine.js (the Rabi Rider game).
// Usage: node web/qublitz/tests/rabi-engine.test.js
'use strict';
const E = require('../rabi-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const { D2R, fromAngles, toAngles, rotate } = E;
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// ---- rotation sense: dr/dt = ω × r, with |0⟩ at z = +1 ----
{
  const dt = 1e-3, th = r => toAngles(r).th;
  const r0 = fromAngles(Math.PI / 2, -Math.PI / 2);                 // on the equator at −y
  check('+X drive lifts a state on the −y side toward |1⟩', th(rotate(r0, [1, 0, 0], dt)) > Math.PI / 2);
  const r1 = fromAngles(Math.PI / 2, Math.PI / 2);                  // at +y
  check('+X drive pushes a state on the +y side toward |0⟩', th(rotate(r1, [1, 0, 0], dt)) < Math.PI / 2);
  const r2 = fromAngles(Math.PI / 2, 0);                            // at +x
  check('+Y drive lifts a state on the +x side toward |1⟩', th(rotate(r2, [0, 1, 0], dt)) > Math.PI / 2);
  const rz = rotate(r2, [0, 0, 1], Math.PI / 2);
  check('Z rotation by π/2 takes +x to +y (precession is counterclockwise seen from |0⟩)', close(rz[1], 1, 1e-12));
  const pi = rotate([0, 0, 1], [2.4, 0, 0], Math.PI / 2.4);
  check('a π pulse about x takes |0⟩ to |1⟩', close(pi[2], -1, 1e-12));
}

// ---- liftRate is the true dθ/dt for any drive phase ----
{
  let worst = 0;
  const R = E.rng('lift');
  for (let i = 0; i < 200; i++) {
    const r = fromAngles(0.2 + 2.7 * R(), 2 * Math.PI * R()), al = 2 * Math.PI * R(), dt = 1e-6;
    const num = (toAngles(rotate(r, [Math.cos(al), Math.sin(al), 0], dt)).th - toAngles(r).th) / dt;
    worst = Math.max(worst, Math.abs(num - E.liftRate(r, al)));
  }
  check('liftRate(r, α) = −Ω sin(φ − α) matches the rotation', worst < 1e-5, `max error ${worst.toExponential(1)}`);
  const r = fromAngles(1, 0.7);
  check('the autopilot axis lifts at the full Rabi rate', close(E.liftRate(r, E.liftAxis(r)), 1, 1e-12));
}

// ---- autopilot: holding climbs at Ω, and φ advances at exactly Δ ----
{
  const w = new E.World({ seed: 't', decay: false }); w.spawn = () => {}; w.amp.a = -1;
  const a0 = w.angles, dt = 1 / 120; w.step({ hold: true }, dt);
  const a1 = w.angles;
  check('autopilot hold: dθ/dt = −Ω (climbs toward |0⟩)', close((a1.th - a0.th) / dt, -E.P.OMEGA, 0.02), ((a1.th - a0.th) / dt).toFixed(3));
  w.amp.a = 1; const b0 = w.angles; w.step({ hold: false }, dt);
  check('autopilot release: dθ/dt = +Ω (sinks toward |1⟩)', close((w.angles.th - b0.th) / dt, E.P.OMEGA, 0.02));
  check('autopilot hold: dφ/dt = Δ (the qubit keeps its place on screen)', close(E.wrap(a1.ph - a0.ph) / dt, w.delta, 0.01));
}

// ---- watched T₁: the no-jump drift is tan(θ/2) ∝ e^{−γt/2} ----
{
  const g = E.P.GAMMA, T = 3, dt = 1 / 120;
  const w = new E.World({ seed: 'nojump', decay: true }); w.spawn = () => {}; w.jumpRand = () => 1;   // never jump
  w.r = fromAngles(2.0, 0.3);
  for (let i = 0; i < T * 120; i++) w.step({ hold: false }, dt);
  // the autopilot pushes down when released, so compare with a world that has no drive at all
  const v = new E.World({ seed: 'nojump', mode: 'pilot', decay: true }); v.spawn = () => {}; v.jumpRand = () => 1;
  v.r = fromAngles(2.0, 0.3);
  for (let i = 0; i < T * 120; i++) v.step({ x: 0, y: 0, z: 0 }, dt);
  const want = 2 * Math.atan(Math.tan(1.0) * Math.exp(-g * T / 2));
  check('no-jump trajectory slides toward |0⟩ as tan(θ/2) e^{−γt/2}', close(v.angles.th, want, 1e-4), `${v.angles.th.toFixed(5)} vs ${want.toFixed(5)}`);
  check('the no-jump state stays pure (unit Bloch vector)', close(E.norm(v.r), 1, 1e-12));
}

// ---- watched T₁: averaged over trajectories, P₁ decays as e^{−γt} ----
{
  const g0 = E.P.GAMMA; E.P.GAMMA = 1;          // speed it up for the test
  const N = 3000, dt = 1 / 120, checks = [0.5, 1, 2], sums = checks.map(() => 0);
  for (let k = 0; k < N; k++) {
    const v = new E.World({ seed: 'ens' + k, mode: 'pilot', decay: true }); v.spawn = () => {}; v.collide = () => {}; v.cull = () => {};
    v.r = fromAngles(Math.PI / 2, 0);
    let t = 0, j = 0;
    while (j < checks.length) { v.step({}, dt); t += dt; if (t >= checks[j] - 1e-9) { sums[j] += (1 - v.r[2]) / 2; j++; } }
  }
  checks.forEach((t, j) => {
    const got = sums[j] / N, want = 0.5 * Math.exp(-t), sd = Math.sqrt(want * (1 - want) / N) + 0.004;
    check(`ensemble P₁(t = ${t}/γ) = ½ e^{−γt}`, Math.abs(got - want) < 3 * sd, `${got.toFixed(4)} vs ${want.toFixed(4)}`);
  });
  E.P.GAMMA = g0;
}

// ---- fidelity and distances ----
{
  check('fidelity |⟨1|0⟩|² = 0', close(E.fidelity([0, 0, 1], [0, 0, -1]), 0, 1e-15));
  check('fidelity |⟨+|0⟩|² = ½', close(E.fidelity([0, 0, 1], [1, 0, 0]), 0.5, 1e-15));
  check('|+i⟩ is at +y', E.NAMED.find(n => n.name === '|+i⟩') && close(fromAngles(Math.PI / 2, Math.PI / 2)[1], 1, 1e-15));
  check('a point on a pillar is inside it', close(E.meridianDist(fromAngles(0.5, 1), 1, 0, 1), 0, 1e-12));
  check('|0⟩ lies on every floor pillar (all meridians meet at the pole)', E.meridianDist([0, 0, 1], 2.2, 0, 1) < 1e-12);
  check('a point beyond the pillar\'s end is outside it', close(E.meridianDist(fromAngles(1.5, 1), 1, 0, 1), 0.5, 1e-9));
}

// ---- course: deterministic, varied, and fair ----
{
  const play = (seed, mode, secs) => {
    const w = new E.World({ seed, mode }); const log = [];
    for (let i = 0; i < secs * 120; i++) { for (const e of w.step({ hold: (i >> 6) & 1, x: Math.sin(i / 50), y: Math.cos(i / 70) }, 1 / 120)) log.push(e.type + i); }
    return { score: Math.round(w.score), log: log.join() };
  };
  const a = play('class-A', 'auto', 60), b = play('class-A', 'auto', 60), c = play('class-B', 'auto', 60);
  check('same seed and same inputs give the same run', a.score === b.score && a.log === b.log);
  check('a different seed gives a different course', a.log !== c.log);

  let overlaps = 0, rings = 0;
  for (const seed of ['s1', 's2', 's3', 's4', 's5']) for (const mode of ['auto', 'pilot']) {
    const w = new E.World({ seed, mode, decay: false }); w.hearts = 1e9; const seen = new Set();
    for (let i = 0; i < 200 * 120; i++) {
      w.step({ hold: i % 200 < 100 }, 1 / 120);
      for (const o of w.objects) {
        if (o.type !== 'ring' || seen.has(o)) continue;
        seen.add(o); rings++;
        for (const q of w.objects) {
          if (q.type === 'cloud' && E.angle(o.v, q.v) < q.rad + o.rad * 0.6) overlaps++;
          if (q.type === 'pillar' && E.meridianDist(o.v, q.ph, q.th0, q.th1) < q.w + o.rad * 0.6) overlaps++;
        }
      }
    }
  }
  check('no ring sits on a cloud or a pillar', overlaps === 0, `${overlaps} overlaps among ${rings} rings`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
