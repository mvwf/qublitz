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

// ---- quiz gates: the right ring has the state the prompt describes ----
{
  const R = E.rng('quiz'); let bad = 0, n = 0, minSep = Infinity;
  for (let i = 0; i < 400; i++) {
    const k = i % E.QUIZ.length, q = E.QUIZ[k](R), p1 = Math.sin(q.th / 2) ** 2, num = +q.prompt.replace('−', '-').match(/= (-?[\d.]+)/)?.[1];
    const ok = k === 0 ? close(p1, num, 1e-9) : k === 1 ? close(1 - p1, num, 1e-9) : k === 2 ? close(Math.cos(q.th), num, 1e-9)
      : close(q.th, (({ 'π/3': 60, 'π/2': 90, '2π/3': 120, '3π/4': 135 })[q.prompt.match(/a (\S+) pulse/)[1]]) * D2R, 1e-9);
    if (!ok) bad++; n++;
    minSep = Math.min(minSep, Math.abs(q.th - q.wrong));
    if (Math.min(q.th, q.wrong) < 22 * D2R || Math.max(q.th, q.wrong) > 158 * D2R) bad++;
  }
  check('every quiz answer matches its prompt (Born rule, ⟨σz⟩ = cos θ, pulse area)', bad === 0, `${bad} bad of ${n}`);
  check('the two rings of a quiz gate are at least 30° apart', minSep >= 30 * D2R - 1e-9, `${(minSep / D2R).toFixed(1)}°`);
}

// ---- power-ups: spin echo blocks a hit, recalibration lowers Δ, a Purcell filter stops jumps ----
{
  const w = new E.World({ seed: 'pw', decay: false }); w.spawn = () => {};
  w.shield = true; w.objects = [{ type: 'cloud', th: w.angles.th, ph: w.angles.ph, v: w.r.slice(), rad: 0.2 }];
  const ev = w.step({ hold: false }, 1 / 120).map(e => e.type);
  check('a spin echo takes the hit instead of a heart', ev.includes('block') && w.hearts === E.P.HEARTS && !w.shield);
  const v = new E.World({ seed: 'pw2', decay: false }); v.spawn = () => {};
  const d0 = v.delta; v.slowT = E.P.SLOW; for (let i = 0; i < 240; i++) v.step({ hold: false }, 1 / 120);
  check('recalibration lowers Δ toward SLOW_FACTOR of its value', v.delta < d0 * (E.P.SLOW_FACTOR + 0.05), `${(v.delta / d0).toFixed(3)}`);
  const u = new E.World({ seed: 'pw3', decay: true }); u.spawn = () => {}; u.jumpRand = () => 0;   // would jump at once
  u.purcellT = 5; u.r = fromAngles(2.5, 0); u.step({ hold: false }, 1 / 120);
  check('no quantum jumps under a Purcell filter', u.jumps === 0);
}

// ---- hits relax to |0⟩; measurement gates follow the Born rule ----
{
  const w = new E.World({ seed: 'hit', decay: false }); w.spawn = () => {};
  w.r = fromAngles(2.0, 0.4); w.objects = [{ type: 'cloud', th: 2.0, ph: 0.4, v: w.r.slice(), rad: 0.2 }];
  const ev = w.step({ hold: false }, 1 / 120).map(e => e.type);
  check('a hit costs a heart and relaxes the qubit to |0⟩', ev.includes('hit') && w.hearts === E.P.HEARTS - 1 && w.angles.th < 0.05);
  for (const th of [Math.PI / 3, Math.PI / 2, 2.5]) {
    let ones = 0; const N = 4000;
    for (let k = 0; k < N; k++) {
      const v = new E.World({ seed: 'meas' + k, decay: false }); v.spawn = () => {};
      v.r = fromAngles(th, 0.3); const ev = []; v.measure({ want: 1 }, ev);
      if (ev[0].one) ones++;
      if (Math.abs(Math.abs(v.r[2]) - 1) > 1e-6) ones = NaN;   // must land on a pole
    }
    const want = Math.sin(th / 2) ** 2, sd = Math.sqrt(want * (1 - want) / N);
    check(`measurement at θ = ${(th / D2R).toFixed(0)}° gives |1⟩ with P = sin²(θ/2)`, Math.abs(ones / N - want) < 4 * sd, `${(ones / N).toFixed(3)} vs ${want.toFixed(3)}`);
  }
}

// ---- gate tokens: X reflects through the equator, √X turns a quarter turn; φ is kept ----
{
  const w = new E.World({ seed: 'gate', decay: false }); w.spawn = () => {};
  w.r = fromAngles(0.5, 1.1); const ev = []; w.collect({ kind: 'X' }, ev);
  check('the X token sends θ to π − θ and keeps φ', close(w.angles.th, Math.PI - 0.5, 1e-9) && close(w.angles.ph, 1.1, 1e-9));
  w.r = fromAngles(2.6, -0.4); w.collect({ kind: 'sx' }, ev);
  check('the √X token turns the state 90° toward the far pole', close(w.angles.th, 2.6 - Math.PI / 2, 1e-9) && close(w.angles.ph, -0.4, 1e-9));
  check('X is a rotation by π: the state moved through angle π − 2θ in the meridian', close(E.gateTheta('X', 0.3), Math.PI - 0.3, 1e-12));
}

// ---- missions: each can be cleared with every ring, by a bot that chases the next target ----
{
  const drive = w => {
    const pole = w.objects.find(o => o.pole && !o.done), me = w.rel(w.angles.ph);
    const c = w.objects.filter(o => (o.type === 'ring' || o.type === 'power') && !o.done && !o.pole && !(o.quiz && !o.correct) && w.rel(o.ph) > me - 0.05)
      .sort((a, b) => w.rel(a.ph) - w.rel(b.ph));
    const err = (pole ? Math.PI : c[0] ? toAngles(c[0].v).th : Math.PI / 2) - w.angles.th;
    if (w.mode === 'auto') return { hold: err < 0.02 + 0.12 * w.amp.a };
    const ph = w.angles.ph, k = Math.sign(err) * Math.min(1, Math.abs(err) * 4);
    return { x: -Math.sin(ph) * k, y: Math.cos(ph) * k };
  };
  const short = [];
  for (const m of E.MISSIONS) {
    const w = new E.World({ mission: m.id }); w.hearts = 1e9; let cleared = false;
    for (let i = 0; i < 120 * 120 && !w.over; i++) for (const e of w.step(drive(w), 1 / 120)) if (e.type === 'clear') cleared = true;
    if (!cleared || w.rings < Math.ceil((m.three ?? 1) * w.total)) short.push(`${m.id} ${w.rings}/${w.total}`);
  }
  check('every mission can be finished with its 3-star share of the rings', short.length === 0, short.join(', '));
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
