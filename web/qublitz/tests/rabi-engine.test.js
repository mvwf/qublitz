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
const DT = 1 / 120;
// a world with no course; decay off unless the test asks for a T1
const bare = opts => { const w = new E.World({ T1: Infinity, ...opts }); w.objects = []; w.generate = () => {}; return w; };
const ONE = { toOne: true }, ZERO = { toZero: true };

// ---- rotation sense: dr/dt = ω × r, with |0⟩ at z = +1 ----
{
  const th = r => toAngles(r).th;
  check('+X drive turns a state on the −y side toward |1⟩', th(rotate(fromAngles(Math.PI / 2, -Math.PI / 2), [1, 0, 0], 1e-3)) > Math.PI / 2);
  check('+X drive turns a state on the +y side toward |0⟩', th(rotate(fromAngles(Math.PI / 2, Math.PI / 2), [1, 0, 0], 1e-3)) < Math.PI / 2);
  check('a π pulse about x takes |0⟩ to |1⟩', close(rotate([0, 0, 1], [2.4, 0, 0], Math.PI / 2.4)[2], -1, 1e-12));
}

// ---- on resonance: nothing moves unless you drive ----
{
  const w = bare({ seed: 'still' }); const r0 = w.r.slice();
  for (let i = 0; i < 600; i++) w.step({}, DT);
  check('with no drive the qubit stays exactly where it is (no precession)', E.angle(w.r, r0) === 0);
}

// ---- ▼ and ▲ turn θ at the Rabi rate, from either side of the circle, and stay in the y–z plane ----
{
  for (const side of [-1, 1]) {
    const w = bare({ seed: 's' }); w.r = E.onCircle(1.2, side); w.amp = 1;
    const a = w.theta; w.step(ONE, DT);
    check(`toward |1⟩ raises θ at Ω (state on the ${side < 0 ? '−' : '+'}y side, drive ${w.driveSign > 0 ? '+X' : '−X'})`, close((w.theta - a) / DT, E.P.OMEGA, 0.02));
    w.amp = -1; const b = w.theta; w.step(ZERO, DT);
    check(`toward |0⟩ lowers θ at Ω (${side < 0 ? '−' : '+'}y side)`, close((w.theta - b) / DT, -E.P.OMEGA, 0.02));
    check('the state stays on the y–z circle (drive about x only)', Math.abs(w.r[0]) < 1e-12);
  }
  // a dive: |0⟩ to |1⟩ by holding ▼, settling without overshoot, in about a π pulse (plus the ramp and ease)
  const w = bare({ seed: 'dive' }); w.r = [0, 0, 1]; let t = 0, max = 0;
  while (t < 4) { w.step(ONE, DT); t += DT; max = Math.max(max, w.theta); if (w.theta > Math.PI - 0.5 * D2R && !w.tArr) w.tArr = t; }
  check('holding toward |1⟩ from |0⟩ reaches |1⟩ and stays there (no decay)', w.theta > Math.PI - 1 * D2R && max <= Math.PI + 1e-9, `θ = ${(w.theta / D2R).toFixed(2)}°`);
  check('the dive takes about a π pulse, π/Ω ≈ 1.3 s', w.tArr > Math.PI / E.P.OMEGA && w.tArr < Math.PI / E.P.OMEGA + 0.6, `${w.tArr.toFixed(2)} s`);
  for (let i = 0; i < 240; i++) w.step({}, DT);
  check('after letting go the drive is off', !w.driving && w.amp === 0);
}

// ---- watched T₁ ----
{
  const g = 1 / E.P.T1.normal, T = 3;
  const w = bare({ seed: 'nojump', T1: E.P.T1.normal }); w.jumpRand = () => 1; w.r = E.onCircle(2.0);
  for (let i = 0; i < T * 120; i++) w.step({}, DT);
  const want = 2 * Math.atan(Math.tan(1.0) * Math.exp(-g * T / 2));
  check('no-jump trajectory slides toward |0⟩ as tan(θ/2) e^{−γt/2}', close(w.theta, want, 1e-4), `${w.theta.toFixed(5)} vs ${want.toFixed(5)}`);
  check('the no-jump state stays pure (unit Bloch vector)', close(E.norm(w.r), 1, 1e-12));

  const N = 3000, ts = [0.5, 1, 2], sums = ts.map(() => 0);
  for (let k = 0; k < N; k++) {
    const v = bare({ seed: 'ens' + k, T1: 1 }); v.r = E.onCircle(Math.PI / 2);
    let t = 0, j = 0;
    while (j < ts.length) { v.step({}, DT); t += DT; if (t >= ts[j] - 1e-9) { sums[j] += v.p1; j++; } }
  }
  ts.forEach((t, j) => {
    const got = sums[j] / N, want = 0.5 * Math.exp(-t), sd = Math.sqrt(want * (1 - want) / N) + 0.004;
    check(`ensemble P₁(t = ${t}/γ) = ½ e^{−γt}`, Math.abs(got - want) < 3 * sd, `${got.toFixed(4)} vs ${want.toFixed(4)}`);
  });
}

// ---- decay is always on, and T₁ is the difficulty ----
{
  const w = new E.World({ seed: 'd' }), m = new E.World({ mission: 'flight', T1: E.P.T1.hard }), j = new E.World({ mission: 'jumps', T1: E.P.T1.easy });
  check('the endless run decays by default, with the normal T₁', w.decay && w.T1 === E.P.T1.normal, `T₁ = ${w.T1} s`);
  check('a mission takes the chosen T₁', m.decay && m.T1 === E.P.T1.hard);
  check('the Quantum jumps mission keeps its own short T₁', j.T1 === 6);
  const T1s = Object.values(E.P.T1);
  check('harder means a shorter T₁', T1s.every((v, i) => i === 0 || v < T1s[i - 1]), T1s.join(' > ') + ' s');
  // with the drive off, decay pulls the qubit down to |0⟩
  const v = bare({ seed: 'fall', T1: E.P.T1.expert }); v.r = E.onCircle(170 * D2R); let t = 0;
  while (t < 40 && v.theta > 1 * D2R) { v.step({}, DT); t += DT; }
  check('with the drive off, decay brings the qubit to |0⟩', v.theta <= 1 * D2R, `θ = ${(v.theta / D2R).toFixed(1)}° after ${t.toFixed(1)} s`);
}

// ---- measurement follows the Born rule; the X gate maps θ → π − θ ----
{
  const th = 60 * D2R, N = 20000; let ones = 0;
  for (let k = 0; k < N; k++) {
    const w = bare({ seed: 'born' + k }); w.r = E.onCircle(th); const ev = [];
    w.measure({ want: 1 }, ev); if (ev[0].one) ones++;
    if (k === 0) check('a measurement leaves the qubit at a pole', Math.abs(Math.abs(w.r[2]) - 1) < 1e-12);
  }
  const p = Math.sin(th / 2) ** 2;
  check('measured P(1) at θ = 60° is sin²(30°) = ¼', Math.abs(ones / N - p) < 3 * Math.sqrt(p * (1 - p) / N), `${(ones / N).toFixed(4)}`);
  const w = bare({ seed: 'x' }); w.r = E.onCircle(40 * D2R); const ev = []; w.applyX({}, ev);
  check('the X token takes θ = 40° to 140°', close(w.theta / D2R, 140, 1e-9));
}

// ---- a noise cloud measures the qubit: it collapses with the Born-rule odds, not to |0⟩ every time ----
{
  const th = 120 * D2R, N = 8000; let ones = 0, poles = true;
  for (let k = 0; k < N; k++) {
    const w = bare({ seed: 'noise' + k }); w.r = E.onCircle(th);
    w.objects = [{ type: 'cloud', t: DT / 2, th, rad: 10 * D2R, w: 0.5 }];
    const hit = w.step({}, DT).find(e => e.type === 'hit');
    if (!hit) { poles = false; break; }
    if (hit.one) ones++;
    if (Math.abs(Math.abs(w.r[2]) - 1) > 1e-12) poles = false;
  }
  const p = Math.sin(th / 2) ** 2;
  check('a noise hit leaves the qubit at a pole', poles);
  check('a noise hit at θ = 120° gives |1⟩ with P = sin²(60°) = ¾', Math.abs(ones / N - p) < 3 * Math.sqrt(p * (1 - p) / N), `${(ones / N).toFixed(4)}`);
}

// ---- a bot that steers to each target in turn (the right ring of a quiz) ----
function bot(w) {
  const next = w.objects.filter(o => !o.done && o.t > w.t - 1e-9 && (o.type === 'token' || (o.type === 'ring' && (!o.quiz || o.correct)))).sort((a, b) => a.t - b.t)[0];
  if (!next) return {};
  const d = next.th - w.theta;
  // a ring at a pole: hold the drive on toward it (the ease settles the state there)
  return { toOne: d > 1.5 * D2R || next.th > Math.PI - 1e-6, toZero: d < -1.5 * D2R || next.th < 1e-6 };
}

// ---- course: deterministic and fair ----
{
  // with decay, as played
  const run = (seed, secs) => { const w = new E.World({ seed }); w.hearts = 1e9; const log = [];
    for (let i = 0; i < secs * 120; i++) for (const e of w.step(bot(w), DT)) log.push(e.type + i);
    return { w, log: log.join() }; };
  const a = run('class-A', 60), b = run('class-A', 60), c = run('class-B', 60);
  check('same seed and same inputs give the same run', a.log === b.log && a.w.score === b.w.score);
  check('a different seed gives a different course', a.log !== c.log);

  let hits = 0, total = 0, covered = 0, struck = 0, kinds = new Set();
  for (const seed of ['s1', 's2', 's3', 's4', 's5', 's6']) {
    const w = new E.World({ seed, T1: Infinity }); w.hearts = 1e9; const seen = new Set();   // the geometry alone
    for (let i = 0; i < 200 * 120; i++) {
      for (const o of w.objects) {
        if (seen.has(o)) continue; seen.add(o); kinds.add(o.type);
        if (o.type === 'ring' || o.type === 'token') for (const c of w.objects) if (c.type === 'cloud' && w.covers(c, o)) covered++;
      }
      for (const e of w.step(bot(w), DT)) {
        if (e.type === 'ring' || (e.type === 'quiz' && e.correct)) { hits++; total++; }
        if (e.type === 'miss' || e.type === 'quizmiss' || (e.type === 'quiz' && !e.correct)) total++;
        if (e.type === 'hit') struck++;
      }
    }
  }
  check('no cloud covers a ring or token', covered === 0, `${covered}`);
  check('a bot that steers to each target reaches at least 98 % of them', hits / total >= 0.98, `${hits}/${total}`);
  check('the bot never needs to touch a cloud', struck === 0, `${struck} hits`);
  check('the endless course has rings, clouds, quizzes, measurements and X tokens', ['ring', 'cloud', 'gate', 'token'].every(k => kinds.has(k)), [...kinds].join(','));
  // with decay, jumps cost some rings; still playable at every difficulty
  for (const [name, T1] of Object.entries(E.P.T1)) {
    let got = 0, all = 0;
    for (const seed of ['s1', 's2', 's3']) {
      const w = new E.World({ seed, T1 }); w.hearts = 1e9;
      for (let i = 0; i < 120 * 120; i++) for (const e of w.step(bot(w), DT)) {
        if (e.type === 'ring' || (e.type === 'quiz' && e.correct)) { got++; all++; }
        if (e.type === 'miss' || e.type === 'quizmiss' || (e.type === 'quiz' && !e.correct)) all++;
      }
    }
    check(`with T₁ = ${T1} s (${name}) the bot still reaches ${name === 'expert' ? '60' : '75'} % of targets`, got / all >= (name === 'expert' ? 0.6 : 0.75), `${got}/${all}`);
  }
}

// ---- stakes: in the endless run a missed ring costs a heart and three in a row win one back; missions forgive misses ----
{
  const w = bare({ seed: 'stakes' }); w.r = E.onCircle(Math.PI / 2); const h0 = w.hearts;
  w.objects = [{ type: 'ring', t: 0.05, th: 0.3, rad: 0.2 }]; for (let i = 0; i < 12; i++) w.step({}, DT);
  check('endless: a missed ring costs a heart', w.hearts === h0 - 1, `${h0} → ${w.hearts}`);
  w.objects = [0.1, 0.2, 0.3].map(t => ({ type: 'ring', t: w.t + t, th: Math.PI / 2, rad: 0.2 })); for (let i = 0; i < 60; i++) w.step({}, DT);
  check('endless: three rings in a row restore it', w.hearts === h0, `${w.hearts}`);
  const m = new E.World({ mission: 'flight' }); m.objects = [{ type: 'ring', t: 0.05, th: 0.3, rad: 0.2 }]; const mh = m.hearts; for (let i = 0; i < 12; i++) m.step({}, DT);
  check('missions: a missed ring costs no heart (stars reward accuracy instead)', m.hearts === mh);
}

// ---- every mission can be cleared at the normal T₁ (decay makes every mission a little lucky) ----
{
  for (const m of E.MISSIONS) {
    const w = new E.World({ mission: m.id }); let clear = null, t = 0;
    while (!w.over && t < 120) { for (const e of w.step(bot(w), DT)) if (e.type === 'clear') clear = e; t += DT; }
    check(`mission "${m.title}" is cleared by the bot (T₁ = ${w.T1} s) with at least 2 stars`, clear && clear.stars >= 2,
      clear ? `${clear.stars}★, ${w.rings}/${w.total} rings, ${w.hits} hits` : 'not cleared');
  }
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
