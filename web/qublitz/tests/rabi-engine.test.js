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
const bare = opts => { const w = new E.World({ T1: Infinity, ghosts: 1, ...opts }); w.objects = []; w.generate = () => {}; return w; };
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
  const w = bare({ seed: 'nojump', T1: E.P.T1.normal }); w.ghosts[0].jump = () => 1; w.r = E.onCircle(2.0);
  for (let i = 0; i < T * 120; i++) w.step({}, DT);
  check('between jumps nothing moves: no drift, decoherence is jumps only', close(w.theta, 2.0, 1e-12), `${w.theta.toFixed(5)}`);

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

// ---- jumps: photons leave at the rate P₁/T₁; phase flips dephase; the Z control turns the phase only ----
{
  const N = 3000, T1 = 4, w = new E.World({ seed: 'photons', T1, ghosts: N }); w.objects = []; w.generate = () => {}; w.r = E.onCircle(120 * D2R);
  let photons = 0; for (let i = 0; i < 12; i++) for (const e of w.step({}, DT)) if (e.type === 'jump') photons += e.n;
  const want = N * 0.75 * (1 - Math.exp(-12 * DT / T1)), sd = Math.sqrt(want);
  check('photons leave at the rate P₁/T₁ (3000 runs at θ = 120° for 0.1 s)', Math.abs(photons - want) < 3 * sd, `${photons} vs ${want.toFixed(0)}`);
  const Tphi = 3, d = new E.World({ seed: 'dephase', T1: 6, Tphi, ghosts: 4000 }); d.objects = []; d.generate = () => {}; d.r = E.onCircle(Math.PI / 2);
  for (let i = 0; i < 120; i++) d.step({}, DT);
  const perp = Math.hypot(d.r[0], d.r[1]), wantPerp = Math.exp(-1 / 6 - 1 / Tphi);
  check('coherence decays as e^{−t/T₁ − t/T_φ} (1/T₂ = 1/T₁ + 1/T_φ)', Math.abs(perp - wantPerp) < 0.03, `${perp.toFixed(3)} vs ${wantPerp.toFixed(3)}`);
  const f = bare({ seed: 'flip', Tphi: 0.01 }); f.r = E.onCircle(60 * D2R); f.ghosts[0].phase = () => 0; f.step({}, DT);
  check('a phase flip keeps the height and moves the run to the other side of the circle', close(f.theta, 60 * D2R, 1e-9) && f.r[1] > 0.8);
  const z = bare({ seed: 'z' }); z.r = E.onCircle(60 * D2R); for (let i = 0; i < 120; i++) z.step({ zPlus: true }, DT);
  check('the Z control turns the phase, not the height', close(z.theta, 60 * D2R, 1e-9) && Math.abs(Math.atan2(z.r[1], z.r[0]) + Math.PI / 2) > 0.5);
}

// ---- free play: no course, no hearts lost, never over; missions can fix the number of runs ----
{
  const f = new E.World({ free: true, seed: 'free', T1: 1, ghosts: 6 });
  for (let i = 0; i < 60 * 120; i++) f.step(i % 400 < 200 ? ONE : {}, DT);
  check('free play has no course and never ends', f.objects.length === 0 && !f.over && f.hearts === E.P.ENDLESS_HEARTS, `${f.t.toFixed(0)} s, ${f.jumps} jumps`);
  check('Flight school and the photon mission fly one run; Many runs flies the ensemble',
    new E.World({ mission: 'flight', ghosts: 12 }).N === 1 && new E.World({ mission: 'photon', ghosts: 12 }).N === 1 && new E.World({ mission: 'ensemble', ghosts: 12 }).N === 12);
}

// ---- decay is always on, and T₁ is the difficulty ----
{
  const w = new E.World({ seed: 'd' }), m = new E.World({ mission: 'ensemble', T1: E.P.T1.hard }), j = new E.World({ mission: 'jumps', T1: E.P.T1.easy });
  check('the endless run decays by default, with the normal T₁', w.decay && w.T1 === E.P.T1.normal, `T₁ = ${w.T1} s`);
  check('a mission takes the chosen T₁', m.decay && m.T1 === E.P.T1.hard);
  check('the Quantum jumps mission keeps its own short T₁', j.T1 === 5);
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
    const w = new E.World({ seed, T1: Infinity, ghosts: 1 }); w.hearts = 1e9; const seen = new Set();   // the geometry alone, one run
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
    const want = { easy: 0.85, normal: 0.65, hard: 0.6, expert: 0.5 }[name];
    check(`with T₁ = ${T1} s (${name}) and ${E.P.GHOSTS} ghosts, aiming the mean passes at least ${100 * want} % of targets`, got / all >= want, `${got}/${all}`);
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

// ---- every mission can be cleared with the ensemble at the normal T₁ (decay makes it partly luck), and a single
//      run without decay clears the luck-free ones with three stars ----
{
  for (const m of E.MISSIONS) {
    const w = new E.World({ mission: m.id }); let clear = null, t = 0;
    while (!w.over && t < 120) { for (const e of w.step(bot(w), DT)) if (e.type === 'clear') clear = e; t += DT; }
    check(`mission "${m.title}" is cleared by the bot (${w.N} ghosts, T₁ = ${w.T1} s)`, !!clear,
      clear ? `${clear.stars}★, ${w.rings}/${w.total} rings, ${w.hits} hits` : 'not cleared');
    if (m.T1 || m.items.some(it => it[0] === 'm')) continue;
    const v = new E.World({ mission: m.id, ghosts: 1, T1: Infinity }); let c1 = null; t = 0;
    while (!v.over && t < 120) { for (const e of v.step(bot(v), DT)) if (e.type === 'clear') c1 = e; t += DT; }
    check(`mission "${m.title}" with one run and no decay: 3 stars`, c1 && c1.stars === 3, c1 ? `${c1.stars}★` : 'not cleared');
  }
}

// ---- the ensemble: every run gets the same pulse, jumps on its own, and on average decays as e^{−t/T₁} ----
{
  const w = new E.World({ seed: 'ghosts', T1: 2, ghosts: 2000 }); w.objects = []; w.generate = () => {}; w.r = [0, 0, -1];
  const got = [];
  for (let i = 0; i < 2 * 120; i++) { w.step({}, DT); if ((i + 1) % 120 === 0) got.push(w.p1); }
  check('2000 ghosts from |1⟩: the fraction left in |1⟩ is e^{−t/T₁} at t = 1 and 2 s', got.every((p, k) => Math.abs(p - Math.exp(-(k + 1) / 2)) < 0.03),
    got.map(p => p.toFixed(3)).join(', ') + ' vs ' + [1, 2].map(k => Math.exp(-k / 2).toFixed(3)).join(', '));
  const v = new E.World({ seed: 'same', T1: Infinity, ghosts: 5 }); v.objects = []; v.generate = () => {}; v.r = E.onCircle(30 * D2R);
  for (let i = 0; i < 60; i++) v.step(ONE, DT);
  const th = v.thetas();
  check('without decay every run follows the same pulse exactly', th.every(x => Math.abs(x - th[0]) < 1e-12));
  const s = new E.World({ seed: 'split', T1: Infinity, ghosts: 400, noReset: true }); s.objects = []; s.generate = () => {}; s.r = E.onCircle(Math.PI / 2);
  s.measure({ want: 1 }, []);
  const ones = s.ghosts.filter(g => g.r[2] < 0).length;
  check('a measurement splits the runs by the Born rule (≈ ½ at the equator)', Math.abs(ones / 400 - 0.5) < 0.08, `${ones}/400`);
  // after the split, one +X π/2 pulse on every run takes both groups to the equator together (θ and π − θ meet at π/2)
  for (const g of s.ghosts) g.r = E.rotate(g.r, [E.P.OMEGA, 0, 0], Math.PI / 2 / E.P.OMEGA);
  const ths = s.thetas();
  check('after a measurement one pulse brings both groups to the equator together', ths.every(x => Math.abs(x - Math.PI / 2) < 0.03),
    `${(Math.min(...ths) / D2R).toFixed(1)}°–${(Math.max(...ths) / D2R).toFixed(1)}°`);
  // the game's gates measure and reset: the outcomes follow the Born rule, then every run is back in |0⟩
  const z = new E.World({ seed: 'reset', T1: Infinity, ghosts: 400 }); z.objects = []; z.generate = () => {}; z.r = E.onCircle(120 * D2R);
  const ev = []; z.measure({ want: 1 }, ev);
  check('measure-and-reset: Born-rule outcomes, then every run in |0⟩', Math.abs(ev[0].ones / 400 - 0.75) < 0.07 && z.ghosts.every(g => g.r[2] === 1), `${ev[0].ones}/400 read 1`);
  // the score is the fidelity, the average of the runs' own fidelities, so aiming the mean is best
  const f = new E.World({ seed: 'fid', T1: 2, ghosts: 50 }); f.objects = []; f.generate = () => {}; f.r = E.onCircle(100 * D2R);
  for (let i = 0; i < 240; i++) f.step(i < 60 ? ONE : {}, DT);
  const avg = f.ghosts.reduce((a, g) => a + E.fidelity(g.r, E.onCircle(110 * D2R, f.r[1] <= 0 ? -1 : 1)), 0) / 50;
  check('the ensemble fidelity F̄ is the mean of the runs\' fidelities', close(f.fidelity(110 * D2R), avg, 1e-12), `${f.fidelity(110 * D2R).toFixed(4)}`);
  const best = f.theta; let top = -1, at = 0;
  for (let d = 0; d <= 180; d += 0.5) { const F = f.fidelity(d * D2R); if (F > top) { top = F; at = d; } }
  check('F̄ is largest with the target at the mean\'s direction', Math.abs(at - best / D2R) < 0.6 && close(top, (1 + f.purity) / 2, 1e-3),
    `best at ${at}°, mean at ${(best / D2R).toFixed(1)}°, F̄max = ${top.toFixed(3)} = (1 + |r̄|)/2`);
  const mixed = new E.World({ seed: 'mix', T1: Infinity, ghosts: 2 }); mixed.objects = []; mixed.generate = () => {};
  mixed.ghosts[0].r = E.onCircle(150 * D2R); mixed.ghosts[1].r = [0, 0, 1];
  check('a spread-out ensemble cannot pass a ring, however well aimed', !mixed.passes({ th: mixed.theta, rad: 20 * D2R }), `|r̄| = ${mixed.purity.toFixed(2)}`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
