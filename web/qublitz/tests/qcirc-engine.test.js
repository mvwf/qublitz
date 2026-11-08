// Checks the circuits of ErrorSweeper's Circuits chapter (web/qublitz/qcirc-engine.js): every circuit does what the
// page says it does. Usage: node web/qublitz/tests/qcirc-engine.test.js
'use strict';
const Q = require('../qcirc-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const C = id => Q.CIRCUITS.find(c => c.id === id);
const a = 0.8, b = 0.6;
const amp = (s, ket) => { const i = [...ket].reduce((v, ch, q) => v | (+ch << q), 0); return [s.re[i], s.im[i]]; };
const close = (x, y, t) => Math.abs(x - y) < (t || 1e-12);

// ---- gates ----
{
  const s = new Q.State(1); Q.gate(s, 'H', 0); Q.gate(s, 'X', 0); Q.gate(s, 'H', 0);   // H X H = Z: |0⟩ → |0⟩
  check('H X H |0⟩ = |0⟩ (H X H = Z)', close(s.re[0], 1) && close(s.re[1], 0));
  const t = new Q.State(1); Q.gate(t, 'X', 0); Q.gate(t, 'H', 0); Q.gate(t, 'X', 0); Q.gate(t, 'H', 0);   // H X H |1⟩ = Z|1⟩ = −|1⟩
  check('H X H |1⟩ = −|1⟩', close(t.re[1], -1));
  const r = new Q.State(1), m = Q.rx(1.1); Q.apply1(r, 0, m[0], m[1], m[2], m[3]);
  check('RX(θ) flips with probability sin²(θ/2)', close(Q.prob1(r, 0), Math.sin(0.55) ** 2));
}

// ---- Fig. 16: the encoder makes α|000⟩ + β|111⟩ ----
{
  const out = Q.run(C('encode'), { alpha: a, beta: b }).snaps.pop();
  const T = Q.terms(out);
  check('encoder: α|000⟩ + β|111⟩ and nothing else', T.length === 2 && close(amp(out, '000')[0], a) && close(amp(out, '111')[0], b));
  const mid = Q.run(C('encode'), { alpha: a, beta: b }).snaps[1];
  check('after one CNOT: α|00⟩ + β|11⟩ on q₁q₂', close(amp(mid, '000')[0], a) && close(amp(mid, '110')[0], b));
}

// ---- the measured syndrome and the measurement-free version correct every single bit flip ----
for (const id of ['measure', 'auto']) {
  const c = C(id), res = {};
  for (const e of ['none', 'X0', 'X1', 'X2']) {
    const r = Q.run(c, { error: e, rng: Q.rng(3) }), v = Q.verdict(c, r.snaps.pop(), a, b);
    res[e] = v.kind;
  }
  check(`${id}: no error and each single X error end with the logical state restored`, Object.values(res).every(k => k === 'ok'), JSON.stringify(res));
  const two = Q.verdict(c, Q.run(c, { error: 'X0X1', rng: Q.rng(4) }).snaps.pop(), a, b);
  check(`${id}: two bit flips become a logical X (α and β swap)`, two.kind === 'logicalX');
}
// the syndrome of each error is Girvin's Table 3
{
  const c = C('measure'), want = { none: '00', X0: '10', X1: '11', X2: '01' };
  let ok = true;
  for (const e in want) { const r = Q.run(c, { error: e, rng: Q.rng(1) }), s = r.snaps.pop(); if (`${s.bits.s1}${s.bits.s2}` !== want[e]) ok = false; }
  check('syndromes match Table 3: X₁ → 10, X₂ → 11, X₃ → 01', ok);
}
// measuring the parities does not disturb α and β: before the error, the encoded state is untouched by the checks
{
  const c = C('measure'), r = Q.run(c, { error: 'none', rng: Q.rng(9) }), s = r.snaps[c.ops.findIndex(o => o.g === 'M') + 2];
  check('parity measurements leave α|000⟩ + β|111⟩ intact', close(Q.dataFidelity(s, c.data, Q.idealData(c, a, b)), 1, 1e-12));
}
// a continuous error is digitized: P(syndrome 11) = sin²(θ/2), and both outcomes end in the corrected state
for (const id of ['measure', 'auto']) {
  const c = C(id), th = 0.9, iM = c.ops.findIndex(o => o.g === 'CX' && o.c.length === 1 && o.c[0] === 2) + 1;   // after the 4th CNOT
  const before = Q.run(c, { error: 'RX1', theta: th, rng: Q.rng(1) }).snaps[iM];
  const p11 = (() => { let p = 0; for (let i = 0; i < before.N; i++) if (((i >> 3) & 1) && ((i >> 4) & 1)) p += before.re[i] ** 2 + before.im[i] ** 2; return p; })();
  check(`${id}: a rotation by θ on q₂ gives syndrome 11 with probability sin²(θ/2)`, close(p11, Math.sin(th / 2) ** 2, 1e-12), `${p11.toFixed(6)} vs ${(Math.sin(th / 2) ** 2).toFixed(6)}`);
  let allOk = true;
  for (let sd = 1; sd <= 40; sd++) { const v = Q.verdict(c, Q.run(c, { error: 'RX1', theta: th, rng: Q.rng(sd) }).snaps.pop(), a, b); if (v.kind !== 'ok') allOk = false; }
  check(`${id}: whichever way the measurement goes, the rotation is fully undone`, allOk);
}
// Fig. 17 leaves the ancillas in |00⟩ and the data unentangled from them
{
  const c = C('auto'), s = Q.run(c, { error: 'X2', rng: Q.rng(2) }).snaps.pop();
  check('Fig. 17: after the reset both ancillas are |0⟩', Q.prob1(s, 3) < 1e-12 && Q.prob1(s, 4) < 1e-12);
}

// ---- the phase-flip code ----
{
  const c = C('phase'), res = {};
  for (const e of ['none', 'Z0', 'Z1', 'Z2']) res[e] = Q.verdict(c, Q.run(c, { error: e, rng: Q.rng(5) }).snaps.pop(), a, b).kind;
  check('phase code: each single Z error is corrected', Object.values(res).every(k => k === 'ok'), JSON.stringify(res));
  const s0 = Q.initial(c, a, b), plus = 1 / Math.sqrt(8);
  check('phase code state is α|+++⟩ + β|−−−⟩', close(amp(s0, '000')[0], (a + b) * plus) && close(amp(s0, '111')[0], (a - b) * plus) && close(amp(s0, '100')[0], (a - b) * plus));
  let allOk = true; for (let sd = 1; sd <= 30; sd++) if (Q.verdict(c, Q.run(c, { error: 'RZ1', theta: 1.2, rng: Q.rng(sd) }).snaps.pop(), a, b).kind !== 'ok') allOk = false;
  check('phase code: a small phase rotation is digitized and undone', allOk);
}

// ---- one surface-code face ([[4,2,2]] checks) ----
{
  const c = C('check'), want = { none: '00', X0: '10', Z2: '01', Y1: '11' };
  let ok = true; const got = {};
  for (const e in want) for (let sd = 1; sd <= 5; sd++) { const s = Q.run(c, { error: e, rng: Q.rng(sd) }).snaps.pop(); got[e] = `${s.bits.s1}${s.bits.s2}`; if (got[e] !== want[e]) ok = false; }
  check('a bit flip lights the Z-check, a phase flip the X-check, Y both, deterministically', ok, JSON.stringify(got));
}

// ---- Shor's code ----
{
  const s = Q.run(C('shor'), { alpha: a, beta: b }).snaps.pop(), k = 1 / (2 * Math.SQRT2);
  // α(|000⟩+|111⟩)^⊗3/2√2 + β(|000⟩−|111⟩)^⊗3/2√2: coefficient of block pattern (x,y,z) ∈ {000,111}³
  let worst = 0;
  for (const x of [0, 1]) for (const y of [0, 1]) for (const z of [0, 1]) {
    const ket = [x, x, x, y, y, y, z, z, z].join(''), sg = (x ? -1 : 1) * (y ? -1 : 1) * (z ? -1 : 1);
    worst = Math.max(worst, Math.abs(amp(s, ket)[0] - k * (a + sg * b)));
  }
  check('Shor encoder gives α(|000⟩ + |111⟩)³/2√2 + β(|000⟩ − |111⟩)³/2√2', worst < 1e-12 && Q.terms(s).length === 8, `max error ${worst.toExponential(1)}`);
}

// ---- classical codes (Girvin §2) ----
{
  check('3-bit repetition code fails with 3ε² − 2ε³ (Girvin eq. 12)', [0.01, 0.1, 0.3].every(e => close(Q.repFail(3, e), 3 * e * e - 2 * e ** 3, 1e-15)));
  check('every repetition code breaks even at ε = 1/2 (eq. 13)', [3, 5, 9, 21].every(N => close(Q.repFail(N, 0.5), 0.5, 1e-12)));
  check('below 1/2, longer codes fail less', Q.repFail(9, 0.2) < Q.repFail(5, 0.2) && Q.repFail(5, 0.2) < Q.repFail(3, 0.2) && Q.repFail(9, 0.6) > Q.repFail(3, 0.6));
  check('a parity bit misses ≈ m(m+1)ε²/2 of messages (eq. 4)', Math.abs(Q.parityMiss(7, 1e-4) / (7 * 8 / 2 * 1e-8) - 1) < 1e-3);
  let ok = true; for (let k = 1; k <= 7; k++) { const w = new Array(7).fill(0); w[k - 1] = 1; const s = Q.hammingSyndrome(w); if (s[0] + 2 * s[1] + 4 * s[2] !== k) ok = false; }
  check('Hamming [7,4,3]: a flip on bit k gives the syndrome k in binary', ok);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
