// Checks web/qublitz/qubit-engine.js against the Python originals.
// Reference data: qubit-reference.json, written by scripts/make_qubit_reference.py.
// Usage: node web/qublitz/tests/qubit-engine.test.js
'use strict';
const fs = require('fs');
const path = require('path');
const Q = require('../qubit-engine.js');

const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'qubit-reference.json'), 'utf8'));
let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

// --- Lindblad solve vs qutip. The tolerance is against the tight-tolerance qutip run;
// the app-default qutip run is reported alongside for context.
const SOLVE_TOL = 2e-4;
for (const c of ref.cases) {
  const p = c.params;
  const r = Q.solve({ omegaQ: p.omega_q, omegaRabi: p.omega_rabi, tFinal: p.t_final, nSteps: p.n_steps,
    omegaD: p.omega_d, envI: p.env_I, envQ: p.env_Q, T1: p.T1, T2: p.T2 });
  const pick = a => c.idx.map(i => a[i]);
  let errTight = 0, errDefault = 0, qutipSelf = 0;
  for (const k of ['x', 'y', 'z', 'p1']) {
    errTight = Math.max(errTight, maxAbs(pick(r[k]), c.tight[k]));
    errDefault = Math.max(errDefault, maxAbs(pick(r[k]), c.default[k]));
    qutipSelf = Math.max(qutipSelf, maxAbs(c.default[k], c.tight[k]));
  }
  check(`solve: ${c.name}`, errTight < SOLVE_TOL,
    `max|JS−qutip(tight)|=${errTight.toExponential(1)}  vs app-default qutip ${errDefault.toExponential(1)}  (qutip default vs tight ${qutipSelf.toExponential(1)})`);
}

// --- Ramsey closed form: identical algebra, so near machine precision.
for (const [det, expected] of Object.entries(ref.ramsey.p1_by_detuning_MHz)) {
  const got = Q.ramseyP1(ref.ramsey.delays, +det, ref.ramsey.T1, ref.ramsey.T2);
  const e = maxAbs(Array.from(got), expected);
  check(`ramsey P1 at Δ=${det} MHz`, e < 1e-12, `max err ${e.toExponential(1)}`);
}

// --- Fits vs scipy curve_fit on identical noisy data.
const near = (a, b, rel, abs = 0) => Math.abs(a - b) <= Math.max(abs, rel * Math.abs(b));
for (const f of ref.fits) {
  const e = f.expect;
  if (f.kind === 'rabi') {
    const r = Q.fitRabi(f.t, f.y);
    check('fit: Rabi frequency', r && near(r.fMHz, e.f_MHz, 1e-4), r && `${r.fMHz.toFixed(5)} vs ${e.f_MHz.toFixed(5)} MHz`);
    check('fit: Rabi decay τ', r && near(r.tauNs, e.tau_ns, 1e-3), r && `${r.tauNs.toFixed(2)} vs ${e.tau_ns.toFixed(2)} ns`);
    check('fit: Rabi frequency error bar', r && near(r.fMHzErr, e.f_MHz_err, 0.02), r && `${r.fMHzErr.toExponential(3)} vs ${e.f_MHz_err.toExponential(3)}`);
  } else if (f.kind === 't1') {
    const r = Q.fitT1(f.t, f.y);
    check('fit: T1', r && near(r.T1Ns, e.T1_ns, 1e-4), r && `${r.T1Ns.toFixed(3)} vs ${e.T1_ns.toFixed(3)} ns`);
    check('fit: T1 error bar', r && near(r.T1NsErr, e.T1_ns_err, 0.02), r && `${r.T1NsErr.toFixed(3)} vs ${e.T1_ns_err.toFixed(3)} ns`);
  } else if (f.kind === 'ramsey') {
    const r = Q.fitRamsey(f.t, f.y, f.freq_guess_MHz);
    check('fit: Ramsey T2', r && near(r.T2Ns, e.T2_ns, 1e-3), r && `${r.T2Ns.toFixed(2)} vs ${e.T2_ns.toFixed(2)} ns`);
    check('fit: Ramsey fringe frequency', r && near(r.freqMHz, e.freq_MHz, 1e-4), r && `${r.freqMHz.toFixed(5)} vs ${e.freq_MHz.toFixed(5)} MHz`);
    check('fit: Ramsey T2 error bar', r && near(r.T2NsErr, e.T2_ns_err, 0.02), r && `${r.T2NsErr.toFixed(2)} vs ${e.T2_ns_err.toFixed(2)} ns`);
  }
}

// --- Shot sampling: unbiased, and spread matches the binomial.
{
  const rng = Q.rngFrom(7), shots = 256, p = 0.3, N = 20000;
  const s = Q.sampleShots(new Float64Array(N).fill(p), shots, rng);
  const m = s.reduce((a, b) => a + b, 0) / N;
  const v = s.reduce((a, b) => a + (b - m) ** 2, 0) / (N - 1);
  const vExp = p * (1 - p) / shots;
  check('shots: mean', Math.abs(m - p) < 4 * Math.sqrt(vExp / N), `${m.toFixed(4)} vs ${p}`);
  check('shots: variance', Math.abs(v / vExp - 1) < 0.05, `ratio ${(v / vExp).toFixed(3)}`);
  const edge = Q.sampleShots([0, 1], shots, rng);
  check('shots: P=0 and P=1 are exact', edge[0] === 0 && edge[1] === 1);
}

// --- Qubit codes: deterministic and inside the advertised ranges.
{
  const a = Q.qubitFromCode('ab3xy7'), b = Q.qubitFromCode(' AB3XY7 ');
  check('qubit code: case/space-insensitive and deterministic', JSON.stringify(a) === JSON.stringify(b));
  let inRange = true;
  const R = Q.QUBIT_RANGES, rng = Q.rngFrom(99);
  for (let i = 0; i < 2000; i++) {
    const q = Q.qubitFromCode(Q.randomCode(rng));
    inRange = inRange && q.omegaQ >= R.omegaQ[0] && q.omegaQ <= R.omegaQ[1]
      && q.omegaRabi * 1000 >= R.omegaRabiMHz[0] && q.omegaRabi * 1000 <= R.omegaRabiMHz[1]
      && q.T1 >= R.T1[0] && q.T1 <= R.T1[1];
  }
  check('qubit code: 2000 random codes stay in range', inRange);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
