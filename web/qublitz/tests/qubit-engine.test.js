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

// --- The Ramsey experiment (closed form in the drive frame) against the full lab-frame solve:
// two square π/2 pulses around an idle time, at several detunings and miscalibrated pulses.
{
  const q = Q.qubitFromCode('K7RX4P'), T2 = 2 * q.T1, piHalf = 1 / (4 * q.omegaRabi);
  let worst = 0;
  for (const [dMHz, k, tau] of [[0, 1, 0], [1.5, 1, 150], [-3, 1, 37], [2, 0.7, 400], [0.5, 1.4, 150]]) {
    const pi2 = k * piHalf, wd = q.omegaQ - dMHz * 1e-3, tf = 2 * pi2 + tau + 1, n = Math.round(200 * tf);
    const env = new Float64Array(n);
    for (let i = 0; i < n; i++) { const tc = (i + 0.5) * tf / n; if (tc < pi2 || (tc >= pi2 + tau && tc < 2 * pi2 + tau)) env[i] = 1; }
    const full = Q.solve({ omegaQ: q.omegaQ, omegaRabi: q.omegaRabi, tFinal: tf, nSteps: n, omegaD: wd, envI: env, T1: q.T1, T2 });
    const iEnd = Math.min(n - 1, Math.round((2 * pi2 + tau) / tf * (n - 1)));
    const closed = Q.ramseyExperiment({ delays: [tau], driveFreqs: [wd], pi2Ns: pi2, omegaQ: q.omegaQ, omegaRabi: q.omegaRabi, T1: q.T1, T2 })[0][0];
    worst = Math.max(worst, Math.abs(full.p1[iEnd] - closed));
  }
  check('Ramsey experiment matches the full Bloch solve', worst < 0.005, `max |ΔP1| ${worst.toFixed(4)}`);
}

// --- Pulse-duration sweep (Rabi calibration) and the Rabi-formula fit.
{
  const q = { omegaQ: 5.0, omegaRabi: 0.3, T1: 150, T2: 300 };
  // P(|1⟩) after a square pulse of length t is P(|1⟩) at time t under a drive that stays on: check against an explicit
  // 20 ns pulse (500 bins of 0.04 ns) followed by idle time. The output grid (step 30/749 ns) does not fall on the pulse
  // end, but with the drive off P(|1⟩) decays as e^{−t/T₁} exactly, so the first idle sample is corrected back to 20 ns.
  const tf = 30, n = 750, env = new Float64Array(n);
  for (let i = 0; i < 500; i++) env[i] = 1;                                  // on for t < 20 ns
  const pulse = Q.solve({ omegaQ: q.omegaQ, omegaRabi: q.omegaRabi, tFinal: tf, nSteps: n, omegaD: q.omegaQ, envI: env, T1: q.T1, T2: q.T2 });
  const i500 = 500, atEnd = pulse.p1[i500] * Math.exp((pulse.t[i500] - 20) / q.T1);
  const sw = Q.rabiDurationSweep({ omegaQ: q.omegaQ, omegaRabi: q.omegaRabi, omegaD: q.omegaQ, amp: 1, tMax: 25, nDur: 126, T1: q.T1, T2: q.T2 });
  check('duration sweep: P1 after a 20 ns pulse equals P1(t = 20 ns) under a continuous drive',
    pulse.t[i500] > 20 && Math.abs(sw.t[100] - 20) < 1e-9 && Math.abs(atEnd - sw.p1[100]) < 1e-6,
    `${atEnd.toFixed(7)} vs ${sw.p1[100].toFixed(7)}`);

  // the Rabi formula without decay is (Ω²/Ω_R²)·sin²(Ω_R t/2)
  let worst = 0;
  for (const [f, d, t] of [[300, 0, 1.7], [300, 200, 2.3], [25, 10, 37], [120, 90, 4.4]]) {
    const wR = 2 * Math.PI * 1e-3 * Math.hypot(f, d), ref = (f * f / (f * f + d * d)) * Math.sin(wR * t / 2) ** 2;
    worst = Math.max(worst, Math.abs(Q.rabiFormula(t, f, d, Infinity) - ref));
  }
  check('Rabi formula: (Ω²/Ω_R²) sin²(Ω_R t/2) without decay', worst < 1e-12, `max err ${worst.toExponential(1)}`);

  // the fit recovers the drive strength and the detuning from engine-simulated sweeps, on and off resonance. The engine
  // keeps the counter-rotating terms, so |Δ| carries the Bloch–Siegert shift ≈ Ω_R²/(4ω_q) ≈ 4.5 MHz here.
  const rng = Q.rngFrom(11);
  for (const [dMHz, amp, tMax, nDur] of [[0, 1, 25, 126], [200, 1, 25, 126], [-120, 1, 25, 126], [0, 0.5, 40, 201]]) {
    const r = Q.rabiDurationSweep({ omegaQ: q.omegaQ, omegaRabi: q.omegaRabi, omegaD: q.omegaQ + dMHz * 1e-3, amp, tMax, nDur, T1: q.T1, T2: q.T2 });
    const want = 1e3 * q.omegaRabi * amp, fit = Q.fitRabiFormula(Array.from(r.t), Array.from(r.p1));
    check(`Rabi-formula fit (noiseless), Δ = ${dMHz} MHz, amp ${amp}: Ω/2π within 2%, |Δ| within 8 MHz`,
      fit && Math.abs(fit.fMHz - want) < 0.02 * want && Math.abs(fit.dMHz - Math.abs(dMHz)) < 8,
      fit && `Ω/2π ${fit.fMHz.toFixed(1)} (${want}) · |Δ| ${fit.dMHz.toFixed(1)} (${Math.abs(dMHz)}) · contrast ${fit.contrast.toFixed(3)}`);
    // near resonance the contrast is ≈ 1 and |Δ| = Ω_R√(1 − c) is ill-conditioned (±1% in c is ±13 MHz here), so the
    // on-resonance tolerance is wider; off resonance |Δ| is well determined.
    const noisy = Q.fitRabiFormula(Array.from(r.t), Array.from(Q.sampleShots(r.p1, 256, rng))), dTol = dMHz === 0 ? 20 : 12;
    check(`Rabi-formula fit (256 shots), Δ = ${dMHz} MHz, amp ${amp}: Ω/2π within 2%, |Δ| within ${dTol} MHz`,
      noisy && Math.abs(noisy.fMHz - want) < 0.02 * want && Math.abs(noisy.dMHz - Math.abs(dMHz)) < dTol,
      noisy && `Ω/2π ${noisy.fMHz.toFixed(1)} ± ${noisy.fMHzErr == null ? '?' : noisy.fMHzErr.toFixed(1)} · |Δ| ${noisy.dMHz.toFixed(1)} ± ${noisy.dMHzErr == null ? '?' : noisy.dMHzErr.toFixed(1)}`);
  }
  // on resonance the oscillation decays at Torrey's rate (1/T₁ + 1/T₂)/2, with and without pure dephasing
  for (const [T1, T2, tMax] of [[150, 300, 400], [150, 100, 400], [150, 60, 600]]) {
    const r = Q.rabiDurationSweep({ omegaQ: 5, omegaRabi: 0.025, omegaD: 5, amp: 1, tMax, nDur: tMax + 1, T1, T2 });
    const fit = Q.fitRabiFormula(Array.from(r.t), Array.from(r.p1)), want = Q.rabiDecayTime(T1, T2);
    check(`Rabi decay time at T₁ = ${T1}, T₂ = ${T2}: τ = 2/(1/T₁ + 1/T₂) = ${want.toFixed(1)} ns within 2%`,
      fit && Math.abs(fit.tauNs - want) < 0.02 * want && Math.abs(fit.fMHz - 25) < 0.5, fit && `τ ${fit.tauNs.toFixed(1)}, Ω/2π ${fit.fMHz.toFixed(2)}`);
  }
}

// --- Dephasing decomposition 1/T₂ = 1/(2T₁) + 1/Tφ, and the Ramsey envelope that shows it.
{
  check('Tφ = ∞ when T₂ = 2T₁', Q.dephasingTime(150, 300) === Infinity);
  check('Tφ = 150 ns for T₁ = 150, T₂ = 100 (1/100 − 1/300 = 1/150)', Math.abs(Q.dephasingTime(150, 100) - 150) < 1e-9, Q.dephasingTime(150, 100).toFixed(3));
  const T1 = 150, T2 = 100, Tphi = Q.dephasingTime(T1, T2);
  check('identity 1/T₂ = 1/(2T₁) + 1/Tφ', Math.abs(1 / T2 - (1 / (2 * T1) + 1 / Tphi)) < 1e-15);
  // a Ramsey experiment on that qubit: the fitted envelope is T₂ (not 2T₁), so the T₁-only envelope e^{−τ/2T₁} lies above it
  const delays = Q.linspace(0, 500, 251), pi2 = 1 / (4 * 0.3);
  const col = Q.ramseyExperiment({ delays, driveFreqs: [5.0 + 4e-3], pi2Ns: pi2, omegaQ: 5.0, omegaRabi: 0.3, T1, T2 })[0];
  const fit = Q.fitRamsey(Array.from(delays), Array.from(col), 4);
  check('Ramsey fit returns the set T₂ (= 100 ns) with pure dephasing present', fit && Math.abs(fit.T2Ns - T2) < 0.03 * T2, fit && `T₂ fit ${fit.T2Ns.toFixed(1)} ns, fringe ${fit.freqMHz.toFixed(3)} MHz`);
  check('... and the T₁-only envelope e^{−τ/2T₁} is wider than the fitted one', fit && Math.exp(-300 / (2 * T1)) > Math.exp(-300 / fit.T2Ns));
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
