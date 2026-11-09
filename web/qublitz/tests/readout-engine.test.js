// Checks web/qublitz/readout-engine.js: pointer states and photon numbers against their closed forms, the exact segments
// against a numerical integration of dα/dt, the steady-state SNR and dephasing formulas, the κ = 2χ optimum, the error
// function, the exact fidelity against simulated shots, and the transmon χ formula.
// Usage: node web/qublitz/tests/readout-engine.test.js
'use strict';
const R = require('../readout-engine.js');
const { C } = R;
let failures = 0;
function check(label, ok, detail) { console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`); if (!ok) failures++; }
const close = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const TAU = 2 * Math.PI;
const P = (o) => Object.assign({}, R.DEFAULTS, o);

// ---- pointer states
{
  const p = P({ chi: 0.7, kappa: 1.3, nbar: 3 });
  check('midpoint drive: n̄ photons in |0⟩', close(R.photons(p, 0), 3, 1e-12));
  check('midpoint drive: n̄ photons in |1⟩', close(R.photons(p, 1), 3, 1e-12));
  const k = TAU * 1.3, x = TAU * 0.7, b = C.abs(C.sub(R.pointer(p, 0), R.pointer(p, 1))) ** 2;
  check('|α₀ − α₁|² = 4χ²n̄/(κ²/4 + χ²)', close(b, 4 * x * x * 3 / (k * k / 4 + x * x), 1e-12));
  // driving on the |1⟩ resonance fills it more than |0⟩
  const q = P({ chi: 0.7, kappa: 1.3, nbar: 3, delta: 0.7 });
  check('drive at f_r + χ: |1⟩ holds 4n̄(κ²/4+χ²)/κ² photons', close(R.photons(q, 1), 3 * (k * k / 4 + x * x) / (k * k / 4), 1e-12));
  check('photonsAt peaks at f_r ∓ χ', R.photonsAt(p, 0, -0.7) > R.photonsAt(p, 0, -0.6) && R.photonsAt(p, 0, -0.7) > R.photonsAt(p, 0, -0.8));
}

// ---- exact segments against RK4 on dα/dt = −(κ/2 + iΔ)α − iε
{
  const p = P({ chi: 0.4, kappa: 2.1, nbar: 5, delta: 0.15 }), r = R.rates(p);
  for (const s of [0, 1]) {
    let a = [0.3, -0.2], I = [0, 0]; const T = 0.9, N = 20000, h = T / N;
    const g = z => C.add(C.scale(C.mul(r.lam[s], z), -1), [0, -r.eps]);
    for (let i = 0; i < N; i++) {
      const k1 = g(a), k2 = g(C.add(a, C.scale(k1, h / 2))), k3 = g(C.add(a, C.scale(k2, h / 2))), k4 = g(C.add(a, C.scale(k3, h)));
      const an = C.add(a, C.scale(C.add(C.add(k1, C.scale(k2, 2)), C.add(C.scale(k3, 2), k4)), h / 6));
      I = C.add(I, C.scale(C.add(a, an), h / 2)); a = an;
    }
    const ex = R.segment(r, [0.3, -0.2], s, T);
    check(`segment matches RK4, state ${s}`, C.abs(C.sub(ex.a, a)) < 1e-9, C.abs(C.sub(ex.a, a)).toExponential(1));
    check(`segment integral matches, state ${s}`, C.abs(C.sub(ex.I, I)) < 1e-6, C.abs(C.sub(ex.I, I)).toExponential(1));
  }
  const tr = R.trajectory(P({ tau: 50 }), 1, 10);
  check('trajectory starts empty and rings up to the pointer state', C.abs(tr[0]) === 0 && C.abs(C.sub(tr[10], R.pointer(P(), 1))) < 1e-9);
}

// ---- SNR, dephasing, κ = 2χ
{
  const p = P({ tau: 200, T1: Infinity });     // τ ≫ 2/κ: ring-up is negligible
  const a = R.analyze(p);
  check('long τ: SNR → 2√(ηκτ)|α₁ − α₀|', close(a.snr, a.snrSteady, 3e-3), `${a.snr.toFixed(3)} vs ${a.snrSteady.toFixed(3)}`);
  const q = P({ chi: 0.6, kappa: 0.9, nbar: 4 }), b = R.analyze(q);
  const k = TAU * 0.9, x = TAU * 0.6;
  check('Γ_d = κ|α₀ − α₁|²/2 for a midpoint drive', close(b.gammaD, k * C.abs(C.sub(b.a0, b.a1)) ** 2 / 2, 1e-12));
  check('Γ_d = 2κχ²n̄/(κ²/4 + χ²)', close(b.gammaD, 2 * k * x * x * 4 / (k * k / 4 + x * x), 1e-12));
  let best = 0, bestK = 0;
  for (let kk = 0.2; kk <= 3; kk += 0.01) { const s = R.analyze(P({ chi: 0.5, kappa: kk, tau: 1e3, T1: Infinity })).snrSteady; if (s > best) { best = s; bestK = kk; } }
  check('steady-state SNR at fixed n̄ peaks at κ = 2χ', Math.abs(bestK - 1) < 0.011, `κ/2π = ${bestK.toFixed(2)} MHz`);
  check('ring-up time 2/κ', close(b.tRing, 2 / k, 1e-12));
}

// ---- error function
check('erfc(0) = 1', close(R.erfc(0), 1, 1e-7));
check('erfc(1)', close(R.erfc(1), 0.157299207050285, 2e-7));
check('erfc(3), relative', Math.abs(R.erfc(3) / 2.209049699858544e-5 - 1) < 2e-7);
check('erfc(−1) = 2 − erfc(1)', close(R.erfc(-1), 2 - 0.157299207050285, 2e-7));
check('Φ(1.96) ≈ 0.975', close(R.Phi(1.96), 0.9750021, 1e-6));

// ---- fidelity
{
  const p = P({ T1: Infinity, pexc: 0 }), a = R.analyze(p);
  check('no T₁, no heating: F = 1 − ½erfc(SNR/2√2)', close(a.F, 1 - 0.5 * R.erfc(a.snr / (2 * Math.SQRT2)), 1e-9));
  const big = R.analyze(P({ nbar: 40, tau: 2, T1: Infinity, pexc: 3 }));
  check('huge SNR: F → 1 − P_exc', close(big.F, 0.97, 1e-6), big.F.toFixed(6));
  const dec = R.analyze(P({ nbar: 40, tau: 2, T1: 20, pexc: 0 }));
  check('huge SNR with T₁: 0 < P(0|1) < P(decay)', dec.p01 > 0 && dec.p01 < dec.pDecay, `${dec.p01.toFixed(4)} < ${dec.pDecay.toFixed(4)}`);
  check('... and about half the decays are misread', dec.p01 > 0.3 * dec.pDecay && dec.p01 < 0.7 * dec.pDecay);
  // simulated shots agree with the exact numbers
  const q = P({ nbar: 1.2, tau: 0.5, T1: 3, pexc: 4 }), e = R.analyze(q), n = 40000, sh = R.shots(q, n, 7);
  const se = v => 3.5 * Math.sqrt(v * (1 - v) / n);
  check('shots: P(1|0) matches', Math.abs(sh.wrong[0] / n - e.p10) < se(e.p10), `${(sh.wrong[0] / n).toFixed(4)} vs ${e.p10.toFixed(4)}`);
  check('shots: P(0|1) matches', Math.abs(sh.wrong[1] / n - e.p01) < se(e.p01), `${(sh.wrong[1] / n).toFixed(4)} vs ${e.p01.toFixed(4)}`);
  check('shots are reproducible from a seed', JSON.stringify(R.shots(q, 50, 3).pts) === JSON.stringify(R.shots(q, 50, 3).pts));
  // a finite T₁ makes an interior best τ
  const bt = R.bestTau(P({ T1: 5, nbar: 1 }), 0.02, 20);
  check('finite T₁: the best integration time is inside the range', bt.tau > 0.05 && bt.tau < 15, `τ = ${bt.tau.toFixed(2)} µs, F = ${bt.F.toFixed(4)}`);
  check('fidelityCurve agrees with analyze', close(R.fidelityCurve(q, [0.5])[0], e.F, 1e-12));
}

// ---- transmon χ (Koch et al. 2007)
{
  const t = R.transmonChi(R.CIRCUIT.g, R.CIRCUIT.Delta, R.CIRCUIT.alpha);
  check('χ = g²α/(Δ(Δ+α)) for the default circuit ≈ −0.991 MHz', close(t.chi, 91 * 91 * -220 / (-1250 * -1470), 1e-12) && Math.abs(t.chi + 0.9915) < 0.001, t.chi.toFixed(4));
  check('the defaults match the circuit', Math.abs(Math.abs(t.chi) - R.DEFAULTS.chi) < 0.01 && Math.abs(t.ncrit - R.DEFAULTS.ncrit) < 1, `n_crit = ${t.ncrit.toFixed(1)}`);
  check('large detuning: matches g²α/(Δ(Δ+α)) far detuned', close(R.transmonChi(50, -5000, -200).chi, 2500 * -200 / (-5000 * -5200), 1e-12));
  check('κ = 2χ in the defaults', close(R.DEFAULTS.kappa, 2 * R.DEFAULTS.chi, 1e-12));
}

// ---- movable decision threshold (p.thrSigma: noise widths from the midpoint, toward |1⟩)
{
  const base = R.analyze(P()), same = R.analyze(P({ thrSigma: 0 }));
  check('thrSigma = 0 is the midpoint line (unchanged F, P(1|0), P(0|1))', same.F === base.F && same.p10 === base.p10 && same.p01 === base.p01);
  const sh = R.analyze(P({ thrSigma: 0.7, T1: Infinity }));
  check('threshold moves by thrSigma·σ', close(sh.thr - sh.mid, 0.7 * sh.sigma, 1e-12));
  check('no T₁: P(1|0) = ½erfc((SNR/2 + s)/√2) at s = 0.7', close(sh.p10, 0.5 * R.erfc((sh.snr / 2 + 0.7) / Math.SQRT2), 1e-9));
  check('no T₁: P(0|1) = ½erfc((SNR/2 − s)/√2) at s = 0.7', close(sh.p01, 0.5 * R.erfc((sh.snr / 2 - 0.7) / Math.SQRT2), 1e-9));
  const m = R.analyze(P({ thrSigma: -0.5, T1: Infinity })), pl = R.analyze(P({ thrSigma: 0.5, T1: Infinity }));
  check('the two error rates swap under s → −s (equal blobs)', close(m.p10, pl.p01, 1e-12) && close(m.p01, pl.p10, 1e-12));
  const b0 = R.bestThreshold(P({ T1: Infinity, pexc: 0 }));
  check('equal Gaussian blobs: the best line is the midpoint', Math.abs(b0.thrSigma) < 1e-3, `s* = ${b0.thrSigma.toExponential(1)}`);
  const q = P({ T1: 3, tau: 0.6, nbar: 3 }), bq = R.bestThreshold(q), mid = R.analyze(q);
  check('with T₁ decay the best line moves toward |0⟩ and beats the midpoint', bq.thrSigma < -0.3 && bq.F > mid.F + 1e-4, `s* = ${bq.thrSigma.toFixed(2)}σ, F ${bq.F.toFixed(4)} vs ${mid.F.toFixed(4)} at the midpoint`);
  check('... and no other scanned line does better', [-2, -1.5, -0.5, 0, 0.5].every(s => R.analyze(Object.assign({}, q, { thrSigma: s })).F <= bq.F + 1e-9));
  const r = P({ thrSigma: -0.8, T1: 3, nbar: 1.2, tau: 0.5 }), e = R.analyze(r), n = 40000, s2 = R.shots(r, n, 7), se = v => 3.5 * Math.sqrt(v * (1 - v) / n);
  check('shots use the moved line: P(1|0) matches', Math.abs(s2.wrong[0] / n - e.p10) < se(e.p10), `${(s2.wrong[0] / n).toFixed(4)} vs ${e.p10.toFixed(4)}`);
  check('shots use the moved line: P(0|1) matches', Math.abs(s2.wrong[1] / n - e.p01) < se(e.p01), `${(s2.wrong[1] / n).toFixed(4)} vs ${e.p01.toFixed(4)}`);
  check('shots report the moved threshold', s2.thr === e.thr);
}

console.log(failures ? `\n${failures} FAILED` : '\nAll readout engine checks passed.');
process.exit(failures ? 1 : 0);
