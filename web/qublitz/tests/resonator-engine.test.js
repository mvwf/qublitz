// Checks web/qublitz/resonator-engine.js: the models against their closed forms, that fits recover the
// parameters of simulated data with honest error bars, the formula parser, and the file readers.
// Usage: node web/qublitz/tests/resonator-engine.test.js
'use strict';
const R = require('../resonator-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const close = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const at = (model, p, f) => { const s = R.sweep(model, p, new Float64Array([f])); return [s.re[0], s.im[0]]; };

// ---- models: values on resonance and far away
{
  const p = { fr: 5, Qi: 1e5, Qc: 3e5, phi: 0 };
  const Ql = R.loadedQ(p.Qi, p.Qc);
  check('1/Ql = 1/Qi + 1/Qc', close(1 / Ql, 1 / 1e5 + 1 / 3e5, 1e-12));
  check('internalQ inverts loadedQ, with asymmetry', close(R.internalQ(R.loadedQ(1e5, 3e5, 0.4), 3e5, 0.4), 1e5, 1e-12));
  check('hanger on resonance = Ql/Qi', close(at('hanger', p, 5)[0], Ql / p.Qi, 1e-12));
  check('reflection at critical coupling = 0', Math.hypot(...at('reflection', { fr: 5, Qi: 2e5, Qc: 2e5 }, 5)) < 1e-12);
  check('reflection on resonance = (Qc − Qi)/(Qc + Qi)', close(at('reflection', p, 5)[0], (3e5 - 1e5) / 4e5, 1e-12));
  check('transmission on resonance = Ql/Qc', close(at('transmission', p, 5)[0], Ql / p.Qc, 1e-12));
  check('hanger far from resonance → 1', close(at('hanger', p, 6)[0], 1, 1e-6));
  // half power: |1 − S21|² falls to half at f = fr(1 ± 1/2Ql)
  const d = f => { const [x, y] = at('hanger', p, f); return (1 - x) ** 2 + y ** 2; };
  check('linewidth: half power at fr ± fr/2Ql', close(d(5 * (1 + 0.5 / Ql)) / d(5), 0.5, 1e-9));
  // environment: |S| scaled by a, delay winds the phase by −2πfτ
  const e = at('hanger', Object.assign({}, p, { a: 0.5, alpha: 0.3, tau: 10 }), 6), e0 = at('hanger', p, 6);
  // ratio e/e0 should be 0.5·exp(i(0.3 − 2π·6·10))
  const den = e0[0] ** 2 + e0[1] ** 2, rr = (e[0] * e0[0] + e[1] * e0[1]) / den, ri = (e[1] * e0[0] - e[0] * e0[1]) / den;
  const th = 0.3 - 2 * Math.PI * 6 * 10;
  check('environment scales by a and rotates by α − 2πfτ', close(rr, 0.5 * Math.cos(th), 1e-9) && close(ri, 0.5 * Math.sin(th), 1e-9));
  check('photon number: 2Ql²P/(Qc ħ ω²) for a hanger', close(R.photonNumber('hanger', -30, 5, 1e5, 2e5),
    2 * 1e10 * 1e-6 / (2e5 * 1.054571817e-34 * (2 * Math.PI * 5e9) ** 2), 1e-12));
  check('powerForPhotons inverts photonNumber', close(R.photonNumber('reflection', R.powerForPhotons('reflection', 1, 6, 1e5, 1e5), 6, 1e5, 1e5), 1, 1e-9));
}

// ---- TLS loss: 1/Qi = (1/Qtls0) tanh(hf/2kT)/(1 + n/nc)^β + 1/Qother
{
  const h = 6.62607015e-34, kB = 1.380649e-23;
  const P = { Qtls0: 1e6, Qother: 4e6, nc: 3, beta: 0.4 };
  const par = 1 / (1 / P.Qtls0 + 1 / P.Qother);
  check('TLS: n → 0, T → 0 gives Qtls0 ∥ Qother', close(R.tlsQi(0, 1e-9, 5, P), par, 1e-12) && close(R.tlsQi(0, 0, 5, P), par, 1e-12), R.tlsQi(0, 0, 5, P).toFixed(1));
  check('TLS: n ≫ nc gives Qother', close(R.tlsQi(1e15, 0.02, 5, P), P.Qother, 1e-5));
  check('TLS: T ≫ hf/k gives Qother', close(R.tlsQi(0, 1e6, 5, P), P.Qother, 1e-5));
  const Ts = R.tlsCrossoverT(5);
  check('TLS: crossover hf = 2kT at 5 GHz is 120 mK', close(Ts, h * 5e9 / (2 * kB), 1e-12) && Math.abs(Ts - 0.12) < 1e-3, `${(1e3 * Ts).toFixed(2)} mK`);
  check('TLS: thermal factor is tanh(1) at hf = 2kT', close(R.tlsThermal(5, Ts), Math.tanh(1), 1e-12) && close(R.tlsThermal(5, Ts), 0.7616, 1e-4));
  check('TLS: at hf = 2kT and n = 0 the TLS loss is tanh(1)/Qtls0', close(R.tlsLoss(0, Ts, 5, P), Math.tanh(1) / P.Qtls0, 1e-12));
  check('TLS: at n = nc and T → 0 the TLS loss is 2^−β/Qtls0', close(R.tlsLoss(P.nc, 1e-9, 5, P), Math.pow(2, -P.beta) / P.Qtls0, 1e-12));
  // high-power slope on log-log: d ln(1/Q_TLS)/d ln n → −β for nc ≪ n, with Qother out of the way
  const Pinf = Object.assign({}, P, { Qother: Infinity });
  const slope = (Math.log(R.tlsQi(1e8, 0.01, 5, Pinf)) - Math.log(R.tlsQi(1e7, 0.01, 5, Pinf))) / Math.log(10);
  check('TLS: Qi ∝ n^β well above nc', close(slope, P.beta, 1e-6), slope.toFixed(6));
  // monotonic in both n and T; frequency enters only through hf/2kT
  let mono = true, last = 0;
  for (let e = -3; e <= 7; e += 0.25) { const q = R.tlsQi(Math.pow(10, e), 0.05, 5, P); if (q <= last) mono = false; last = q; }
  last = 0;
  for (let e = -2; e <= 0.5; e += 0.1) { const q = R.tlsQi(1, Math.pow(10, e), 5, P); if (q <= last) mono = false; last = q; }
  check('TLS: Qi rises monotonically with n and with T', mono);
  check('TLS: f and T enter only as f/T', close(R.tlsQi(2, 0.1, 5, P), R.tlsQi(2, 0.2, 10, P), 1e-12));
  check('TLS: refuses nonsense (negative n, zero nc)', Number.isNaN(R.tlsQi(-1, 0.1, 5, P)) && Number.isNaN(R.tlsLoss(1, 0.1, 5, { Qtls0: 1e6, nc: 0, beta: 0.5 })));
}

// ---- fits: recover the truth from practice data, and the 1σ ranges cover it about 68% of the time
{
  let n = 0, bad = 0;
  const cover = { Qi: 0, Qc: 0, fr: 0 };
  for (const model of ['hanger', 'reflection', 'transmission']) for (const level of [1, 2, 3]) for (let seed = 1; seed <= 25; seed++) {
    const P = R.practice(seed, level, model);
    const F = R.fitResonator(model, { f: P.f, re: P.re, im: P.im });
    const D = F.derived;
    if (!F.converged || !(D.Qi.sigma > 0) || Math.abs(D.Qi.value - P.truth.Qi) > 5 * D.Qi.sigma) { bad++; continue; }
    n++;
    for (const k of Object.keys(cover)) if (Math.abs(D[k].value - P.truth[k]) <= D[k].sigma) cover[k]++;
  }
  check('fits converge near the truth on 225 practice sets', bad <= 2, `${bad} failed`);
  for (const k of Object.keys(cover)) {
    const c = cover[k] / n;
    check(`1σ coverage of ${k} is about 68%`, c > 0.58 && c < 0.78, `${(100 * c).toFixed(0)}%`);
  }
}
{
  // magnitude only, hanger: no phase needed
  const P = R.practice(4, 1, 'hanger');
  const F = R.fitResonator('hanger', { f: P.f, mag: Float64Array.from(P.re, (x, j) => Math.hypot(x, P.im[j])) });
  check('hanger |S21|-only fit finds Qi', F.converged && Math.abs(F.derived.Qi.value - P.truth.Qi) < 4 * F.derived.Qi.sigma,
    `${F.derived.Qi.value.toExponential(3)} vs ${P.truth.Qi.toExponential(3)}`);
}

// ---- formula parser and formula fits
{
  const v = (s, P = {}, f = 2) => { const r = R.evaluate(R.parse(s).ast, new Float64Array([f]), P); return [r.re[0], r.im ? r.im[0] : 0]; };
  check('precedence and right-associative power', v('2 + 3*4^2^0.5 - -1')[0] === 2 + 3 * Math.pow(4, Math.pow(2, 0.5)) + 1);
  check('unary minus binds looser than ^', v('-x^2', { x: 3 })[0] === -9);
  check('complex arithmetic: (1 + 2i)(3 − i) = 5 + 5i', v('(1 + 2*i)*(3 - i)').join() === '5,5');
  check('sqrt of a negative number is imaginary', close(v('sqrt(-4)')[1], 2, 1e-15));
  check('exp(i*pi) = −1', close(v('exp(i*pi)')[0], -1, 1e-15) && Math.abs(v('exp(i*pi)')[1]) < 1e-15);
  check('the variable f', v('f^2', {}, 3)[0] === 9);
  check('parameters in order of first use', R.parse('A/(1 + ((f - f0)/w)^2) + A*C').params.join() === 'A,f0,w,C');
  for (const [s, want] of [['2f', 'side by side'], ['sin x', 'brackets'], ['a*(b', 'expected ")"'], ['foo(1)', 'not a known function'],
    ['1 +', 'ends too early'], ['a $ b', 'not allowed'], ['a)', 'without its']]) {
    let msg = ''; try { R.parse(s); } catch (e) { msg = e.message; }
    check(`parser rejects "${s}"`, msg.includes(want), msg);
  }
  const hang = 'a*exp(i*alpha)*exp(-2*pi*i*f*tau)*(1 - Ql/Qc*exp(i*phi)/(1 + 2*i*Ql*(f/fr - 1)))';
  const p = { fr: 5, Ql: 4e4, Qc: 6e4, phi: 0.2, a: 0.7, alpha: 1.1, tau: 30 };
  const f = Float64Array.from({ length: 50 }, (_, j) => 5 - 2e-4 + 4e-4 * j / 49);
  const s1 = R.sweep('hanger', p, f), s2 = R.evaluate(R.parse(hang).ast, f, p);
  let e = 0; for (let j = 0; j < f.length; j++) e = Math.max(e, Math.hypot(s1.re[j] - s2.re[j], s1.im[j] - s2.im[j]));
  check('the hanger written as a formula equals the built-in model', e < 1e-12, e.toExponential(1));

  const P = R.practice(3, 2, 'hanger');
  const g = R.guess('hanger', P.f, P.re, P.im);
  const F = R.fitFormula(hang, 'complex', { f: P.f, re: P.re, im: P.im }, g, {}, ['Qi = 1/(1/Ql - cos(phi)/Qc)']);
  const B = R.fitResonator('hanger', { f: P.f, re: P.re, im: P.im });
  check('formula fit gives the same Qi and σ as the built-in fit', close(F.derived[0].value, B.derived.Qi.value, 1e-4) && close(F.derived[0].sigma, B.derived.Qi.sigma, 0.02),
    `${F.derived[0].value.toExponential(4)} ± ${F.derived[0].sigma.toExponential(2)} vs ${B.derived.Qi.value.toExponential(4)} ± ${B.derived.Qi.sigma.toExponential(2)}`);
  // a Lorentzian on |S21|² of a transmission resonator: Ql = f0/w
  const T = R.practice(5, 1, 'transmission');
  const y = Array.from(T.re, (x, j) => x * x + T.im[j] * T.im[j]);
  const L = R.fitFormula('A/(1 + (2*(f - f0)/w)^2)', 'pow', { f: T.f, y }, { A: 0.5, f0: T.f[200], w: (T.f[400] - T.f[0]) / 10 }, {}, ['Ql = f0/w']);
  const QlT = R.loadedQ(T.truth.Qi, T.truth.Qc);
  check('Lorentzian fit of |S21|² gives Ql', Math.abs(L.derived[0].value - QlT) < 4 * L.derived[0].sigma, `${L.derived[0].value.toFixed(0)} vs ${QlT.toFixed(0)}`);
  check('fixed parameters stay fixed', R.fitFormula('A/(1 + (2*(f - f0)/w)^2)', 'pow', { f: T.f, y }, { A: 0.5, f0: T.f[200], w: 1e-3 }, { w: true }).params.w === 1e-3);
  check('derived quantities name unknown parameters', R.fitFormula('A*f', 'y', { f: T.f, y }, { A: 1 }, {}, ['B = A*Z']).derived[0].error.includes('Z'));
  check('Python export uses numpy names', /np\.exp\(1j \* alpha\)/.test(R.toPython(R.parse(hang).ast)) && /\*\*/.test(R.toPython(R.parse('x^2').ast)));
}

// ---- readers
{
  const T = R.parseTable('# comment\nfreq_GHz,re,im\n5.0,0.9,0.1\n5.1,0.8,0.2\nbad,row,here\n5.2,0.7,0.3\n');
  check('CSV with header and a bad row', T.header.join() === 'freq_GHz,re,im' && T.cols[0].length === 3 && T.skipped === 1);
  check('whitespace-separated table', R.parseTable('1 2\n3 4\n5 6').cols[1][2] === 6);
  const s2p = '! two-port\n# MHz S DB R 50\n5000 -40 0 -6.0206 90 -40 0 -40 0\n5001 -40 0 0 0 -40 0 -40 0\n';
  const t = R.parseTouchstone(s2p);
  check('Touchstone .s2p in MHz and dB/angle', close(t.fGHz[0], 5, 1e-12) && close(t.S.S21.re[0], 0, 1e-9) && close(t.S.S21.im[0], 0.5, 1e-4) && t.S.S21.re[1] === 1);
  const s1p = '# Hz S RI R 50\n5e9 0.5 -0.5\n5.1e9 0.4 -0.4\n';
  const t1 = R.parseTouchstone(s1p);
  check('Touchstone .s1p in Hz and real/imaginary', !t1.S.S21 && t1.S.S11.im[1] === -0.4 && close(t1.fGHz[1], 5.1, 1e-12));
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
