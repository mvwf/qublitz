// Checks web/qublitz/fridge-engine.js against the Python original.
// Reference data: fridge-reference.json, written by scripts/make_fridge_reference.py.
// Usage: node web/qublitz/tests/fridge-engine.test.js
'use strict';
const fs = require('fs');
const path = require('path');
const F = require('../fridge-engine.js');

const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'fridge-reference.json'), 'utf8'));
let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const num = v => (typeof v === 'string' ? ({ inf: Infinity, '-inf': -Infinity, nan: NaN })[v] : v);

// Largest relative error over two arrays; infinities and NaNs must match exactly.
// `floor` sets the scale below which differences count as absolute, for quantities
// such as dB that are legitimately 0 (where both sides are rounding noise).
function relErr(got, want, floor = 1e-300) {
  let e = 0;
  for (let i = 0; i < want.length; i++) {
    const a = got[i], b = num(want[i]);
    if (!Number.isFinite(b)) { if (!(Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b)))) return Infinity; continue; }
    e = Math.max(e, Math.abs(a - b) / Math.max(Math.abs(b), floor));
  }
  return e;
}

const TOL = 1e-12;
for (const c of ref.cases) {
  const freqs = F.linspace(c.f_min, c.f_max, c.n_points);
  const { nEff, TEff } = F.propagateChain(freqs, c.temps, c.atten);
  const nRoom = Array.from(freqs, f => F.thermalN(c.temps[0], f));
  const red = nEff.map(row => Array.from(row, (n, k) => F.noiseReductionDB(nRoom[k], n)));
  const P = F.drivePowers(c.p_in, c.atten, c.cable);
  const flux = P.map(p => Array.from(freqs, f => F.photonFlux(p, f)));

  let e = relErr(freqs, c.freqs);
  for (let i = 0; i < c.temps.length; i++) {
    e = Math.max(e, relErr(nEff[i], c.n_eff[i]), relErr(TEff[i], c.T_eff[i]),
      relErr(red[i], c.reduction_dB[i], 1), relErr(flux[i], c.flux[i]));
  }
  e = Math.max(e, relErr(P, c.P_dBm));
  check(`chain: ${c.name}`, e < TOL, `max relative error ${e.toExponential(1)}`);
}

for (const [T, want] of ref.format_temp) {
  const got = F.formatTemp(T);
  check(`format_temp(${T})`, got === want, `${got} vs ${want}`);
}

// ---- what the noise does to the qubit (fridge-noise.html's consequence panel) ----
{
  const close = (a, b, tol) => Math.abs(a - b) <= tol;
  const n = 0.01, f = 5;                                   // the challenge threshold, at the judging frequency
  const T = F.nToTeff(n, f);
  check('n̄ = 0.01 at 5 GHz is T_eff ≈ 52.0 mK', close(T, 0.051995, 2e-5), `${(T * 1e3).toFixed(3)} mK`);
  check('n̄(T_eff) inverts back to n̄', close(F.thermalN(T, f), n, 1e-12));
  // two-level population: 1/(e^x + 1) with x = hf/kB T, for several n̄
  let worst = 0;
  for (const nn of [1e-4, 0.01, 0.1, 1, 10]) {
    const x = F.h * f * 1e9 / (F.kB * F.nToTeff(nn, f));
    worst = Math.max(worst, Math.abs(F.qubitExcitedPopulation(nn) - 1 / (Math.exp(x) + 1)),
      Math.abs(F.qubitBoltzmannRatio(nn) - Math.exp(-x)));
  }
  check('P_exc = n̄/(1 + 2n̄) equals 1/(e^{hf/kT} + 1), and n̄/(1 + n̄) equals e^{−hf/kT}', worst < 1e-12, worst.toExponential(1));
  check('P_exc(0.01) = 0.98 %', close(F.qubitExcitedPopulation(0.01), 0.0098039215686, 1e-12));
  check('P_exc → 1/2 for n̄ → ∞ (a two-level system never inverts thermally)', close(F.qubitExcitedPopulation(1e9), 0.5, 1e-9));

  // photon shot noise dephasing, κ and χ as angular rates
  const kappa = 2 * Math.PI * 1e6, chi = 2 * Math.PI * 0.5e6;   // κ/2π = 1 MHz, χ/2π = 0.5 MHz: 2χ = κ
  const G = F.photonDephasingRate(n, kappa, chi);
  check('Γφ(n̄ = 0.01, κ/2π = 1 MHz, χ/2π = 0.5 MHz) = n̄κ/2 = 3.14×10⁴ /s, Tφ = 31.8 µs',
    close(G, n * kappa / 2, 1e-6) && close(1e6 / G, 31.831, 1e-3), `Tφ = ${(1e6 / G).toFixed(3)} µs`);
  check('χ ≫ κ: Γφ → n̄κ (every photon dephases)', close(F.photonDephasingRate(n, kappa, 1e3 * chi) / (n * kappa), 1, 1e-5));
  check('χ ≪ κ: Γφ → 4χ²n̄/κ', close(F.photonDephasingRate(n, kappa, 1e-3 * chi) / (4 * (1e-3 * chi) ** 2 * n / kappa), 1, 1e-5));
  // the linearised form is the small-n̄ limit of the full Clerk–Utami expression
  const ratioSmall = F.photonDephasingRateExact(1e-5, kappa, chi) / F.photonDephasingRate(1e-5, kappa, chi);
  check('full Clerk–Utami rate → linear form as n̄ → 0', close(ratioSmall, 1, 1e-4), ratioSmall.toFixed(6));
  // at n̄ = 1 with 2χ = κ: z = (1 + i)² + 4i = 6i, Re √(6i) = √3, so Γφ = (κ/2)(√3 − 1)
  check('full rate at n̄ = 1, 2χ = κ equals (κ/2)(√3 − 1)', close(F.photonDephasingRateExact(1, kappa, chi), kappa / 2 * (Math.sqrt(3) - 1), 1e-6));
  check('full rate is below the linear one (saturation) for n̄ = 0.1', F.photonDephasingRateExact(0.1, kappa, chi) < F.photonDephasingRate(0.1, kappa, chi));
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
