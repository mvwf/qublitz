// Checks the hydrogen-atom physics in web/courses/engs23/hydrogen-engine.js (the Hydrogen Atom page).
// Usage: node web/courses/engs23/tests/hydrogen-engine.test.js
'use strict';
const H = require('../hydrogen-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const rel = (a, b) => Math.abs(a - b) / Math.abs(b);

// ---- radial functions: normalization, orthogonality, nodes, ⟨r⟩ (Griffiths 4.89 and Problem 4.13-type results) ----
{
  const integ = (f, rmax) => { const N = 40000, h = rmax / N; let s = 0; for (let i = 0; i <= N; i++) { const w = i === 0 || i === N ? 1 : i % 2 ? 4 : 2; s += w * f(i * h); } return s * h / 3; };
  let worstN = 0, worstO = 0, worstR = 0;
  for (let n = 1; n <= H.NMAX; n++) for (let l = 0; l < n; l++) {
    const rm = 8 * n * n + 40;
    worstN = Math.max(worstN, Math.abs(integ(r => (H.radialR(n, l, r) * r) ** 2, rm) - 1));
    worstR = Math.max(worstR, rel(integ(r => H.radialR(n, l, r) ** 2 * r ** 3, rm), H.meanR(n, l)));
    for (let n2 = l + 1; n2 < n; n2++) worstO = Math.max(worstO, Math.abs(integ(r => H.radialR(n, l, r) * H.radialR(n2, l, r) * r * r, rm)));
  }
  check('∫ R_nl² r² dr = 1 for every n ≤ 7', worstN < 1e-8, `max error ${worstN.toExponential(1)}`);
  check('R_nl ⟂ R_n′l (same l)', worstO < 1e-8, `max overlap ${worstO.toExponential(1)}`);
  check('⟨r⟩ = (a/2)[3n² − l(l+1)]', worstR < 1e-8, `max rel. error ${worstR.toExponential(1)}`);
  // closed forms from Griffiths Table 4.7
  check('R_10 = 2 e^{−r}', rel(H.radialR(1, 0, 0.7), 2 * Math.exp(-0.7)) < 1e-12);
  check('R_21 = (1/√24) r e^{−r/2}', rel(H.radialR(2, 1, 1.3), 1.3 * Math.exp(-0.65) / Math.sqrt(24)) < 1e-12);
  check('R_20 = (1/√2)(1 − r/2) e^{−r/2}', rel(H.radialR(2, 0, 0.9), (1 - 0.45) * Math.exp(-0.45) / Math.sqrt(2)) < 1e-12);
  // radial nodes: n − l − 1 sign changes
  let ok = true;
  for (let n = 1; n <= H.NMAX; n++) for (let l = 0; l < n; l++) {
    let ch = 0, prev = H.radialR(n, l, 1e-3);
    for (let r = 0.01; r < 8 * n * n; r += 0.01) { const v = H.radialR(n, l, r); if (v * prev < 0) ch++; if (v !== 0) prev = v; }
    if (ch !== n - l - 1) ok = false;
  }
  check('R_nl has n − l − 1 radial nodes', ok);
}

// ---- angular part ----
{
  let worst = 0;
  for (let l = 0; l <= 4; l++) for (let m = -l; m <= l; m++) {
    const N = 2000; let s = 0;
    for (let i = 0; i <= N; i++) { const th = Math.PI * i / N, w = i === 0 || i === N ? 1 : i % 2 ? 4 : 2; s += w * H.Ylm2(l, m, th) * Math.sin(th); }
    worst = Math.max(worst, Math.abs(2 * Math.PI * s * Math.PI / N / 3 - 1));
  }
  check('∮ |Y_l^m|² dΩ = 1 (l ≤ 4)', worst < 1e-7, `max error ${worst.toExponential(1)}`);
  check('|Y_1^0|² = (3/4π) cos²θ', rel(H.Ylm2(1, 0, 0.4), 3 / (4 * Math.PI) * Math.cos(0.4) ** 2) < 1e-12);
}

// ---- energies and wavelengths (NIST ASD, vacuum; our gross-structure values sit within 0.002 nm of the measured lines) ----
{
  check('E_1 = −13.598 eV (reduced mass)', Math.abs(H.energyEV(1) + 13.5984) < 1e-3, H.energyEV(1).toFixed(4));
  // NIST Atomic Spectra Database (Kramida, Ralchenko, Reader & NIST ASD Team), H I, vacuum wavelengths of the multiplets
  const ref = [[1, 2, 121.567], [1, 3, 102.572], [1, 4, 97.2537], [1, 5, 94.9743], [2, 3, 656.46], [2, 4, 486.27], [2, 5, 434.17], [3, 4, 1875.6]];
  // fine structure and the Lamb shift, which the gross-structure model leaves out, move lines by ~α²/n ~ 10⁻⁵
  let worst = 0; for (const [a, b, nm] of ref) worst = Math.max(worst, rel(H.wavelength(a, b) * 1e9, nm));
  check('Lyman, Balmer and Paschen lines match NIST (vacuum)', worst < 2e-5, `max |Δλ|/λ ${worst.toExponential(1)}`);
  check('Lyman limit 91.18 nm', Math.abs(1e9 / H.K.RH - 91.176) < 0.01);
}

// ---- radial integrals and spontaneous emission ----
{
  check('∫ R_21 R_10 r³ dr = 128√6/243', rel(H.radialIntegral(2, 1, 1, 0), 128 * Math.sqrt(6) / 243) < 1e-9);
  check('A(2p→1s) = 6.2649×10⁸ s⁻¹', rel(H.einsteinA(2, 1, 1, 0), 6.2649e8) < 2e-3, H.einsteinA(2, 1, 1, 0).toExponential(4));
  // non-relativistic lifetimes (Bethe & Salpeter; Wiese & Fuhr 2009 tables), ns
  const ref = { '2,1': 1.596, '3,0': 158.4, '3,1': 5.271, '3,2': 15.47, '4,0': 226.6, '4,1': 12.30, '4,2': 36.16, '4,3': 72.53, '5,1': 23.79 };
  let worst = 0, txt = [];
  for (const k in ref) { const [n, l] = k.split(',').map(Number), t = H.lifetime(n, l) * 1e9; worst = Math.max(worst, rel(t, ref[k])); txt.push(`${H.label(n, l)} ${t.toFixed(2)}`); }
  check('lifetimes match the tabulated values', worst < 3e-3, txt.join(', '));
  const b = H.decays(3, 1); check('3p branches 88.2% to 1s, 11.8% to 2s', Math.abs(b.find(d => d.n === 1).branch - 0.8817) < 1e-3);
  check('2s has no dipole decay; two-photon τ = 0.122 s', H.einsteinA(2, 0, 1, 0) === 0 && Math.abs(H.lifetime(2, 0) - 0.1215) < 1e-3);
  check('no Δl = 0 or |Δl| = 2 emission', H.einsteinA(3, 0, 2, 0) === 0 && H.einsteinA(3, 2, 1, 0) === 0);
  // cascade conserves probability: every excitation ends in 1s
  const c = H.cascade(5, 1), into1s = c.filter(x => x.to[0] === 1).reduce((s, x) => s + x.num, 0);
  check('a 5p cascade always ends in 1s', Math.abs(into1s - 1) < 1e-12);
}

// ---- driving ----
{
  const L = H.lines([1, 0]);
  check('from 1s, π light reaches only np (n = 2…7)', L.length === 6 && L.every(x => x.l === 1));
  check('⟨2p0|z|1s0⟩ = 128√2/243 a', rel(H.zElement(1, 0, 2, 1), 128 * Math.sqrt(2) / 243) < 1e-9);
  check('Lyman-α I_sat ≈ 7.25 W/cm² (two-level, π light)', rel(H.Isat(L[0], [1, 0]) / 1e4, 7.25) < 0.01);
  check('Lyman-α Doppler FWHM at 300 K ≈ 30.5 GHz', rel(H.dopplerFWHM(L[0].nu, 300), 30.47e9) < 0.01);
  // s = I/I_sat and the steady state of the Bloch equations
  const I = 2 * H.Isat(L[0], [1, 0]), r = H.lineResponse(L[0], L[0].nu, I, 0, 0, [1, 0]);
  check('s = I/I_sat', Math.abs(r.s - 2) < 1e-9);
  check('cold, on resonance: ρ_ee = (s/2)/(1+s)', rel(r.rho, 1 / 3) < 1e-6, r.rho.toFixed(6));
  const ob = H.bloch(r.Om, 0, L[0].Gamma, 30 * L[0].tau, 200);
  check('Bloch equations settle to the same ρ_ee', Math.abs(ob.rho[ob.rho.length - 1] - 1 / 3) < 1e-4);
  const ob2 = H.bloch(40 * L[0].Gamma, 0, L[0].Gamma, 0.1 * L[0].tau, 400);
  check('strong drive: Rabi flopping reaches ρ_ee ≈ 1', Math.max(...ob2.rho) > 0.95);
  // Doppler averaging conserves the line area
  const T = 300, step = 2e7; let a0 = 0, a1 = 0;
  for (let d = -3e11; d <= 3e11; d += step) { a0 += H.lineResponse(L[0], L[0].nu + d, 1e-3, 0, 0, [1, 0]).rho * step; a1 += H.lineResponse(L[0], L[0].nu + d, 1e-3, T, 0, [1, 0]).rho * step; }
  check('Doppler broadening conserves the line area', rel(a1, a0) < 0.02, `${(a1 / a0).toFixed(4)}`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
