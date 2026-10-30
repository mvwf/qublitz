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

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
