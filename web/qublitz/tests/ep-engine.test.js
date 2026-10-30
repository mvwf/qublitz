// Checks web/qublitz/ep-engine.js against the Python original.
// Reference data: ep-reference.json, written by scripts/make_ep_reference.py.
// Usage: node web/qublitz/tests/ep-engine.test.js
'use strict';
const fs = require('fs');
const path = require('path');
const E = require('../ep-engine.js');

const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'ep-reference.json'), 'utf8'));
let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const num = v => (typeof v === 'string' ? ({ inf: Infinity, '-inf': -Infinity, nan: NaN })[v] : v);

// Largest relative error over two arrays; infinities and NaNs must match exactly.
// `floor` sets the scale below which differences count as absolute, for quantities
// such as the instability that cross zero (where both sides are rounding noise).
function relErr(got, want, floor = 1e-300) {
  let e = 0;
  for (let i = 0; i < want.length; i++) {
    const a = got[i], b = num(want[i]);
    if (!Number.isFinite(b)) { if (!(Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b)))) return Infinity; continue; }
    e = Math.max(e, Math.abs(a - b) / Math.max(Math.abs(b), floor));
  }
  return e;
}
const phase = k => E.PHASES[k][1];
const TOL = 1e-9;

for (const c of ref.fields) {
  const f = E.fields(phase(c.k), c.kc), n = f.x.length, S = ref.stride;
  const sub = arr => { const out = []; for (let j = 0; j < n; j += S) for (let i = 0; i < n; i += S) out.push(arr[j * n + i]); return out; };
  const e = Math.max(relErr(sub(f.K), c.K.flat()), relErr(sub(f.instability), c.instability.flat(), 1),
    relErr(sub(f.minPetermann), c.min_petermann.flat(), 1), relErr(sub(f.disc), c.disc.flat(), 1),
    relErr(sub(f.q), c.q.flat(), 1));
  check(`fields: φ = ${E.PHASES[c.k][0]}, κc = ${c.kc}`, e < TOL, `max error ${e.toExponential(1)}`);
}

// Degeneracies are compared as sets: numpy.roots and the JS root finder list roots in different orders.
const key = l => l.map(([t, dk, df]) => [t, dk, df]).sort((a, b) => a[0].localeCompare(b[0]) || a[1] - b[1] || a[2] - b[2]);
let locWorst = 0, locBad = [];
for (const c of ref.locations) {
  const got = key([...E.epLocation(phase(c.k)), ...E.tpdLocation(phase(c.k), c.kc)].map(d => [d.type, d.dk, d.df]));
  const want = key([...c.ep, ...c.tpd]);
  let e = got.length === want.length ? 0 : Infinity;
  if (Number.isFinite(e)) got.forEach((g, i) => {
    if (g[0] !== want[i][0]) e = Infinity;
    else e = Math.max(e, Math.abs(g[1] - want[i][1]) / Math.max(1, Math.abs(want[i][1])), Math.abs(g[2] - want[i][2]) / Math.max(1, Math.abs(want[i][2])));
  });
  locWorst = Math.max(locWorst, e);
  if (!(e < TOL)) locBad.push(`φ=${E.PHASES[c.k][0]} κc=${c.kc}`);
}
check(`EP and TPD locations (${ref.locations.length} cases)`, locWorst < TOL,
  `max error ${locWorst.toExponential(1)}${locBad.length ? '  failing: ' + locBad.join(', ') : ''}`);

for (const c of ref.splitting) {
  const s = E.splitting(phase(c.k), c.kc), S = ref.sweep_stride;
  const sub = arr => Array.from(arr).filter((_, i) => i % S === 0);
  let e = Math.max(relErr(sub(s.x), c.x), relErr(sub(s.nuPlus), c.nu_plus, 1), relErr(sub(s.nuMinus), c.nu_minus, 1),
    relErr(sub(s.nu0), c.nu_0, 1), relErr(sub(s.lamPlus), c.lambda_plus, 1), relErr(sub(s.lamMinus), c.lambda_minus, 1),
    relErr([s.tpd, s.ep, s.ymin, s.ymax], [c.tpd, c.ep, c.ymin, c.ymax], 1));
  if ((s.instability === null) !== (c.instability === null)) e = Infinity;
  else if (c.instability !== null) e = Math.max(e, Math.abs(s.instability - c.instability));
  check(`peak splitting: φ = ${E.PHASES[c.k][0]}, κc = ${c.kc}`, e < TOL, `max error ${e.toExponential(1)}`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
