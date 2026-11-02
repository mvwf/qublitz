// Checks the circular-membrane math in web/courses/engs23/drum-engine.js (the Drum Head page).
// Usage: node web/courses/engs23/tests/drum-engine.test.js
'use strict';
const D = require('../drum-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// ---- Bessel values (Abramowitz & Stegun, Table 9.1; J_10(30) checked by Miller's backward recurrence) ----
{
  const ref = [[0, 1, 0.765197686557967], [1, 1, 0.440050585744934], [0, 10, -0.245935764451348],
    [1, 10, 0.043472746168862], [5, 10, -0.234061528186794], [2, 5, 0.046565116277752], [10, 30, -0.129876893998589]];
  let worst = 0;
  for (const [n, x, v] of ref) worst = Math.max(worst, Math.abs(D.besselJ(n, x) - v));
  check('J_n(x) matches tabulated values', worst < 1e-12, `max error ${worst.toExponential(1)}`);
  check('J_{-1}(x) = −J_1(x)', close(D.besselJ(-1, 2.3), -D.besselJ(1, 2.3), 1e-14));
}

// ---- zeros k_mn (A&S Table 9.5; Boas §6 Problem 2 asks for these) ----
{
  const ref = {
    0: [2.404825557695773, 5.520078110286311, 8.653727912911012, 11.79153443901428],
    1: [3.831705970207512, 7.015586669815619, 10.17346813506272, 13.32369193631422],
    2: [5.135622301840683, 8.417244140399865, 11.61984117214906, 14.79595178235126],
    3: [6.380161895923984, 9.761023129981670, 13.01520072169843, 16.22346616031877],
    4: [7.588342434503805, 11.06470948850119, 14.37253667161759, 17.61596604980483],
    5: [8.771483815959954, 12.33860419746694, 15.70017407971167, 18.98013387517992],
  };
  let worst = 0;
  for (const n in ref) D.besselZeros(+n, 4).forEach((z, i) => { worst = Math.max(worst, Math.abs(z - ref[n][i])); });
  check('first four zeros of J_0 … J_5', worst < 1e-12, `max error ${worst.toExponential(1)}`);
  const z8 = D.besselZeros(8, 1)[0];
  check('first zero of J_8 (no zero skipped at high order)', close(z8, 12.225092264004655, 1e-11), z8.toFixed(12));
}

const modes = D.buildModes(6, 5);

// ---- Problem 2: the first six frequencies as multiples of the fundamental ----
{
  const first = modes.slice(0, 6).map(m => m.ratio);
  const j = [2.404825557695773, 3.831705970207512, 5.135622301840683, 5.520078110286311, 6.380161895923984, 7.015586669815619];
  const ref = j.map(k => k / j[0]);
  check('ν_mn/ν_10 for the six lowest modes', first.every((r, i) => close(r, ref[i], 1e-9)), first.map(r => r.toFixed(4)).join(', '));
  check('lowest six modes are (1,0) (1,1) (1,2) (2,0) (1,3) (2,1)',
    modes.slice(0, 6).map(m => `${m.m}${m.n}`).join(' ') === '10 11 12 20 13 21');
  check('no overtone is an integer multiple of the fundamental',
    modes.slice(1).every(m => Math.abs(m.ratio - Math.round(m.ratio)) > 1e-3));
}

// ---- each mode solves the Helmholtz equation (6.3) and vanishes on the rim ----
{
  let worst = 0, rim = 0;
  const h = 2e-4;
  for (const md of modes.filter(m => m.k < 12)) {
    const F = (x, y) => D.besselJ(md.n, md.k * Math.hypot(x, y)) * Math.cos(md.n * Math.atan2(y, x));
    for (const [x, y] of [[0.31, 0.12], [-0.2, 0.45], [0.05, -0.6]]) {
      const lap = (F(x + h, y) + F(x - h, y) + F(x, y + h) + F(x, y - h) - 4 * F(x, y)) / (h * h);
      worst = Math.max(worst, Math.abs(lap + md.k * md.k * F(x, y)));
    }
    rim = Math.max(rim, Math.abs(D.besselJ(md.n, md.k)));
  }
  check('∇²F + K²F = 0 inside the disk (5-point stencil, K²F up to ~70)', worst < 1e-4, `residual ${worst.toExponential(1)}`);
  check('F = 0 on r = a (clamped rim)', rim < 1e-13);
}

// ---- normal modes are orthogonal with weight r, and the norm is J_{n+1}(k)²/2 ----
{
  const { x, w } = D.gaussLegendre(80);
  let worstOff = 0, worstNorm = 0;
  for (const p of modes) for (const q of modes) {
    if (p.n !== q.n || q.index < p.index) continue;
    let s = 0;
    for (let i = 0; i < x.length; i++) s += D.besselJ(p.n, p.k * x[i]) * D.besselJ(q.n, q.k * x[i]) * x[i] * w[i];
    if (p === q) worstNorm = Math.max(worstNorm, Math.abs(s - p.normR));
    else worstOff = Math.max(worstOff, Math.abs(s));
  }
  check('∫ J_n(k_m ρ) J_n(k_m′ ρ) ρ dρ = 0 for m ≠ m′', worstOff < 1e-12, worstOff.toExponential(1));
  check('∫ J_n(kρ)² ρ dρ = J_{n+1}(k)²/2', worstNorm < 1e-12, worstNorm.toExponential(1));
}

// ---- projection recovers a pure mode, and the table lookup matches J_n ----
{
  const md = modes.find(m => m.m === 2 && m.n === 1);
  const coef = D.project(modes, (r, th) => D.besselJ(1, md.k * r) * (0.7 * Math.cos(th) - 0.3 * Math.sin(th)));
  let off = 0;
  coef.forEach(([c, s], i) => { if (i !== md.index) off = Math.max(off, Math.abs(c), Math.abs(s)); });
  check('projecting J_1(k_21 r)(0.7 cos θ − 0.3 sin θ) gives back 0.7 and −0.3',
    close(coef[md.index][0], 0.7, 1e-10) && close(coef[md.index][1], -0.3, 1e-10) && off < 1e-10);
  let worst = 0;
  for (const m of modes) for (const r of [0.1234, 0.5, 0.777, 0.999]) worst = Math.max(worst, Math.abs(D.radial(m, r) - D.besselJ(m.n, m.k * r)));
  check('radial table interpolation error', worst < 2e-4, worst.toExponential(1));
}

// ---- the series solution obeys the initial conditions ----
{
  const big = D.buildModes(12, 10), w = 0.15;
  const f = D.mallet(0.4, 0.8, w);
  const press = D.strike(big, 0.4, 0.8, w, 'press');
  let worst = 0;
  for (const [r, th] of [[0.4, 0.8], [0.2, 2], [0.6, -1], [0.0, 0]]) worst = Math.max(worst, Math.abs(D.displacement(big, press, r, th, 0) - f(r, th)));
  check('pressed head starts in the mallet\'s shape (truncated series, 143 modes)', worst < 5e-3, worst.toExponential(1));

  const hit = D.strike(big, 0.4, 0.8, w, 'strike');
  const dt = 1e-5;
  let flat = 0, vel = 0;
  for (const [r, th] of [[0.4, 0.8], [0.2, 2], [0.6, -1]]) {
    flat = Math.max(flat, Math.abs(D.displacement(big, hit, r, th, 0)));
    const v = (D.displacement(big, hit, r, th, dt) - D.displacement(big, hit, r, th, -dt)) / (2 * dt);
    vel = Math.max(vel, Math.abs(v - f(r, th)));
  }
  check('struck head starts flat', flat < 1e-15);
  check('struck head starts with the mallet\'s velocity', vel < 5e-3, vel.toExponential(1));
}

// ---- the solution is periodic only for a pure mode: z(t + T_mn) = z(t) ----
{
  const md = modes.find(m => m.m === 1 && m.n === 2), st = D.emptyState(modes);
  st[md.index].Ac = 1;
  const T = 1 / md.ratio;
  check('pure mode (1,2) repeats after 1/ν_12',
    close(D.displacement(modes, st, 0.5, 0.3, 0.37 + T), D.displacement(modes, st, 0.5, 0.3, 0.37), 1e-12));
}

// ---- physical frequency ----
{
  const md = modes[0], f = D.frequencyHz(md, 0.25, 4000, 0.25);
  check('ν_10 = k_10 √(T/σ) / (2πa)', close(f, 2.404825557695773 * Math.sqrt(16000) / (2 * Math.PI * 0.25), 1e-9), f.toFixed(3) + ' Hz');
}

if (failures) { console.log(`\n${failures} check(s) failed`); process.exit(1); }
console.log('\nall drum engine checks passed');
