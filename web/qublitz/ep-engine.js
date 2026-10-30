/* ep-engine.js: browser port of the model in pages/EP_TPD_exploration.py,
 * utils/tpd_locations_nd.py and utils/pt_peaks_MODEL.py. No dependencies. Loads as a
 * classic <script> (window.EpEngine) or from Node (require), so the same file is
 * what the page runs and what tests/ep-engine.test.js checks against Python.
 *
 * Model, same as the Python page (all quantities normalized by the coupling J = 1):
 *   Δλ² = −Δf² + 2iΔfΔκ + Δκ² − 4e^{iφ}
 *   Petermann factor K₂ = (Δf² + Δκ² + |Δλ|² + 4) / (2|Δλ|²)
 *   p̃ = ((κc − Δκ)² + Re Δλ²)/4,  q̃ = (κc − Δκ)·Im Δλ²/8,  disc = −4p̃³ − 27q̃²
 *   instability = Re Δλ − (κc − Δκ)   (≤ 0 is the stable region)
 * Coordinates: x is Δκ, y is Δf.
 */
(function (root) {
  'use strict';

  // The 17 stops of the page's phase slider; "2π" maps to 0, as in the Python page.
  const PHASES = [
    ['0', 0], ['π/8', Math.PI / 8], ['π/4', Math.PI / 4], ['3π/8', 3 * Math.PI / 8],
    ['π/2', Math.PI / 2], ['5π/8', 5 * Math.PI / 8], ['3π/4', 3 * Math.PI / 4],
    ['7π/8', 7 * Math.PI / 8], ['π', Math.PI], ['9π/8', 9 * Math.PI / 8],
    ['5π/4', 5 * Math.PI / 4], ['11π/8', 11 * Math.PI / 8], ['3π/2', 3 * Math.PI / 2],
    ['13π/8', 13 * Math.PI / 8], ['7π/4', 7 * Math.PI / 4], ['15π/8', 15 * Math.PI / 8], ['2π', 0],
  ];

  // numpy.linspace: start + i·step, with the last point pinned to stop.
  function linspace(a, b, n) {
    const out = new Float64Array(n), step = (b - a) / (n - 1);
    for (let i = 0; i < n; i++) out[i] = a + i * step;
    if (n > 1) out[n - 1] = b;
    return out;
  }

  // Principal square root of re + i·im, as numpy computes it (no cancellation for re < 0).
  function csqrt(re, im) {
    const m = Math.hypot(re, im);
    if (m === 0) return [0, im];
    if (re >= 0) { const t = Math.sqrt((re + m) / 2); return [t, im / (2 * t)]; }
    const t = Math.sqrt((m - re) / 2);
    const neg = im < 0 || Object.is(im, -0);
    return [Math.abs(im) / (2 * t), neg ? -t : t];
  }

  // Δλ² = −Δf² + 2iΔfΔκ + Δκ² − 4J·e^{iφ}, as [re, im].
  function deltaLambdaSq(df, dk, phi, J = 1) {
    return [-(df * df) + dk * dk - 4 * J * Math.cos(phi), 2 * df * dk - 4 * J * Math.sin(phi)];
  }

  /* The n×n maps behind the main figure, row-major with rows along y (Δf):
   * K (Petermann factor), instability, minPetermann, disc and q (q̃). */
  function fields(phi, kc, n = 500) {
    const x = linspace(-4, 4, n), N = n * n;
    const K = new Float64Array(N), instability = new Float64Array(N), minPetermann = new Float64Array(N),
      disc = new Float64Array(N), q = new Float64Array(N);
    const c = Math.cos(phi), s = Math.sin(phi), cot = 1 / Math.tan(phi / 2);
    for (let j = 0; j < n; j++) {
      const Y = x[j];
      for (let i = 0; i < n; i++) {
        const X = x[i], k = j * n + i;
        const re = -(Y * Y) + X * X - 4 * c, im = 2 * Y * X - 4 * s;
        const mod2 = Math.hypot(re, im);                 // |Δλ|² = |Δλ²|
        K[k] = (Y * Y + X * X + mod2 + 4) / (2 * mod2);
        const kb = kc - X, qq = kb * im / 8, p = (kb * kb + re) / 4;
        q[k] = qq;
        disc[k] = -4 * p * p * p - 27 * qq * qq;
        instability[k] = csqrt(re, im)[0] - kb;
        minPetermann[k] = phi === 0 || phi === 2 * Math.PI ? X : phi === Math.PI ? Y : -cot * X - Y;
      }
    }
    return { x, K, instability, minPetermann, disc, q };
  }

  // ---- degeneracy locations (utils/tpd_locations_nd.py) ----
  const deg = (type, dk, df) => ({ type, dk, df });

  function epLocation(phi) {
    const a = 2 * Math.cos(phi / 2), b = 2 * Math.sin(phi / 2);
    return phi < Math.PI
      ? [deg('PRIMARY_EP', -a, -b), deg('SECONDARY_EP', a, b)]
      : [deg('PRIMARY_EP', a, b), deg('SECONDARY_EP', -a, -b)];
  }

  // Real roots of a real polynomial (coefficients highest power first), via
  // Durand–Kerner on the complex plane; numpy.roots keeps |Im| < 1e-10 as real.
  function realRoots(coeffs) {
    const lead = coeffs[0], c = coeffs.map(v => v / lead), deg = c.length - 1;
    let zr = [], zi = [];
    for (let k = 0; k < deg; k++) { const a = 2 * Math.PI * k / deg + 0.4; zr.push(1.8 * Math.cos(a)); zi.push(1.8 * Math.sin(a)); }
    const evalP = (r, i) => { let pr = 1, pi = 0; for (let k = 1; k <= deg; k++) { const t = pr * r - pi * i + c[k]; pi = pr * i + pi * r; pr = t; } return [pr, pi]; };
    for (let it = 0; it < 500; it++) {
      let moved = 0;
      for (let k = 0; k < deg; k++) {
        let [nr, ni] = evalP(zr[k], zi[k]), dr = 1, di = 0;
        for (let m = 0; m < deg; m++) if (m !== k) {
          const ar = zr[k] - zr[m], ai = zi[k] - zi[m], t = dr * ar - di * ai; di = dr * ai + di * ar; dr = t;
        }
        const den = dr * dr + di * di, wr = (nr * dr + ni * di) / den, wi = (ni * dr - nr * di) / den;
        zr[k] -= wr; zi[k] -= wi; moved = Math.max(moved, Math.hypot(wr, wi));
      }
      if (moved < 1e-15) break;
    }
    const out = [];
    for (let k = 0; k < deg; k++) if (Math.abs(zi[k]) < 1e-10) {
      let r = zr[k];                                    // polish on the real line
      for (let it = 0; it < 4; it++) {
        let p = 1, dp = 0; for (let m = 1; m <= deg; m++) { dp = dp * r + p; p = p * r + c[m]; }
        if (dp === 0) break; r -= p / dp;
      }
      out.push(r);
    }
    return out;
  }

  function tpdLocation(phi, kc) {
    const L = [];
    if (phi === 0) {
      if (kc * kc <= 8) {
        const r = Math.sqrt(8 - kc * kc);
        L.push(deg('PRIMARY_TPD', (kc - r) / 2, 0), deg('SECONDARY_TPD', (kc + r) / 2, 0));
      }
      if (kc >= 2) {
        const r = Math.sqrt(kc * kc - 4);
        L.push(deg('ROGUE_TPD', 0, r), deg('ROGUE_TPD', 0, -r), deg('ROGUE_TPD', kc, r), deg('ROGUE_TPD', kc, -r));
      }
    } else if (phi === Math.PI) {
      const r = Math.sqrt(kc * kc + 4);
      L.push(deg('PRIMARY_TPD', 0, r), deg('SECONDARY_TPD', 0, -r), deg('ROGUE_TPD', kc, r), deg('ROGUE_TPD', kc, -r));
    } else {
      const c = Math.cos(phi), s = Math.sin(phi);
      if (kc * kc >= 4 * c) {
        const r = Math.sqrt(kc * kc - 4 * c);
        L.push(deg('ROGUE_TPD', kc, r), deg('ROGUE_TPD', kc, -r));
      }
      for (const x of realRoots([2, -2 * kc, kc * kc - 4 * c, 0, 4 * c * c - 4])) {
        const df = -1 * (-kc * kc * x + 2 * kc * x * x - 2 * x * x * x + 4 * c * x) / (2 * s);
        L.push(deg(x < 0 ? 'PRIMARY_TPD' : 'SECONDARY_TPD', x, df));
      }
    }
    return L;
  }

  // ---- transmission peaks and eigenvalues (utils/pt_peaks_MODEL.py) ----
  function peakLocation(J, fc, kc, df, dk, phi) {
    const offset = fc - df / 2, kb = kc - dk;
    const p = -((df / 2) ** 2) + (dk / 2) ** 2 - Math.cos(phi) * J ** 2 + (kb / 2) ** 2;
    const q = (kb / 4) * (df * dk - 2 * J ** 2 * Math.sin(phi));
    const disc = -4 * p ** 3 - 27 * q ** 2, eps = 1e-20;
    if (disc > eps) {
      const theta = Math.PI / 3 + (1 / 3) * Math.acos((3 * q / (2 * p)) * Math.sqrt(-3 / p));
      const fac = 2 * Math.sqrt(-p / 3);
      return [fac * Math.cos(theta + Math.PI / 3) + offset, fac * Math.cos(theta - Math.PI / 3) + offset];
    }
    if (disc < -eps) {
      if (p < 0) {
        const theta = (1 / 3) * Math.acosh((-3 * Math.abs(q) / (2 * p)) * Math.sqrt(-3 / p));
        return [-2 * (Math.abs(q) / q) * Math.sqrt(-p / 3) * Math.cosh(theta) + offset];
      }
      const theta = (1 / 3) * Math.asinh((3 * q / (2 * p)) * Math.sqrt(3 / p));
      return [-2 * Math.sqrt(p / 3) * Math.sinh(theta) + offset];
    }
    return [2 * Math.cbrt(-q / 2) + offset];
  }

  // Imaginary parts of λ± = λ0 ± Δλ/2, highest first (the Python sort order).
  function eigenImag(J, fc, kc, df, dk, phi) {
    const [re, im] = deltaLambdaSq(df, dk, phi, J), d = csqrt(re, im)[1] / 2, i0 = df / 2 - fc;
    return [Math.max(i0 + d, i0 - d), Math.min(i0 + d, i0 - d)];
  }

  /* The second figure: transmission peaks ν± and |Im λ±| along a cut through the
   * primary EP, with the primary TPD, EP and the nearest instability crossing. */
  function splitting(phi, kc, n = 2000) {
    const J = 1, fc = 0;
    const tpd = tpdLocation(phi, kc).find(d => d.type === 'PRIMARY_TPD') || null;
    const ep = epLocation(phi)[0];
    const alongF = phi === Math.PI;
    const x = alongF ? linspace(ep.df - 1, ep.df + 2.5, n)
      : phi === 0 ? linspace(ep.dk - 1, ep.dk + 2.5, n) : linspace(ep.dk - 0.5, -0.18, n);
    const nan = () => new Float64Array(n).fill(NaN);
    const nuPlus = nan(), nuMinus = nan(), nu0 = nan(), lamPlus = nan(), lamMinus = nan();
    const crossings = [];
    let prev = null;
    for (let i = 0; i < n; i++) {
      const df = alongF ? x[i] : (2 * Math.sin(phi)) / x[i], dk = alongF ? 0 : x[i];
      const peaks = peakLocation(J, fc, kc, df, dk, phi);
      if (peaks.length === 2) { nuPlus[i] = peaks[0]; nuMinus[i] = peaks[1]; } else nu0[i] = peaks[0];
      const [a, b] = eigenImag(J, fc, kc, df, dk, phi);
      lamPlus[i] = -a; lamMinus[i] = -b;
      const [re, im] = deltaLambdaSq(df, dk, phi, J);
      const cur = csqrt(re, im)[0] - (kc - dk);
      if (prev !== null && Math.sign(cur) !== Math.sign(prev)) crossings.push(alongF ? df : dk);
      prev = cur;
    }
    const target = alongF ? ep.df : ep.dk;
    let instability = null;
    for (const v of crossings) if (instability === null || Math.abs(v - target) < Math.abs(instability - target)) instability = v;
    let ymin = Infinity, ymax = -Infinity;
    for (const arr of [lamPlus, lamMinus, nuPlus, nuMinus, nu0]) for (const v of arr) {
      if (v < ymin) ymin = v; if (v > ymax) ymax = v;
    }
    return { x, axis: alongF ? 'df' : 'dk', nuPlus, nuMinus, nu0, lamPlus, lamMinus,
      tpd: tpd ? (alongF ? tpd.df : tpd.dk) : null, ep: target, instability, ymin, ymax };
  }

  const api = { PHASES, linspace, csqrt, fields, epLocation, tpdLocation, realRoots, peakLocation,
    eigenImag, splitting };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.EpEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
