/* resonator-engine.js: the models, fitter and formula parser behind resonator.html. No dependencies.
 * Loads as a classic <script> (window.ResonatorEngine) or from Node (require), so the same file is what
 * the page runs and what tests/resonator-engine.test.js checks.
 *
 * Models (Probst et al., Rev. Sci. Instrum. 86, 024706 (2015), arXiv:1410.3365; Khalil et al.,
 * J. Appl. Phys. 111, 054510 (2012), arXiv:1108.3117). With x = f/fr − 1 and the environment
 * env(f) = a·e^{iα}·e^{−2πi f τ}:
 *   hanger (notch)   S21 = env · [1 − (Ql/|Qc|) e^{iφ} / (1 + 2i Ql x)]
 *   reflection       S11 = env · [1 − (2Ql/|Qc|) e^{iφ} / (1 + 2i Ql x)]
 *   transmission     S21 = env · (Ql/|Qc|) / (1 + 2i Ql x)            (two equal ports, Qc = both together)
 *   1/Ql = 1/Qi + cos φ / |Qc|   (the "diameter correction": only Re(1/Qc) is a loss channel)
 * Units: frequency in GHz, τ in ns, so 2π f τ is in radians.
 *
 * Fitting: Levenberg–Marquardt with central-difference Jacobians whose steps are calibrated per
 * parameter, covariance s²(JᵀJ)⁻¹ with s² = SSR/(N − p), and derived quantities propagated through
 * the full covariance.
 *
 * Formulas: a small parser for fit functions typed by students, e.g.
 *   A/(1 + (2*(f - f0)/w)^2) + C        or, complex,   1 - Ql/Qc/(1 + 2*i*Ql*(f/fr - 1))
 * Operators + − * / ^ (or **), parentheses, i (the imaginary unit), pi, the variable f and
 * functions sqrt exp log ln log10 abs re im conj arg sin cos tan atan sinh cosh tanh. Every other
 * name is a fit parameter. There is no eval: the formula is parsed into a tree and evaluated on
 * whole arrays, with complex numbers wherever they appear.
 */
(function (root) {
  'use strict';
  const TAU = 2 * Math.PI;
  const HBAR = 1.054571817e-34;   // J·s

  /* ------------------------------------------------------------------ models */

  const MODELS = {
    hanger:       { name: 'Hanger (notch)', port: 'S21', scale: 1, background: true },
    reflection:   { name: 'Reflection',     port: 'S11', scale: 2, background: true },
    transmission: { name: 'Transmission',   port: 'S21', scale: 1, background: false },
  };

  /** 1/Ql = 1/Qi + cos φ/|Qc|. */
  const loadedQ = (Qi, Qc, phi = 0) => 1 / (1 / Qi + Math.cos(phi) / Qc);
  /** Internal Q from the loaded and coupling Q: 1/Qi = 1/Ql − cos φ/|Qc|. Negative means unphysical. */
  const internalQ = (Ql, Qc, phi = 0) => 1 / (1 / Ql - Math.cos(phi) / Qc);

  /**
   * S parameter of `model` at frequencies f [GHz]. p = {fr, Ql, Qc, phi, a, alpha, tau}
   * (or Qi in place of Ql). Returns {re, im} Float64Arrays.
   */
  function sweep(model, p, f) {
    const M = MODELS[model];
    if (!M) throw new Error('unknown model ' + model);
    const Ql = p.Ql != null ? p.Ql : loadedQ(p.Qi, p.Qc, p.phi || 0);
    const phi = M.background ? (p.phi || 0) : 0;
    const a = p.a == null ? 1 : p.a, alpha = p.alpha || 0, tau = p.tau || 0;
    const k = M.scale * Ql / p.Qc, cph = Math.cos(phi), sph = Math.sin(phi);
    const n = f.length, re = new Float64Array(n), im = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const y = 2 * Ql * (f[j] / p.fr - 1);
      // k e^{iφ} / (1 + i y) = k (cosφ + i sinφ)(1 − i y) / (1 + y²)
      const d = k / (1 + y * y);
      let r = d * (cph + sph * y), i = d * (sph - cph * y);
      if (M.background) { r = 1 - r; i = -i; }
      const th = alpha - TAU * f[j] * tau, c = a * Math.cos(th), s = a * Math.sin(th);
      re[j] = r * c - i * s; im[j] = r * s + i * c;
    }
    return { re, im };
  }

  /** Coupling regime from Qi and the loss-channel part of the coupling, Qc/cos φ. */
  function regime(Qi, Qc, phi = 0) {
    const Qce = Qc / Math.cos(phi), r = Qi / Qce;
    if (!(Qi > 0)) return { key: 'unphysical', ratio: r };
    if (r > 1.25) return { key: 'over', ratio: r };
    if (r < 0.8) return { key: 'under', ratio: r };
    return { key: 'critical', ratio: r };
  }

  /**
   * Mean photon number in the resonator on resonance for power P [dBm] arriving at the device port.
   * Hanger and two-port transmission: n = 2 Ql² P / (|Qc| ħ ωr²); reflection: 4 Ql² P / (|Qc| ħ ωr²).
   */
  function photonNumber(model, PdBm, frGHz, Ql, Qc) {
    const P = 1e-3 * Math.pow(10, PdBm / 10), w = TAU * frGHz * 1e9;
    return (model === 'reflection' ? 4 : 2) * Ql * Ql * P / (Qc * HBAR * w * w);
  }
  /** Inverse of photonNumber: the power [dBm] at the device that gives n photons. */
  function powerForPhotons(model, n, frGHz, Ql, Qc) {
    return 10 * Math.log10(n / photonNumber(model, 0, frGHz, Ql, Qc));
  }

  /* ------------------------------------------------------------- linear algebra */

  /** Solve A x = b for a small dense symmetric system (Gaussian elimination, partial pivoting). */
  function solve(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (!(Math.abs(M[piv][c]) > 1e-300)) return null;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = c + 1; r < n; r++) {
        const g = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= g * M[c][k];
      }
    }
    const x = new Array(n);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  function invert(A) {
    const n = A.length, out = A.map(() => new Array(n));
    for (let c = 0; c < n; c++) {
      const e = new Array(n).fill(0); e[c] = 1;
      const x = solve(A, e);
      if (!x) return null;
      for (let r = 0; r < n; r++) out[r][c] = x[r];
    }
    return out;
  }

  const ssq = r => { let s = 0; for (let i = 0; i < r.length; i++) s += r[i] * r[i]; return s; };

  /* ------------------------------------------------------- Levenberg–Marquardt */

  /**
   * Minimise Σ r(p)² over the free entries of p.
   *   resid(p) → array of residuals (finite numbers)
   *   opts.fixed: booleans, true keeps that entry at its starting value
   * Returns {p, cov (free-parameter order mapped back to the full vector, null if singular), ssr, dof,
   *          iters, converged, message}.
   */
  function levmar(resid, p0, opts = {}) {
    const fixed = opts.fixed || p0.map(() => false);
    const free = p0.map((_, i) => i).filter(i => !fixed[i]);
    const maxIter = opts.maxIter || 300;
    let p = p0.slice();
    let r = resid(p);
    if (!r.every(Number.isFinite)) return { p, cov: null, ssr: NaN, dof: r.length - free.length, iters: 0, converged: false,
      message: 'The function gives non-numbers (NaN or ∞) at the starting values. Change the starting values.' };
    let ssr = ssq(r);
    const m = r.length, nf = free.length;
    if (nf === 0) return { p, cov: [], ssr, dof: m, iters: 0, converged: true, message: 'Nothing to fit: every parameter is fixed.' };
    if (m <= nf) return { p, cov: null, ssr, dof: m - nf, iters: 0, converged: false, message: `Only ${m} data values for ${nf} free parameters.` };

    // Per-parameter step: start small relative to the value and adjust once until the residuals move by a
    // modest but resolvable amount. A fixed relative step fails for a resonance frequency with a Q of 10⁶.
    const scaleR = Math.sqrt(ssr / m) + Math.sqrt(ssq(opts.yScale || [])) / Math.max(1, (opts.yScale || []).length) + 1e-300;
    const h = free.map(j => {
      let hj = 1e-6 * Math.max(Math.abs(p[j]), 1e-9);
      for (let t = 0; t < 12; t++) {
        const q = p.slice(); q[j] += hj;
        const rq = resid(q);
        let d = 0; for (let i = 0; i < m; i++) d = Math.max(d, Math.abs(rq[i] - r[i]));
        const ref = Math.max(scaleR, 1e-12);
        if (!Number.isFinite(d) || d > 1e-3 * ref) hj /= 10;
        else if (d < 1e-9 * ref) hj *= 10;
        else break;
      }
      return hj;
    });

    function jac(pp) {
      const J = free.map(() => new Float64Array(m));
      free.forEach((j, k) => {
        const a = pp.slice(), b = pp.slice(); a[j] += h[k]; b[j] -= h[k];
        const ra = resid(a), rb = resid(b);
        for (let i = 0; i < m; i++) J[k][i] = (ra[i] - rb[i]) / (2 * h[k]);
      });
      return J;
    }
    function normal(J, rr) {
      const A = free.map(() => new Array(nf).fill(0)), g = new Array(nf).fill(0);
      for (let a = 0; a < nf; a++) {
        for (let b = a; b < nf; b++) { let s = 0; for (let i = 0; i < m; i++) s += J[a][i] * J[b][i]; A[a][b] = A[b][a] = s; }
        let s = 0; for (let i = 0; i < m; i++) s += J[a][i] * rr[i]; g[a] = s;
      }
      return { A, g };
    }

    let lambda = 1e-3, iters = 0, converged = false, message = '', stall = 0;
    let J = jac(p), N = normal(J, r);
    while (iters < maxIter) {
      iters++;
      const A = N.A.map((row, a) => row.map((v, b) => a === b ? v * (1 + lambda) + 1e-300 : v));
      const step = solve(A, N.g.map(v => -v));
      if (!step) { lambda *= 10; if (lambda > 1e16) { message = 'The fit is stuck: the parameters cannot be told apart by the data.'; break; } continue; }
      const q = p.slice(); free.forEach((j, k) => { q[j] += step[k]; });
      const rq = resid(q), sq = rq.every(Number.isFinite) ? ssq(rq) : Infinity;
      if (sq < ssr) {
        const drop = (ssr - sq) / Math.max(ssr, 1e-300);
        p = q; r = rq; ssr = sq; lambda = Math.max(lambda / 10, 1e-12);
        const small = free.every((j, k) => Math.abs(step[k]) <= 1e-10 * (Math.abs(p[j]) + h[k] * 1e3));
        stall = drop < 1e-9 ? stall + 1 : 0;      // three accepted steps in a row that barely help: done
        if (drop < 1e-13 || small || stall >= 3) { converged = true; break; }
        J = jac(p); N = normal(J, r);
      } else {
        lambda *= 10;
        if (lambda > 1e16) { converged = true; break; }   // no step lowers the sum any more: a minimum
      }
    }
    if (!converged && !message) message = `Stopped after ${maxIter} steps without settling. Try better starting values.`;
    J = jac(p); N = normal(J, r);
    const dof = m - nf, s2 = ssr / Math.max(dof, 1);
    const inv = invert(N.A);
    let cov = null;
    if (inv && inv.every((row, a) => row[a] > 0 && Number.isFinite(row[a]))) {
      cov = p.map(() => p.map(() => 0));
      free.forEach((ja, a) => free.forEach((jb, b) => { cov[ja][jb] = inv[a][b] * s2; }));
    } else if (!message) message = 'Some parameters are not determined by the data (their uncertainty is infinite).';
    return { p, cov, ssr, dof, iters, converged, message };
  }

  /** Value and 1σ of g(p) through the covariance, with a central-difference gradient. */
  function propagate(g, p, cov) {
    const v = g(p);
    if (!cov) return { value: v, sigma: NaN };
    const grad = p.map((pj, j) => {
      if (!cov[j][j]) return 0;
      const hj = Math.sqrt(cov[j][j]) * 1e-3 || 1e-9 * Math.max(Math.abs(pj), 1);
      const a = p.slice(), b = p.slice(); a[j] += hj; b[j] -= hj;
      return (g(a) - g(b)) / (2 * hj);
    });
    let s = 0;
    for (let a = 0; a < p.length; a++) for (let b = 0; b < p.length; b++) s += grad[a] * cov[a][b] * grad[b];
    return { value: v, sigma: Math.sqrt(Math.max(s, 0)) };
  }

  /* ----------------------------------------------- resonator guesses and fits */

  function linfit(x, y) {
    const n = x.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i]; }
    const d = n * sxx - sx * sx;
    const slope = d ? (n * sxy - sx * sy) / d : 0;
    return { slope, icpt: (sy - slope * sx) / n };
  }

  function unwrap(ph) {
    const out = Float64Array.from(ph);
    for (let i = 1; i < out.length; i++) {
      let d = out[i] - out[i - 1];
      d -= TAU * Math.round(d / TAU);
      out[i] = out[i - 1] + d;
    }
    return out;
  }

  /** Least-squares circle through points (Kåsa's algebraic fit): {xc, yc, r}. */
  function circleFit(re, im) {
    const n = re.length; let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += re[i]; my += im[i]; }
    mx /= n; my /= n;
    let suu = 0, svv = 0, suv = 0, suuu = 0, svvv = 0, suvv = 0, svuu = 0;
    for (let i = 0; i < n; i++) {
      const u = re[i] - mx, v = im[i] - my;
      suu += u * u; svv += v * v; suv += u * v; suuu += u * u * u; svvv += v * v * v; suvv += u * v * v; svuu += v * u * u;
    }
    const s = solve([[suu, suv], [suv, svv]], [(suuu + suvv) / 2, (svvv + svuu) / 2]);
    if (!s) return null;
    return { xc: s[0] + mx, yc: s[1] + my, r: Math.sqrt(s[0] * s[0] + s[1] * s[1] + (suu + svv) / n) };
  }

  /** How far points are from their best circle once a delay τ is taken out (RMS, in the data's units). */
  function circleResid(f, re, im, tau) {
    const n = f.length, x = new Float64Array(n), y = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const t = TAU * f[j] * tau, c = Math.cos(t), s = Math.sin(t);
      x[j] = re[j] * c - im[j] * s; y[j] = re[j] * s + im[j] * c;
    }
    const C = circleFit(x, y);
    if (!C) return Infinity;
    let e = 0;
    for (let j = 0; j < n; j++) { const d = Math.hypot(x[j] - C.xc, y[j] - C.yc) - C.r; e += d * d; }
    return Math.sqrt(e / n);
  }

  /**
   * Starting values for a resonator fit from data f [GHz], re, im. Returns
   * {fr, Ql, Qc, phi, a, alpha, tau}. Works from the data alone: the cable delay from the phase slope
   * at both edges, the background from the edges, the centre and width from the resonant response.
   */
  function guess(model, f, re, im, opts = {}) {
    const M = MODELS[model], n = f.length;
    const edge = Math.max(3, Math.floor(n * 0.1));
    const ph = unwrap(Array.from(re, (x, i) => Math.atan2(im[i], x)));
    let tau = 0;
    if (!opts.noDelay) {
      // first guess from the phase slope at both edges, then the delay that puts the data best on a circle
      const L = linfit(f.slice(0, edge), ph.slice(0, edge)), R = linfit(f.slice(n - edge), ph.slice(n - edge));
      const t0 = -(L.slope + R.slope) / 2 / TAU;
      const span = Math.abs(f[n - 1] - f[0]) || 1, cost = t => circleResid(f, re, im, t);
      // the edge slope also picks up the resonance's own tail on a narrow span, so search near it and near zero
      let bt = t0, bc = cost(t0);
      for (const c0 of [t0, 0]) for (let k = -60; k <= 60; k++) {
        const t = c0 + k / (60 * span), c = cost(t);
        if (c < bc) { bc = c; bt = t; }
      }
      let lo = bt - 1 / (60 * span), hi = bt + 1 / (60 * span);
      for (let k = 0; k < 40; k++) {
        const m1 = lo + (hi - lo) * 0.382, m2 = lo + (hi - lo) * 0.618;
        if (cost(m1) < cost(m2)) hi = m2; else lo = m1;
      }
      tau = (lo + hi) / 2;
    }
    const zr = new Float64Array(n), zi = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const t = TAU * f[j] * tau, c = Math.cos(t), s = Math.sin(t);
      zr[j] = re[j] * c - im[j] * s; zi[j] = re[j] * s + im[j] * c;
    }
    let a = 1, alpha = 0;
    if (M.background) {
      let sr = 0, si = 0;
      for (let j = 0; j < n; j++) if (j < edge || j >= n - edge) { sr += zr[j]; si += zi[j]; }
      a = Math.hypot(sr, si) / (2 * edge); alpha = Math.atan2(si, sr);
    }
    // the resonant part d, normalised by the background
    const ca = Math.cos(-alpha) / a, sa = Math.sin(-alpha) / a;
    const dr = new Float64Array(n), di = new Float64Array(n), mag2 = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let r = zr[j] * ca - zi[j] * sa, i = zr[j] * sa + zi[j] * ca;
      if (M.background) { r = 1 - r; i = -i; }
      dr[j] = r; di[j] = i; mag2[j] = r * r + i * i;
    }
    // smoothed peak of |d|², so one noisy point does not win
    let best = 0, bi = 0;
    for (let j = 1; j < n - 1; j++) { const v = mag2[j - 1] + 2 * mag2[j] + mag2[j + 1]; if (v > best) { best = v; bi = j; } }
    const peak = mag2[bi];
    if (!M.background) { a = 1; alpha = Math.atan2(di[bi], dr[bi]); }
    const half = peak / 2, cross = (j0, step) => {
      let j = j0;
      while (j + step >= 0 && j + step < n && mag2[j + step] > half) j += step;
      const k = j + step;
      if (k < 0 || k >= n) return f[j];
      return f[j] + (f[k] - f[j]) * (mag2[j] - half) / (mag2[j] - mag2[k] || 1);
    };
    const fr = f[bi], df = Math.abs(f[n - 1] - f[0]) / Math.max(n - 1, 1);
    const fwhm = Math.max(Math.abs(cross(bi, 1) - cross(bi, -1)), df);
    const Ql = fr / fwhm, depth = Math.sqrt(peak);
    let Qc = M.scale * Ql / Math.max(depth, 1e-6), phi = M.background ? Math.atan2(di[bi], dr[bi]) : 0;
    if (M.background) {
      if (Math.abs(phi) > 1.2) phi = Math.sign(phi) * 1.2;
      if (1 / Ql - Math.cos(phi) / Qc <= 0) Qc = 1.05 * Ql * Math.cos(phi);   // keep Qi positive
    }
    return { fr, Ql, Qc, phi, a, alpha, tau };
  }

  const KEYS = ['fr', 'Ql', 'Qc', 'phi', 'a', 'alpha', 'tau'];
  // fitted internally as logs for the Q's and a, so they stay positive and the steps are well scaled
  const toVec = p => [p.fr, Math.log(p.Ql), Math.log(p.Qc), p.phi, Math.log(p.a), p.alpha, p.tau];
  const fromVec = v => ({ fr: v[0], Ql: Math.exp(v[1]), Qc: Math.exp(v[2]), phi: v[3], a: Math.exp(v[4]), alpha: v[5], tau: v[6] });

  /**
   * Fit a built-in resonator model.
   *   data: {f, re, im} (complex) or {f, mag} (linear |S|, no phase)
   *   opts: {noDelay: τ = 0, fixA: keep a = 1 (transmission always does), start: starting values}
   * Returns {model, params (best values), sigma, derived: {Qi, Qc, Ql, fr, ...} each {value, sigma},
   *          curve(f) → {re, im}, rms, converged, message, notes[]}.
   */
  function fitResonator(model, data, opts = {}) {
    const M = MODELS[model], f = data.f, complex = !!data.re;
    const notes = [];
    let start;
    if (opts.start) start = Object.assign({}, opts.start);
    else if (complex) start = guess(model, f, data.re, data.im, opts);
    else start = guess(model, f, data.mag, new Float64Array(f.length), { noDelay: true });
    if (!M.background) { start.a = 1; start.phi = 0; }
    if (!complex) { start.alpha = 0; start.tau = 0; }
    if (opts.noDelay) start.tau = 0;

    const m = f.length;
    const resid = v => {
      const s = sweep(model, fromVec(v), f), out = new Float64Array(complex ? 2 * m : m);
      if (complex) for (let j = 0; j < m; j++) { out[j] = s.re[j] - data.re[j]; out[m + j] = s.im[j] - data.im[j]; }
      else for (let j = 0; j < m; j++) out[j] = Math.hypot(s.re[j], s.im[j]) - data.mag[j];
      return Array.from(out);
    };
    const fixedBase = KEYS.map(k =>
      (k === 'tau' && (opts.noDelay || !complex)) || (k === 'alpha' && !complex) ||
      (k === 'a' && (!M.background || opts.fixA)) || (k === 'phi' && !M.background));
    // Stage 1: the resonance only, background held; stage 2: everything free.
    const run = st => {
      const s1 = levmar(resid, toVec(st), { fixed: KEYS.map((k, i) => fixedBase[i] || k === 'a' || k === 'alpha' || k === 'tau') });
      const s2 = levmar(resid, s1.p, { fixed: fixedBase, maxIter: 2000 });
      s2.iters += s1.iters;
      return s2;
    };
    // The noise level from point-to-point differences, to judge whether a fit is as good as the data allow.
    let dn = 0;
    for (let j = 1; j < m; j++) dn += complex ? (data.re[j] - data.re[j - 1]) ** 2 + (data.im[j] - data.im[j - 1]) ** 2 : (data.mag[j] - data.mag[j - 1]) ** 2;
    const noise = Math.sqrt(dn / ((m - 1) * (complex ? 4 : 2)));
    let fit = run(start);
    // A weakly coupled resonator makes a small circle, and the delay can lock in whole turns off. Try those too.
    if (complex && !opts.noDelay && !opts.start && (!fit.converged || Math.sqrt(fit.ssr / (fit.dof + 7)) > 1.3 * noise)) {
      const span = Math.abs(f[m - 1] - f[0]) || 1;
      // lowest sum of squares wins; a settled fit beats an unsettled one unless that is clearly better
      const better = (x, y) => x.ssr < y.ssr * (x.converged || !y.converged ? 1 : 0.9);
      for (const t of [1, -1, 2, -2, 0.5, -0.5].map(dt => start.tau + dt / span).concat(0)) {
        const alt = run(Object.assign({}, start, { tau: t }));
        if (better(alt, fit)) fit = alt;
      }
    }
    const best = fromVec(fit.p);
    const Qi = v => { const q = fromVec(v); return internalQ(q.Ql, q.Qc, M.background ? q.phi : 0); };
    const derived = {
      fr: propagate(v => v[0], fit.p, fit.cov),
      Ql: propagate(v => Math.exp(v[1]), fit.p, fit.cov),
      Qc: propagate(v => Math.exp(v[2]), fit.p, fit.cov),
      Qi: propagate(Qi, fit.p, fit.cov),
      phi: propagate(v => v[3], fit.p, fit.cov),
      a: propagate(v => Math.exp(v[4]), fit.p, fit.cov),
      alpha: propagate(v => v[5], fit.p, fit.cov),
      tau: propagate(v => v[6], fit.p, fit.cov),
      kappa: propagate(v => v[0] / Math.exp(v[1]), fit.p, fit.cov),            // κ/2π in GHz
      kappaI: propagate(v => v[0] / Qi(v), fit.p, fit.cov),
      kappaC: propagate(v => v[0] * Math.cos(M.background ? v[3] : 0) / Math.exp(v[2]), fit.p, fit.cov),
    };
    if (!(derived.Qi.value > 0)) notes.push('The fit gives a negative internal Q, which is unphysical: the dip is deeper than the coupling allows. Check the background level (a) or try another model.');
    if (!M.background) notes.push('In transmission the depth of the peak sets Ql/Qc only if |S21| is calibrated, so a is held at 1. Normalise your data to the line\'s through-transmission first, or treat Qi and Qc as relative.');
    if (!complex && model === 'reflection') notes.push('From |S11| alone, an over-coupled and an under-coupled resonator with Qi and Qc swapped look the same. The phase tells them apart: over-coupled reflection winds the full circle around the origin.');
    let ss = 0;
    const r = resid(fit.p); for (let i = 0; i < r.length; i++) ss += r[i] * r[i];
    return {
      model, complex, params: best, start, derived, cov: fit.cov,
      rms: Math.sqrt(ss / r.length), noise, converged: fit.converged, message: fit.message, iters: fit.iters, notes,
      curve: ff => sweep(model, best, ff),
    };
  }

  /* --------------------------------------------------------- formula parser */

  const FUNCS = {
    sqrt: 1, exp: 1, log: 1, ln: 1, log10: 1, abs: 1, re: 1, real: 1, im: 1, imag: 1, conj: 1, arg: 1, angle: 1,
    sin: 1, cos: 1, tan: 1, atan: 1, sinh: 1, cosh: 1, tanh: 1,
  };
  const RESERVED = new Set(['f', 'i', 'pi', ...Object.keys(FUNCS)]);

  class FormulaError extends Error { constructor(msg, pos) { super(msg); this.pos = pos; } }

  function tokenize(src) {
    const out = []; let i = 0;
    while (i < src.length) {
      const c = src[i];
      if (/\s/.test(c)) { i++; continue; }
      const num = /^(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/.exec(src.slice(i));
      if (num) { out.push({ t: 'num', v: parseFloat(num[0]), pos: i }); i += num[0].length; continue; }
      const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
      if (id) { out.push({ t: 'id', v: id[0], pos: i }); i += id[0].length; continue; }
      if (src.startsWith('**', i)) { out.push({ t: 'op', v: '^', pos: i }); i += 2; continue; }
      if ('+-*/^(),'.includes(c)) { out.push({ t: 'op', v: c, pos: i }); i++; continue; }
      if (c === '·' || c === '×') { out.push({ t: 'op', v: '*', pos: i }); i++; continue; }
      if (c === '−') { out.push({ t: 'op', v: '-', pos: i }); i++; continue; }
      throw new FormulaError(`"${c}" is not allowed in a formula`, i);
    }
    out.push({ t: 'end', pos: src.length });
    return out;
  }

  /** Parse a formula. Returns {ast, params (names in order of first use), src}. */
  function parse(src) {
    const tk = tokenize(String(src)); let k = 0;
    const peek = () => tk[k], next = () => tk[k++];
    const isOp = v => peek().t === 'op' && peek().v === v;
    const params = [];
    function expect(v) {
      if (!isOp(v)) throw new FormulaError(`expected "${v}"`, peek().pos);
      next();
    }
    function expr() {
      let node = term();
      while (isOp('+') || isOp('-')) { const op = next().v; node = { k: 'bin', op, a: node, b: term() }; }
      return node;
    }
    function term() {
      let node = unary();
      while (isOp('*') || isOp('/')) { const op = next().v; node = { k: 'bin', op, a: node, b: unary() }; }
      if (peek().t === 'num' || peek().t === 'id' || isOp('('))
        throw new FormulaError('two terms side by side: put "*" between them', peek().pos);
      return node;
    }
    function unary() {
      if (isOp('-')) { next(); return { k: 'neg', a: unary() }; }
      if (isOp('+')) { next(); return unary(); }
      return power();
    }
    function power() {
      const base = atom();
      if (isOp('^')) { next(); return { k: 'bin', op: '^', a: base, b: unary() }; }
      return base;
    }
    function atom() {
      const t = next();
      if (t.t === 'num') return { k: 'num', v: t.v };
      if (t.t === 'op' && t.v === '(') { const e = expr(); expect(')'); return e; }
      if (t.t === 'id') {
        if (FUNCS[t.v]) {
          if (!isOp('(')) throw new FormulaError(`${t.v} needs brackets, as in ${t.v}(x)`, t.pos);
          next(); const a = expr(); expect(')');
          return { k: 'fn', name: t.v, a };
        }
        if (isOp('(')) throw new FormulaError(`"${t.v}" is not a known function`, t.pos);
        if (t.v === 'f') return { k: 'var' };
        if (t.v === 'i') return { k: 'imag' };
        if (t.v === 'pi') return { k: 'num', v: Math.PI };
        if (!params.includes(t.v)) params.push(t.v);
        return { k: 'par', name: t.v };
      }
      if (t.t === 'end') throw new FormulaError('the formula ends too early', t.pos);
      throw new FormulaError(`unexpected "${t.v}"`, t.pos);
    }
    const ast = expr();
    if (peek().t !== 'end') throw new FormulaError(isOp(')') ? 'there is a ")" without its "("' : `unexpected "${peek().v}"`, peek().pos);
    return { ast, params, src: String(src) };
  }

  /* Evaluation on whole arrays. A value is {re: Float64Array, im: Float64Array | null}; null = real. */
  function evaluate(ast, f, P) {
    const n = f.length;
    const fill = v => new Float64Array(n).fill(v);
    function ev(node) {
      switch (node.k) {
        case 'num': return { re: fill(node.v), im: null };
        case 'var': return { re: Float64Array.from(f), im: null };
        case 'imag': return { re: fill(0), im: fill(1) };
        case 'par': {
          const v = P[node.name];
          if (!Number.isFinite(v)) throw new FormulaError(`parameter ${node.name} has no value`, 0);
          return { re: fill(v), im: null };
        }
        case 'neg': { const a = ev(node.a); return { re: a.re.map(x => -x), im: a.im && a.im.map(x => -x) }; }
        case 'fn': return fn(node.name, ev(node.a));
        case 'bin': return bin(node.op, ev(node.a), ev(node.b));
      }
      throw new FormulaError('bad formula', 0);
    }
    function bin(op, a, b) {
      const re = new Float64Array(n);
      if (!a.im && !b.im && op !== '^') {
        for (let j = 0; j < n; j++) {
          const x = a.re[j], y = b.re[j];
          re[j] = op === '+' ? x + y : op === '-' ? x - y : op === '*' ? x * y : x / y;
        }
        return { re, im: null };
      }
      if (op === '^' && !a.im && !b.im) {
        let ok = true;
        for (let j = 0; j < n; j++) {
          const x = a.re[j], y = b.re[j];
          if (x < 0 && y !== Math.round(y)) { ok = false; break; }
          re[j] = Math.pow(x, y);
        }
        if (ok) return { re, im: null };
      }
      const ar = a.re, ai = a.im || new Float64Array(n), br = b.re, bi = b.im || new Float64Array(n);
      const im = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        const x = ar[j], y = ai[j], u = br[j], v = bi[j];
        if (op === '+') { re[j] = x + u; im[j] = y + v; }
        else if (op === '-') { re[j] = x - u; im[j] = y - v; }
        else if (op === '*') { re[j] = x * u - y * v; im[j] = x * v + y * u; }
        else if (op === '/') { const d = u * u + v * v; re[j] = (x * u + y * v) / d; im[j] = (y * u - x * v) / d; }
        else {   // z^w
          if (v === 0 && u === Math.round(u) && Math.abs(u) <= 64) {
            let pr = 1, pi = 0, br2 = x, bi2 = y, e = Math.abs(u);
            while (e) { if (e & 1) { const t = pr * br2 - pi * bi2; pi = pr * bi2 + pi * br2; pr = t; } const t = br2 * br2 - bi2 * bi2; bi2 = 2 * br2 * bi2; br2 = t; e >>= 1; }
            if (u < 0) { const d = pr * pr + pi * pi; pr = pr / d; pi = -pi / d; }
            re[j] = pr; im[j] = pi;
          } else if (x === 0 && y === 0) { re[j] = u > 0 ? 0 : NaN; im[j] = 0; }
          else {
            const lr = Math.log(Math.hypot(x, y)), li = Math.atan2(y, x);
            const er = u * lr - v * li, ei = u * li + v * lr, m = Math.exp(er);
            re[j] = m * Math.cos(ei); im[j] = m * Math.sin(ei);
          }
        }
      }
      return { re, im };
    }
    function fn(name, a) {
      const re = new Float64Array(n);
      const real = g => { for (let j = 0; j < n; j++) re[j] = g(a.re[j]); return { re, im: null }; };
      if (name === 're' || name === 'real') return { re: Float64Array.from(a.re), im: null };
      if (name === 'im' || name === 'imag') return { re: a.im ? Float64Array.from(a.im) : fill(0), im: null };
      if (name === 'abs') { for (let j = 0; j < n; j++) re[j] = a.im ? Math.hypot(a.re[j], a.im[j]) : Math.abs(a.re[j]); return { re, im: null }; }
      if (name === 'arg' || name === 'angle') { for (let j = 0; j < n; j++) re[j] = Math.atan2(a.im ? a.im[j] : 0, a.re[j]); return { re, im: null }; }
      if (name === 'conj') return { re: Float64Array.from(a.re), im: a.im && a.im.map(x => -x) };
      if (!a.im) {
        const neg = a.re.some(x => x < 0);
        if (name === 'sqrt' && !neg) return real(Math.sqrt);
        if ((name === 'log' || name === 'ln') && !neg) return real(Math.log);
        if (name === 'log10' && !neg) return real(Math.log10);
        const g = { exp: Math.exp, sin: Math.sin, cos: Math.cos, tan: Math.tan, atan: Math.atan, sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh }[name];
        if (g) return real(g);
      }
      const im = new Float64Array(n), ai = a.im || new Float64Array(n);
      for (let j = 0; j < n; j++) {
        const x = a.re[j], y = ai[j];
        let r, i;
        switch (name) {
          case 'sqrt': { const m = Math.sqrt(Math.hypot(x, y)), t = Math.atan2(y, x) / 2; r = m * Math.cos(t); i = m * Math.sin(t); break; }
          case 'exp': { const m = Math.exp(x); r = m * Math.cos(y); i = m * Math.sin(y); break; }
          case 'log': case 'ln': r = Math.log(Math.hypot(x, y)); i = Math.atan2(y, x); break;
          case 'log10': r = Math.log(Math.hypot(x, y)) / Math.LN10; i = Math.atan2(y, x) / Math.LN10; break;
          case 'sin': r = Math.sin(x) * Math.cosh(y); i = Math.cos(x) * Math.sinh(y); break;
          case 'cos': r = Math.cos(x) * Math.cosh(y); i = -Math.sin(x) * Math.sinh(y); break;
          case 'sinh': r = Math.sinh(x) * Math.cos(y); i = Math.cosh(x) * Math.sin(y); break;
          case 'cosh': r = Math.cosh(x) * Math.cos(y); i = Math.sinh(x) * Math.sin(y); break;
          case 'tan': case 'tanh': case 'atan':
            throw new FormulaError(`${name} of a complex number is not supported; write it with exp`, 0);
        }
        re[j] = r; im[j] = i;
      }
      return { re, im };
    }
    return ev(ast);
  }

  /** The formula as numpy code (for the "Python" export). */
  function toPython(ast) {
    const PY = { sqrt: 'np.emath.sqrt', exp: 'np.exp', log: 'np.emath.log', ln: 'np.emath.log', log10: 'np.emath.log10', abs: 'np.abs', re: 'np.real',
      real: 'np.real', im: 'np.imag', imag: 'np.imag', conj: 'np.conj', arg: 'np.angle', angle: 'np.angle', sin: 'np.sin',
      cos: 'np.cos', tan: 'np.tan', atan: 'np.arctan', sinh: 'np.sinh', cosh: 'np.cosh', tanh: 'np.tanh' };
    const PREC = { '+': 1, '-': 1, '*': 2, '/': 2, '^': 4 };
    function py(node, ctx = 0) {
      switch (node.k) {
        case 'num': return node.v === Math.PI ? 'np.pi' : String(node.v);
        case 'var': return 'f';
        case 'imag': return '1j';
        case 'par': return node.name;
        case 'neg': { const s = '-' + py(node.a, 3); return ctx > 3 ? `(${s})` : s; }
        case 'fn': return `${PY[node.name]}(${py(node.a)})`;
        case 'bin': {
          const p = PREC[node.op], op = node.op === '^' ? '**' : node.op;
          const left = py(node.a, node.op === '^' ? p + 1 : p);
          const right = py(node.b, node.op === '^' ? 3 : p + 1);
          const s = `${left} ${op} ${right}`;
          return p < ctx ? `(${s})` : s;
        }
      }
      return '';
    }
    return py(ast);
  }

  /* ---------------------------------------------------- fitting a typed formula */

  const TARGETS = {
    complex: 'complex S (real and imaginary parts)',
    mag: '|S| (linear)',
    dB: '|S| in dB',
    pow: '|S|² (power)',
    phase: 'phase (radians)',
    y: 'a real number (any y data)',
  };

  /** What a formula's value is compared with for a target: complex values stay complex, others go real. */
  function project(target, v) {
    const n = v.re.length;
    if (target === 'complex') return v;
    if (!v.im || target === 'y') {
      if (v.im && v.im.some((x, j) => Math.abs(x) > 1e-9 * (Math.abs(v.re[j]) + 1e-300))) return { re: Float64Array.from(v.re, (x, j) => Math.hypot(x, v.im[j])), im: null, tookAbs: true };
      return { re: v.re, im: null };
    }
    const out = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      const m2 = v.re[j] * v.re[j] + v.im[j] * v.im[j];
      out[j] = target === 'mag' ? Math.sqrt(m2) : target === 'dB' ? 10 * Math.log10(m2) : target === 'pow' ? m2 : Math.atan2(v.im[j], v.re[j]);
    }
    return { re: out, im: null, fromComplex: true };
  }

  /**
   * Fit a typed formula.
   *   formula: string, or a parse() result
   *   data: {f, y} for real targets, {f, re, im} for 'complex'
   *   start: {name: value}; fixed: {name: true}; derived: ["Qi = 1/(1/Ql - 1/Qc)", ...]
   * Returns {params, sigma, derived [{name, value, sigma, error}], rms, curve(f) → projected values,
   *          converged, message, notes, python}.
   */
  function fitFormula(formula, target, data, start = {}, fixed = {}, derived = []) {
    const F = typeof formula === 'string' ? parse(formula) : formula;
    const names = F.params, f = data.f, m = f.length, notes = [];
    const p0 = names.map(nm => (Number.isFinite(start[nm]) ? start[nm] : 1));
    const missing = names.filter(nm => !Number.isFinite(start[nm]));
    if (missing.length) notes.push(`No starting value for ${missing.join(', ')}; 1 was used.`);
    const P = v => Object.fromEntries(names.map((nm, i) => [nm, v[i]]));
    let tookAbs = false;
    const resid = v => {
      const y = project(target, evaluate(F.ast, f, P(v)));
      tookAbs = tookAbs || !!y.tookAbs;
      if (target === 'complex') {
        const out = new Array(2 * m), yi = y.im || new Float64Array(m);
        for (let j = 0; j < m; j++) { out[j] = y.re[j] - data.re[j]; out[m + j] = yi[j] - data.im[j]; }
        return out;
      }
      return Array.from(y.re, (x, j) => x - data.y[j]);
    };
    const fit = levmar(resid, p0, { fixed: names.map(nm => !!fixed[nm]) });
    if (tookAbs) notes.push('Your formula gives complex values, so its magnitude was compared with the data.');
    const sigma = Object.fromEntries(names.map((nm, i) => [nm, fit.cov ? Math.sqrt(fit.cov[i][i]) : NaN]));
    const der = derived.filter(s => s.trim()).map(line => {
      const mm = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$/.exec(line);
      if (!mm) return { name: line.trim(), error: 'write it as  name = formula' };
      try {
        const D = parse(mm[2]);
        const unknown = D.params.filter(nm => !names.includes(nm));
        if (unknown.length) return { name: mm[1], error: `${unknown.join(', ')} ${unknown.length > 1 ? 'are not fit parameters' : 'is not a fit parameter'}` };
        const one = new Float64Array([1]);
        const g = v => { const r = evaluate(D.ast, one, P(v)); return r.re[0]; };
        return Object.assign({ name: mm[1], expr: mm[2].trim() }, propagate(g, fit.p, fit.cov));
      } catch (e) { return { name: mm[1], error: e.message }; }
    });
    const r = resid(fit.p);
    let ss = 0; for (const x of r) ss += x * x;
    const best = P(fit.p);
    return {
      params: best, sigma, derived: der, cov: fit.cov, names,
      rms: Math.sqrt(ss / r.length), converged: fit.converged, message: fit.message, iters: fit.iters, notes,
      curve: ff => project(target, evaluate(F.ast, ff, best)),
      python: pythonScript(F, target, best, fixed),
    };
  }

  /** A self-contained scipy script that repeats a formula fit on the user's own computer. */
  function pythonScript(F, target, best, fixed = {}) {
    const free = F.params.filter(nm => !fixed[nm]), held = F.params.filter(nm => fixed[nm]);
    const fmt = v => Number(v.toPrecision(10)).toString();
    const proj = { complex: 'model(f, *p)', mag: 'np.abs(model(f, *p))', dB: '20 * np.log10(np.abs(model(f, *p)))',
      pow: 'np.abs(model(f, *p))**2', phase: 'np.angle(model(f, *p))', y: 'np.real(model(f, *p))' }[target];
    const lines = [
      '# Fit made with the Qublitz Resonator Lab (https://mvwf.github.io/qublitz/qublitz/resonator.html)',
      'import numpy as np',
      'from scipy.optimize import least_squares',
      '',
      '# Load your data here. f must be in the same unit as on the page.',
      target === 'complex' ? '# f, S = ...   (S complex: S = re + 1j*im)' : '# f, y = ...',
      '',
      ...held.map(nm => `${nm} = ${fmt(best[nm])}  # held fixed`),
      `def model(f${free.map(nm => ', ' + nm).join('')}):`,
      `    return ${toPython(F.ast)}${/\bf\b/.test(toPython(F.ast)) ? '' : ' + 0 * f'}`,
      '',
      'def residuals(p):',
      target === 'complex'
        ? `    d = ${proj} - S\n    return np.concatenate([d.real, d.imag])`
        : `    return ${proj} - y`,
      '',
      `p0 = [${free.map(nm => fmt(best[nm])).join(', ')}]  # ${free.join(', ')}`,
      'res = least_squares(residuals, p0, x_scale="jac")',
      'J = res.jac',
      's2 = 2 * res.cost / (J.shape[0] - J.shape[1])',
      'err = np.sqrt(np.diag(np.linalg.inv(J.T @ J)) * s2)',
      `for name, v, e in zip(${JSON.stringify(free)}, res.x, err):`,
      '    print(f"{name} = {v:.6g} ± {e:.2g}")',
    ];
    return lines.join('\n');
  }

  /* ------------------------------------------------------------ data import */

  /** Parse a text table (CSV, TSV, spaces or semicolons). Returns {header, cols: Float64Array[], skipped}. */
  function parseTable(text) {
    const lines = String(text).split(/\r?\n/).map(l => l.trim()).filter(l => l && !/^[#%!]/.test(l));
    if (!lines.length) return { header: [], cols: [], skipped: 0 };
    const delim = /\t/.test(lines[0]) ? /\t/ : /;/.test(lines[0]) ? /;/ : /,/.test(lines[0]) ? /,/ : /\s+/;
    const split = l => l.split(delim).map(s => s.trim().replace(/^"|"$/g, ''));
    let header = null, start = 0;
    const first = split(lines[0]);
    if (first.some(s => s !== '' && !Number.isFinite(Number(s)))) { header = first; start = 1; }
    const rows = []; let skipped = 0;
    for (let i = start; i < lines.length; i++) {
      const v = split(lines[i]).map(Number);
      if (v.length && v.every(Number.isFinite)) rows.push(v); else skipped++;
    }
    const nc = rows.reduce((a, r) => Math.max(a, r.length), 0);
    const good = rows.filter(r => r.length === nc);
    skipped += rows.length - good.length;
    const cols = Array.from({ length: nc }, (_, c) => Float64Array.from(good, r => r[c]));
    return { header: header && header.length === nc ? header : Array.from({ length: nc }, (_, c) => `column ${c + 1}`), cols, skipped };
  }

  /**
   * Parse a Touchstone version 1 file (.s1p, .s2p). Returns {fGHz, S: {S11: {re, im}, ...}, z0}.
   * The number of ports comes from the first data row (3 numbers for one port, 9 for two); two-port
   * data are in the order S11 S21 S12 S22, as the format defines.
   */
  function parseTouchstone(text) {
    let unit = 1e-9, fmt = 'MA', z0 = 50, per = 0;
    const nums = [];
    for (const raw of String(text).split(/\r?\n/)) {
      const line = raw.replace(/!.*$/, '').trim();
      if (!line) continue;
      if (line[0] === '#') {
        const w = line.slice(1).trim().toUpperCase().split(/\s+/);
        for (let i = 0; i < w.length; i++) {
          if (w[i] === 'HZ') unit = 1e-9; else if (w[i] === 'KHZ') unit = 1e-6; else if (w[i] === 'MHZ') unit = 1e-3; else if (w[i] === 'GHZ') unit = 1;
          else if (['MA', 'DB', 'RI'].includes(w[i])) fmt = w[i];
          else if (w[i] === 'R' && w[i + 1]) { z0 = Number(w[i + 1]); i++; }
          else if (w[i] !== 'S') throw new Error(`only S-parameter Touchstone files are supported (found "${w[i]}")`);
        }
        continue;
      }
      if (line[0] === '[') throw new Error('Touchstone 2.0 keyword files are not supported; export version 1 (.s1p or .s2p)');
      const v = line.split(/\s+/).map(Number);
      if (!per) per = v.length;
      nums.push(...v);
    }
    if (!nums.length) throw new Error('no data rows found');
    if (nums.some(x => !Number.isFinite(x))) throw new Error('the file has text where numbers should be');
    if (per !== 3 && per !== 9) throw new Error(`rows have ${per} numbers; one-port files have 3 and two-port files 9`);
    const n = Math.floor(nums.length / per), names = per === 9 ? ['S11', 'S21', 'S12', 'S22'] : ['S11'];
    const fGHz = new Float64Array(n), S = {};
    names.forEach(k => { S[k] = { re: new Float64Array(n), im: new Float64Array(n) }; });
    for (let r = 0; r < n; r++) {
      fGHz[r] = nums[r * per] * unit;
      names.forEach((k, c) => {
        const x = nums[r * per + 1 + 2 * c], y = nums[r * per + 2 + 2 * c];
        if (fmt === 'RI') { S[k].re[r] = x; S[k].im[r] = y; return; }
        const mag = fmt === 'DB' ? Math.pow(10, x / 20) : x, th = y * Math.PI / 180;
        S[k].re[r] = mag * Math.cos(th); S[k].im[r] = mag * Math.sin(th);
      });
    }
    return { fGHz, S, z0 };
  }

  /* ------------------------------------------------------------ practice data */

  function rng(seed) {
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(TAU * r());

  const LEVELS = {
    1: { label: 'Clean', noise: 0.004, delay: false, phi: 0, points: 401 },
    2: { label: 'With cable delay', noise: 0.01, delay: true, phi: 0.15, points: 401 },
    3: { label: 'Realistic', noise: 0.03, delay: true, phi: 0.4, points: 201 },
  };

  /** A reproducible mystery resonator. Returns {seed, level, model, truth: {fr, Qi, Qc, phi, a, alpha, tau}, f, re, im}. */
  function practice(seed, level = 2, model = 'hanger') {
    const L = LEVELS[level] || LEVELS[2], r = rng(seed * 7919 + level * 104729 + Object.keys(MODELS).indexOf(model));
    const Qi = Math.pow(10, 4 + 2 * r());
    const Qc = Qi * Math.pow(10, -1 + 2 * r());
    const phi = M_bg(model) ? L.phi * (2 * r() - 1) : 0;
    const fr = 4 + 4 * r();
    const truth = {
      fr: Math.round(fr * 1e6) / 1e6, Qi, Qc, phi,
      a: M_bg(model) ? 0.3 + 0.6 * r() : 1, alpha: L.delay ? TAU * r() - Math.PI : 0, tau: L.delay ? 20 + 60 * r() : 0,
    };
    const Ql = loadedQ(Qi, Qc, phi), span = (6 + 6 * r()) * truth.fr / Ql;
    const f = Float64Array.from({ length: L.points }, (_, j) => truth.fr - span / 2 + span * j / (L.points - 1));
    const s = sweep(model, truth, f), sc = L.noise * truth.a;
    for (let j = 0; j < f.length; j++) { s.re[j] += sc * gauss(r); s.im[j] += sc * gauss(r); }
    return { seed, level, model, truth, f, re: s.re, im: s.im };
  }
  const M_bg = model => MODELS[model].background;

  const api = {
    MODELS, TARGETS, LEVELS, loadedQ, internalQ, sweep, regime, photonNumber, powerForPhotons,
    levmar, propagate, guess, fitResonator, circleFit, unwrap,
    parse, evaluate, toPython, fitFormula, pythonScript, FormulaError, RESERVED,
    parseTable, parseTouchstone, practice, rng, circleResid,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ResonatorEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
