/* qubit-engine.js: browser port of quantum_simulator.py plus the fit and Ramsey
 * helpers from pages/Qubit_Simulator.py. No dependencies. Loads as a classic
 * <script> (window.QubitEngine) or from Node (require), so the same file is
 * what the page runs and what tests/qubit-engine.test.js checks against qutip.
 *
 * Physics, same model and conventions as quantum_simulator.py:
 *   H(t) = 2π·ωq·σz/2 + 2π·Ωr·σx·I(t)·cos(2π·ωd·t) + 2π·Ωr·σy·Q(t)·cos(2π·ωd·t)
 *   lab frame, no rotating-wave approximation; frequencies in GHz, times in ns
 *   start in basis(2,1) (⟨σz⟩ = −1); P(|1⟩) = ρ00 = (1 + ⟨σz⟩)/2
 *   collapse ops √(1/T1)·σ− and √(γφ/2)·σz with γφ = 1/T2 − 1/(2T1)
 *   I(t), Q(t) are piecewise constant: env[min(floor(t/tFinal·n), n−1)]
 *
 * For a two-level system the Lindblad equation is exactly the Bloch equations
 * for r = (⟨σx⟩, ⟨σy⟩, ⟨σz⟩), which is what we integrate. The fast ωq precession
 * is removed analytically (exact z-rotation) and only the drive and decay are
 * stepped with RK4, so long runs at GHz frequencies stay accurate.
 *
 * One deliberate difference: quantum_simulator.py overwrites T2 with 2·T1
 * whenever T1 > 0. Here T2 is honoured as passed (floored at the T1 limit), and
 * the Qubit Simulator page passes T2 = 2·T1, its documented teaching model.
 */
(function (root) {
  'use strict';
  const TAU = 2 * Math.PI;

  /* ---------------- core solve ---------------- */

  /**
   * Bloch-equation solve, the counterpart of _solve_mesolve().
   * @returns {{t:Float64Array,x:Float64Array,y:Float64Array,z:Float64Array,p1:Float64Array}}
   *   expectation values and P(|1⟩) at tlist = linspace(0, tFinal, nSteps)
   */
  function solve({ omegaQ, omegaRabi, tFinal, nSteps, omegaD, envI, envQ, T1 = 0, T2 = 0 }) {
    const n = nSteps | 0;
    const Bz = TAU * omegaQ;
    const wd = TAU * omegaD;
    const amp = 2 * TAU * omegaRabi;              // H = ½ B·σ, so B = 2× the σ prefactor
    const g1 = T1 > 0 ? 1 / T1 : 0;               // population relaxation
    let g2 = 0;                                   // coherence decay
    if (T1 > 0) g2 = Math.max(0.5 * g1, T2 > 0 ? 1 / T2 : 0);
    const bin = tFinal / n;
    const tl = linspace(0, tFinal, n);

    // RK4 step budget: resolve the counter-rotating term at ωq + ωd and the drive itself.
    const fastest = Math.abs(omegaQ) + Math.abs(omegaD) + 2 * Math.abs(omegaRabi) + 1e-9;
    const dtMax = Math.min(1 / (48 * fastest), bin);

    const X = new Float64Array(n), Y = new Float64Array(n), Z = new Float64Array(n), P = new Float64Array(n);
    // r' is the state in the frame co-rotating with H0 at Bz (exact; decay is isotropic in x–y).
    let x = 0, y = 0, z = -1, t = 0;
    let out = 0, nextOut = tl[0], nextEdge = bin, k = 0;

    const record = () => {
      const c = Math.cos(Bz * t), s = Math.sin(Bz * t);
      X[out] = x * c - y * s; Y[out] = x * s + y * c; Z[out] = z;
      P[out] = clamp01(0.5 * (1 + z));
      out++; nextOut = out < n ? tl[out] : Infinity;
    };

    // dr'/dt with the lab drive B = (aI, aQ, 0)·cos(wd t) rotated into the H0 frame.
    let aI = 0, aQ = 0;
    const deriv = (tt, x, y, z, o) => {
      const d = Math.cos(wd * tt), c = Math.cos(Bz * tt), s = Math.sin(Bz * tt);
      const bx0 = aI * d, by0 = aQ * d;
      const bx = bx0 * c + by0 * s, by = -bx0 * s + by0 * c;
      o[0] = by * z - g2 * x;
      o[1] = -bx * z - g2 * y;
      o[2] = bx * y - by * x - g1 * (z + 1);
    };
    const k1 = [0, 0, 0], k2 = [0, 0, 0], k3 = [0, 0, 0], k4 = [0, 0, 0];

    while (out < n) {
      if (t >= nextOut - 1e-12) { record(); continue; }
      const segEnd = Math.min(nextOut, nextEdge);
      k = Math.min(Math.floor(((t + segEnd) / 2) / tFinal * n), n - 1);
      aI = amp * (envI ? +envI[k] || 0 : 0);
      aQ = amp * (envQ ? +envQ[k] || 0 : 0);
      const steps = Math.max(1, Math.ceil((segEnd - t) / dtMax - 1e-9));
      const h = (segEnd - t) / steps;
      for (let i = 0; i < steps; i++) {
        deriv(t, x, y, z, k1);
        deriv(t + h / 2, x + h / 2 * k1[0], y + h / 2 * k1[1], z + h / 2 * k1[2], k2);
        deriv(t + h / 2, x + h / 2 * k2[0], y + h / 2 * k2[1], z + h / 2 * k2[2], k3);
        deriv(t + h, x + h * k3[0], y + h * k3[1], z + h * k3[2], k4);
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        z += h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
        t += h;
      }
      t = segEnd;
      if (segEnd === nextEdge) nextEdge += bin;
    }
    return { t: tl, x: X, y: Y, z: Z, p1: P };
  }

  /** run_quantum_simulation(): the solve plus shot sampling. */
  function runQuantumSimulation(params, shots, rng) {
    const r = solve(params);
    r.p1Sampled = shots > 0 ? sampleShots(r.p1, shots, rng) : Float64Array.from(r.p1);
    return r;
  }

  /** run_frequency_sweep(): constant I = 1, Q = 0 drive at each frequency. Z[f][t] = P(|1⟩). */
  function frequencySweep({ startFreq, stopFreq, numPoints, tFinal, nSteps, omegaQ, omegaRabi, T1, T2 }, onProgress) {
    const freqs = linspace(startFreq, stopFreq, numPoints);
    const envI = new Float64Array(nSteps).fill(1);
    const Z = [];
    for (let i = 0; i < freqs.length; i++) {
      Z.push(solve({ omegaQ, omegaRabi, tFinal, nSteps, omegaD: freqs[i], envI, envQ: null, T1, T2 }).p1);
      if (onProgress) onProgress((i + 1) / freqs.length);
    }
    return { frequencies: freqs, timeList: linspace(0, tFinal, nSteps), Z };
  }

  /** Lab-frame ⟨σx⟩,⟨σy⟩ to the drive frame, exactly as run_time_domain() does it. */
  function rotatingFrame(t, x, y, omegaD) {
    const n = t.length, xr = new Float64Array(n), yr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const c = Math.cos(TAU * omegaD * t[i]), s = Math.sin(TAU * omegaD * t[i]);
      yr[i] = -(x[i] * c + y[i] * s);
      xr[i] = x[i] * s - y[i] * c;
    }
    return { x: xr, y: yr };
  }

  /* ---------------- Ramsey (closed form, as in the page) ---------------- */

  function ramseyP1(delays, detuningMHz, T1, T2) {
    const d = TAU * 1e-3 * detuningMHz, out = new Float64Array(delays.length);
    for (let i = 0; i < delays.length; i++) {
      // Rx(π/2)·(0,0,−1) = (0, 1, 0)
      let x = 0, y = 1, z = 0;
      const tau = delays[i], e2 = Math.exp(-tau / T2), e1 = Math.exp(-tau / T1);
      const c = Math.cos(d * tau), s = Math.sin(d * tau);
      const xn = e2 * (c * x - s * y), yn = e2 * (s * x + c * y), zn = e1 * z + (e1 - 1);
      // second Rx(π/2): z' = sin(π/2)·y + cos(π/2)·z = y
      x = xn; y = yn; z = zn;
      out[i] = clamp01(0.5 * (1 + y));
    }
    return out;
  }

  function ramseyChevron(delays, detunings, T1, T2, shots, rng) {
    return Array.from(detunings, det => {
      const col = ramseyP1(delays, det, T1, T2);
      return shots > 0 ? sampleShots(col, shots, rng) : col;
    });
  }

  /* ---------------- shots & randomness ---------------- */

  /** Seedable PRNG (mulberry32). */
  function rngFrom(seed) {
    let a = (seed >>> 0) || 0x9e3779b9;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Exact Binomial(n, p) by geometric waiting times: O(n·min(p, 1−p)) work. */
  function binomial(n, p, rng) {
    if (p <= 0) return 0;
    if (p >= 1) return n;
    const flip = p > 0.5, q = flip ? 1 - p : p, lq = Math.log1p(-q);
    let count = 0, pos = 0;
    for (;;) {
      pos += Math.floor(Math.log(1 - rng()) / lq) + 1;
      if (pos > n) break;
      count++;
    }
    return flip ? n - count : count;
  }

  function sampleShots(probs, shots, rng = Math.random) {
    const out = new Float64Array(probs.length);
    for (let i = 0; i < probs.length; i++) out[i] = binomial(shots, clamp01(probs[i]), rng) / shots;
    return out;
  }

  /* ---------------- hidden qubits from a shareable code ---------------- */

  const CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';   // no 0/O, 1/I/L

  function randomCode(rng = Math.random) {
    let s = '';
    for (let i = 0; i < 6; i++) s += CODE_ALPHABET[Math.floor(rng() * CODE_ALPHABET.length)];
    return s;
  }

  function hashString(str) {          // FNV-1a
    let h = 0x811c9dc5;
    for (const ch of str) { h ^= ch.codePointAt(0); h = Math.imul(h, 0x01000193); }
    return h >>> 0;
  }

  /**
   * Deterministic qubit for a code. Same code, same qubit, on any machine, so an
   * instructor can assign codes and check answers with the page's Reveal toggle.
   * Ranges match the assigned band the Streamlit page's defaults were tuned for.
   */
  const QUBIT_RANGES = { omegaQ: [4.0, 4.7], omegaRabiMHz: [10, 40], T1: [60, 250] };
  function qubitFromCode(code) {
    const norm = String(code).trim().toUpperCase();
    const r = rngFrom(hashString('qublitz:' + norm));
    const pick = ([a, b]) => a + (b - a) * r();
    return {
      code: norm,
      omegaQ: round(pick(QUBIT_RANGES.omegaQ), 4),
      omegaRabi: round(pick(QUBIT_RANGES.omegaRabiMHz) / 1000, 5),
      T1: round(pick(QUBIT_RANGES.T1), 1),
    };
  }

  /* ---------------- curve fitting ---------------- */

  /**
   * Bounded nonlinear least squares (Levenberg–Marquardt with projection onto
   * the box), standing in for scipy.optimize.curve_fit(..., bounds=...).
   * pcov follows curve_fit's default (absolute_sigma=False):
   *   pcov = (JᵀJ)⁻¹ · SSR / (m − n)
   * @returns {{popt:number[], perr:number[], ssr:number}|null}
   */
  function curveFit(model, xs, ys, p0, lower, upper, maxIter = 400) {
    const m = xs.length, np = p0.length;
    if (m <= np) return null;
    const lo = lower || p0.map(() => -Infinity), hi = upper || p0.map(() => Infinity);
    const proj = p => p.map((v, i) => Math.min(hi[i], Math.max(lo[i], v)));
    let p = proj(p0.slice());
    const resid = q => { const r = new Float64Array(m); for (let i = 0; i < m; i++) r[i] = ys[i] - model(xs[i], q); return r; };
    const ssrOf = r => { let s = 0; for (let i = 0; i < m; i++) s += r[i] * r[i]; return s; };
    const jac = (q, r0) => {
      const J = Array.from({ length: np }, () => new Float64Array(m));
      for (let j = 0; j < np; j++) {
        let h = 1.4901161193847656e-8 * Math.max(1, Math.abs(q[j]));
        if (q[j] + h > hi[j]) h = -h;                       // step inward at an upper bound
        const qh = q.slice(); qh[j] += h;
        for (let i = 0; i < m; i++) J[j][i] = (ys[i] - model(xs[i], qh) - r0[i]) / -h;
      }
      return J;                                              // J[j][i] = ∂f(x_i)/∂p_j
    };
    let r = resid(p), ssr = ssrOf(r), lambda = 1e-3;
    if (!isFinite(ssr)) return null;
    for (let it = 0; it < maxIter; it++) {
      const J = jac(p, r);
      const A = Array.from({ length: np }, (_, a) => Array.from({ length: np }, (_, b) => dot(J[a], J[b])));
      const g = Array.from({ length: np }, (_, a) => dot(J[a], r));
      let improved = false;
      for (let tries = 0; tries < 30; tries++) {
        const Ad = A.map((row, a) => row.map((v, b) => (a === b ? v * (1 + lambda) + 1e-15 : v)));
        const step = solveLinear(Ad, g);
        if (!step) { lambda *= 10; continue; }
        const pn = proj(p.map((v, i) => v + step[i]));
        const rn = resid(pn), sn = ssrOf(rn);
        if (isFinite(sn) && sn < ssr) {
          const rel = (ssr - sn) / Math.max(ssr, 1e-300);
          p = pn; r = rn; ssr = sn; lambda = Math.max(lambda / 10, 1e-12); improved = true;
          if (rel < 1e-12) it = maxIter;
          break;
        }
        lambda *= 10;
        if (lambda > 1e16) break;
      }
      if (!improved) break;
    }
    const J = jac(p, r);
    const A = Array.from({ length: np }, (_, a) => Array.from({ length: np }, (_, b) => dot(J[a], J[b])));
    const inv = invert(A), s2 = ssr / (m - np);
    const perr = p.map((_, i) => (inv && inv[i][i] >= 0 ? Math.sqrt(inv[i][i] * s2) : NaN));
    return { popt: p, perr, ssr };
  }

  /**
   * Dominant non-DC frequency of a mean-removed trace: the argmax of |rfft| over
   * bins ≥ 1, as the Python fit guesses do. Goertzel per bin, so long traces stay fast.
   */
  function dominantFreq(ts, ys) {
    const n = ys.length, dt = median(diff(ts));
    let mu = 0; for (const v of ys) mu += v; mu /= n;
    const nb = Math.floor(n / 2) + 1;
    let best = 1, bestMag = -1;
    for (let kk = 1; kk < nb; kk++) {
      const coeff = 2 * Math.cos(TAU * kk / n);
      let s1 = 0, s2 = 0;
      for (let i = 0; i < n; i++) { const s = ys[i] - mu + coeff * s1 - s2; s2 = s1; s1 = s; }
      const mag = s1 * s1 + s2 * s2 - coeff * s1 * s2;
      if (mag > bestMag) { bestMag = mag; best = kk; }
    }
    return nb > 1 ? best / (n * dt) : null;                  // cycles per ns
  }

  const rabiModel = (t, [A, fMHz, phi, tau, C]) => C + A * Math.exp(-t / tau) * Math.cos(TAU * fMHz * 1e-3 * t + phi);
  const t1Model = (t, [A, T1, C]) => C + A * Math.exp(-t / T1);
  const ramseyModel = (t, [A, T2, fMHz, phi, C]) => C + A * Math.exp(-t / T2) * Math.cos(TAU * fMHz * 1e-3 * t + phi);

  /** _fit_rabi_trace() */
  function fitRabi(ts, ys) {
    if (ts.length < 10) return null;
    const dt = median(diff(ts));
    if (!(dt > 0)) return null;
    const f = dominantFreq(ts, ys);
    const f0 = f != null ? Math.max(1e3 * f, 0.01) : 1.0;
    const p0 = [0.5 * (max(ys) - min(ys)), f0, 0, Math.max(0.6 * (ts[ts.length - 1] - ts[0]), dt), mean(ys)];
    const res = curveFit(rabiModel, ts, ys, p0, [-1.5, 0, -4 * Math.PI, dt, -0.5], [1.5, 5e4, 4 * Math.PI, 1e7, 1.5], 600);
    if (!res) return null;
    const [A, fMHz, phi, tauNs, C] = res.popt;
    return { A, fMHz, fMHzErr: finiteOrNull(res.perr[1]), phi, tauNs, tauNsErr: finiteOrNull(res.perr[3]), C,
      fitCurve: Array.from(ts, t => rabiModel(t, res.popt)), ok: true };
  }

  /** _fit_t1_trace() */
  function fitT1(ts, ys) {
    if (ts.length < 8) return null;
    const dt = median(diff(ts));
    if (!(dt > 0)) return null;
    const p0 = [max(ys) - min(ys), Math.max(0.5 * (ts[ts.length - 1] - ts[0]), dt), min(ys)];
    const res = curveFit(t1Model, ts, ys, p0, [0, dt, -0.2], [2, 1e7, 1.2]);
    if (!res) return null;
    const [A, T1, C] = res.popt;
    return { A, T1Ns: T1, T1NsErr: finiteOrNull(res.perr[1]), C, fitCurve: Array.from(ts, t => t1Model(t, res.popt)), ok: true };
  }

  /** _fit_ramsey_trace() */
  function fitRamsey(delays, ys, freqGuessMHz) {
    if (delays.length < 8) return null;
    const dt = median(diff(delays));
    if (!(dt > 0)) return null;
    const f = dominantFreq(delays, ys);
    let f0 = f != null ? Math.max(1e3 * f, 0.02) : 0.1;
    if (freqGuessMHz != null && isFinite(freqGuessMHz) && Math.abs(freqGuessMHz) >= 0.02) f0 = Math.abs(freqGuessMHz);
    const dmax = max(delays);
    const T20 = dmax > 0 ? 0.6 * dmax : 2 * dt;
    const p0 = [Math.max(0.5 * (max(ys) - min(ys)), 1e-3), Math.max(T20, dt), Math.max(Math.abs(f0), 0.05), 0, mean(ys)];
    const res = curveFit(ramseyModel, delays, ys, p0, [-1.5, dt, 0, -4 * Math.PI, -0.5], [1.5, 1e6, 5e3, 4 * Math.PI, 1.5]);
    if (!res) return null;
    const [A, T2, fMHz, phi, C] = res.popt;
    const errIfSane = (e, v) => (isFinite(e) && e <= 10 * Math.max(Math.abs(v), 1e-12) ? e : null);
    return { A, T2Ns: T2, T2NsErr: errIfSane(res.perr[1], T2), freqMHz: fMHz, freqMHzErr: errIfSane(res.perr[2], fMHz),
      phi, C, fitCurve: Array.from(delays, t => ramseyModel(t, res.popt)), ok: true };
  }

  /** _active_segments(): [start, end] index runs where |env| > threshold. */
  function activeSegments(env, threshold = 1e-8) {
    const segs = [];
    let start = -1;
    for (let i = 0; i < env.length; i++) {
      const on = Math.abs(env[i]) > threshold;
      if (on && start < 0) start = i;
      if (!on && start >= 0) { segs.push([start, i - 1]); start = -1; }
    }
    if (start >= 0) segs.push([start, env.length - 1]);
    return segs;
  }

  /** _analyze_time_domain_fits(): Rabi fit for a lone σx pulse, T1 fit on the post-pulse tail. */
  function analyzeTimeDomainFits(tl, p1, sx, sy) {
    const res = { rabi: null, t1: null, rabiSlice: null, t1Slice: null, messages: [] };
    const sxs = activeSegments(sx), sys = activeSegments(sy);
    if (sys.length === 0 && sxs.length === 1) {
      const [s0, s1] = sxs[0];
      if (s1 - s0 >= 8) {
        const t = Array.from(tl.slice(s0, s1 + 1), v => v - tl[s0]);
        const fit = fitRabi(t, Array.from(p1.slice(s0, s1 + 1)));
        if (fit) { res.rabi = fit; res.rabiSlice = [s0, s1]; }
      } else res.messages.push('Rabi fit skipped: the active pulse is too short.');
    } else {
      res.messages.push('Rabi fit is only shown for a single σx pulse with no σy pulse, because that corresponds to the standard pulse-duration sweep used to extract Ω_R.');
    }
    const all = sxs.concat(sys).sort((a, b) => a[0] - b[0]);
    if (all.length) {
      const lastEnd = all[all.length - 1][1];
      const startIdx = Math.min(lastEnd + 1, tl.length - 1);
      if (tl.length - startIdx >= 8) {
        const tail = Array.from(p1.slice(startIdx));
        const baseline = median(tail.slice(-Math.max(5, Math.floor(tail.length / 8))));
        let peak = 0, best = -1;
        tail.forEach((v, i) => { const d = Math.abs(v - baseline); if (d > best) { best = d; peak = i; } });
        const fitStart = Math.min(startIdx + peak, tl.length - 8);
        const t = Array.from(tl.slice(fitStart), v => v - tl[fitStart]);
        const fit = fitT1(t, Array.from(p1.slice(fitStart)));
        if (fit) { res.t1 = fit; res.t1Slice = [fitStart, tl.length - 1]; }
        else res.messages.push('T₁ fit skipped: the post-pulse relaxation segment could not be fit.');
      } else res.messages.push('T₁ fit skipped: there is not enough idle time after the last pulse.');
    } else res.messages.push('T₁ fit skipped: there is no pulse in the current schedule.');
    return res;
  }

  /* ---------------- small numerics ---------------- */

  function linspace(a, b, n) {
    const out = new Float64Array(n);
    if (n === 1) { out[0] = a; return out; }
    const step = (b - a) / (n - 1);
    for (let i = 0; i < n; i++) out[i] = a + i * step;
    out[n - 1] = b;
    return out;
  }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function round(v, d) { const f = 10 ** d; return Math.round(v * f) / f; }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function diff(a) { const o = []; for (let i = 1; i < a.length; i++) o.push(a[i] - a[i - 1]); return o; }
  function median(a) { const s = Array.from(a).sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])) : NaN; }
  function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }
  function max(a) { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; }
  function min(a) { let m = Infinity; for (const v of a) if (v < m) m = v; return m; }
  function finiteOrNull(v) { return isFinite(v) ? v : null; }
  function solveLinear(A, b) {
    const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < 1e-300) return null;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((row, i) => row[n] / row[i]);
  }
  function invert(A) {
    const n = A.length, cols = [];
    for (let j = 0; j < n; j++) { const e = new Array(n).fill(0); e[j] = 1; const c = solveLinear(A, e); if (!c) return null; cols.push(c); }
    return A.map((_, i) => cols.map(c => c[i]));
  }

  const api = {
    solve, runQuantumSimulation, frequencySweep, rotatingFrame,
    ramseyP1, ramseyChevron,
    rngFrom, binomial, sampleShots, randomCode, qubitFromCode, QUBIT_RANGES,
    curveFit, fitRabi, fitT1, fitRamsey, activeSegments, analyzeTimeDomainFits,
    linspace, median,
  };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QubitEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
