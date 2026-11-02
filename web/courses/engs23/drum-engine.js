/* drum-engine.js: the circular membrane of Boas, Mathematical Methods, ch. 13 §6, for drum-head.html.
 * No dependencies. Loads as a classic <script> (window.DrumEngine) or from Node (require), so
 * tests/drum-engine.test.js checks the same code the page runs.
 *
 * The membrane is clamped at r = a and obeys the wave equation (6.1), ∇²z = (1/v²) ∂²z/∂t².
 * Separating z = R(r)Θ(θ)T(t) gives the normal modes (6.6)
 *
 *   z = J_n(k_mn r/a) {cos nθ, sin nθ} {cos, sin}(k_mn v t/a),   ν_mn = k_mn v / (2πa),
 *
 * where k_mn is the m-th positive zero of J_n (Boas's labels: m counts zeros, n is the Bessel order).
 * Everything here uses ρ = r/a, so a mode's shape is J_n(k_mn ρ) and only the frequency carries a and v.
 *
 * Time is measured in periods of the fundamental (1,0): mode (m,n) has angular frequency
 * ω_mn = 2π k_mn/k_10 in those units.
 */
(function (root) {
  'use strict';

  // J_n(x) from Bessel's integral J_n(x) = (1/2π) ∫_0^{2π} cos(nτ − x sin τ) dτ. The integrand is periodic and
  // analytic, so the trapezoid rule converges geometrically; with N > |x| + n + 40 points the error is far below
  // double precision for the arguments used here (|x| < 60).
  function besselJ(n, x) {
    const N = Math.ceil(Math.abs(x)) + Math.abs(n) + 40;
    let s = 0;
    for (let j = 0; j < N; j++) {
      const t = 2 * Math.PI * j / N;
      s += Math.cos(n * t - x * Math.sin(t));
    }
    return s / N;
  }
  const besselJp = (n, x) => 0.5 * (besselJ(n - 1, x) - besselJ(n + 1, x));   // J_n'(x)

  // The first `count` positive zeros of J_n. J_n has none in (0, n], and consecutive zeros are more than
  // π/2 apart, so stepping by 0.05 from x = n cannot skip one; each bracket is closed by bisection and
  // polished with Newton.
  function besselZeros(n, count) {
    const out = [];
    let x0 = Math.max(n, 1e-6), f0 = besselJ(n, x0);
    while (out.length < count) {
      const x1 = x0 + 0.05, f1 = besselJ(n, x1);
      if (f0 === 0) out.push(x0);
      else if (f0 * f1 < 0) {
        let lo = x0, hi = x1, flo = f0;
        for (let i = 0; i < 40; i++) {
          const mid = 0.5 * (lo + hi), fm = besselJ(n, mid);
          if (flo * fm <= 0) hi = mid; else { lo = mid; flo = fm; }
        }
        let x = 0.5 * (lo + hi);
        for (let i = 0; i < 3; i++) x -= besselJ(n, x) / besselJp(n, x);
        out.push(x);
      }
      x0 = x1; f0 = f1;
    }
    return out;
  }

  // Gauss-Legendre nodes and weights on [0, 1].
  function gaussLegendre(N) {
    const x = new Float64Array(N), w = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let z = Math.cos(Math.PI * (i + 0.75) / (N + 0.5)), dp = 0;
      for (let it = 0; it < 100; it++) {
        let p0 = 1, p1 = z;
        for (let k = 2; k <= N; k++) { const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
        dp = N * (z * p1 - p0) / (z * z - 1);
        const dz = p1 / dp; z -= dz;
        if (Math.abs(dz) < 1e-15) break;
      }
      x[i] = 0.5 * (1 - z); w[i] = 1 / ((1 - z * z) * dp * dp);
    }
    return { x, w };
  }

  const RTAB = 1024;   // radial samples per mode for fast lookup on the page

  // The normal modes with order n = 0..nMax and m = 1..mMax.
  //   k      the zero k_mn (so K = k/a in (6.3))
  //   ratio  ν_mn / ν_10 = k_mn / k_10
  //   omega  angular frequency in fundamental periods, 2π·ratio
  //   normR  ∫_0^1 J_n(kρ)² ρ dρ = J_{n+1}(k)²/2
  //   normT  ∫_0^{2π} cos²nθ dθ  (π, or 2π for n = 0)
  //   peak   max over the disk of |J_n(kρ)|, used to scale a pure mode to unit height
  //   tab    J_n(kρ) sampled at ρ = i/RTAB
  function buildModes(nMax, mMax) {
    const zeros = [];
    for (let n = 0; n <= nMax; n++) zeros.push(besselZeros(n, mMax));
    const k10 = zeros[0][0], modes = [];
    for (let n = 0; n <= nMax; n++) for (let m = 1; m <= mMax; m++) {
      const k = zeros[n][m - 1], tab = new Float32Array(RTAB + 1);
      let peak = 0;
      for (let i = 0; i <= RTAB; i++) { const v = besselJ(n, k * i / RTAB); tab[i] = v; peak = Math.max(peak, Math.abs(v)); }
      const jn1 = besselJ(n + 1, k);
      modes.push({ m, n, k, ratio: k / k10, omega: 2 * Math.PI * k / k10,
        normR: 0.5 * jn1 * jn1, normT: n === 0 ? 2 * Math.PI : Math.PI, peak, tab });
    }
    modes.sort((p, q) => p.k - q.k);
    modes.forEach((md, i) => { md.index = i; });
    return modes;
  }

  // J_n(k_mn ρ) by linear interpolation in the mode's table.
  function radial(md, rho) {
    if (rho >= 1) return 0;
    const u = rho * RTAB, i = Math.floor(u), f = u - i;
    return md.tab[i] + (md.tab[i + 1] - md.tab[i]) * f;
  }

  // A state is one row per mode: z = Σ J_n(kρ)[(Ac cos ωt + Bc sin ωt) cos nθ + (As cos ωt + Bs sin ωt) sin nθ].
  // A are the initial displacement coefficients and B the initial velocity coefficients divided by ω.
  const emptyState = modes => modes.map(() => ({ Ac: 0, As: 0, Bc: 0, Bs: 0 }));

  // The cos nθ and sin nθ factors of mode md at time t: z = J_n(kρ)(c cos nθ + s sin nθ).
  function temporal(md, st, t) {
    const C = Math.cos(md.omega * t), S = Math.sin(md.omega * t);
    return [st.Ac * C + st.Bc * S, st.As * C + st.Bs * S];
  }

  function displacement(modes, state, rho, theta, t) {
    if (rho >= 1) return 0;
    let z = 0;
    for (let i = 0; i < modes.length; i++) {
      const st = state[i]; if (!st.Ac && !st.As && !st.Bc && !st.Bs) continue;
      const md = modes[i], [c, s] = temporal(md, st, t);
      z += radial(md, rho) * (c * Math.cos(md.n * theta) + s * Math.sin(md.n * theta));
    }
    return z;
  }

  // Project f(ρ, θ) onto the modes: the double-series coefficients of Boas (5.17), with r dr dθ as the weight.
  // Gauss-Legendre in ρ and the trapezoid rule (exact for trigonometric polynomials) in θ.
  function project(modes, f, opts) {
    const NR = (opts && opts.nr) || 96, NT = (opts && opts.nt) || 192;
    const { x, w } = gaussLegendre(NR);
    const vals = new Float64Array(NR * NT);
    for (let i = 0; i < NR; i++) for (let j = 0; j < NT; j++) vals[i * NT + j] = f(x[i], 2 * Math.PI * j / NT);
    return modes.map(md => {
      let c = 0, s = 0;
      for (let i = 0; i < NR; i++) {
        const R = besselJ(md.n, md.k * x[i]) * x[i] * w[i];
        let ci = 0, si = 0;
        for (let j = 0; j < NT; j++) {
          const th = 2 * Math.PI * j / NT, v = vals[i * NT + j];
          ci += v * Math.cos(md.n * th); si += v * Math.sin(md.n * th);
        }
        c += R * ci; s += R * si;
      }
      const dth = 2 * Math.PI / NT;
      return [c * dth / (md.normR * md.normT), md.n === 0 ? 0 : s * dth / (md.normR * Math.PI)];
    });
  }

  // A round mallet of width w (in units of a) centred at (ρ0, θ0): a Gaussian, peak 1.
  function mallet(rho0, theta0, w) {
    const x0 = rho0 * Math.cos(theta0), y0 = rho0 * Math.sin(theta0);
    return (rho, th) => {
      const dx = rho * Math.cos(th) - x0, dy = rho * Math.sin(th) - y0;
      return Math.exp(-(dx * dx + dy * dy) / (2 * w * w));
    };
  }

  // kind 'strike': the head is at rest and flat, and the mallet gives it velocity of the mallet's shape.
  // kind 'press':  the head is pushed into the mallet's shape and released from rest.
  function strike(modes, rho0, theta0, w, kind) {
    const coef = project(modes, mallet(rho0, theta0, w));
    return modes.map((md, i) => {
      const [c, s] = coef[i];
      return kind === 'press' ? { Ac: c, As: s, Bc: 0, Bs: 0 } : { Ac: 0, As: 0, Bc: c / md.omega, Bs: s / md.omega };
    });
  }

  // Energy in each mode (kinetic + potential, in units where σ = 1 and a = 1): ½ω²(A² + B²)·normR·normT,
  // with the cos and sin halves added. Modes of different frequency never exchange energy.
  const modeEnergy = (md, st) => 0.5 * md.omega * md.omega * md.normR *
    (md.normT * (st.Ac * st.Ac + st.Bc * st.Bc) + Math.PI * (st.As * st.As + st.Bs * st.Bs));

  // Time-averaged ⟨z²⟩ at a point. Different frequencies average out; the cos nθ and sin nθ halves of one
  // mode share a frequency and interfere. Sand gathers where this is small (Boas, end of §6).
  function meanSquare(modes, state, rho, theta) {
    if (rho >= 1) return 0;
    let s = 0;
    for (let i = 0; i < modes.length; i++) {
      const st = state[i]; if (!st.Ac && !st.As && !st.Bc && !st.Bs) continue;
      const md = modes[i], R = radial(md, rho), cn = Math.cos(md.n * theta), sn = Math.sin(md.n * theta);
      const a = st.Ac * cn + st.As * sn, b = st.Bc * cn + st.Bs * sn;
      s += 0.5 * R * R * (a * a + b * b);
    }
    return s;
  }

  // Physical frequency of a mode in Hz: ν = k v / (2πa) with v = √(T/σ) (tension per length, areal density).
  const waveSpeed = (tension, sigma) => Math.sqrt(tension / sigma);
  const frequencyHz = (md, a, tension, sigma) => md.k * waveSpeed(tension, sigma) / (2 * Math.PI * a);

  const api = { besselJ, besselJp, besselZeros, gaussLegendre, buildModes, radial, emptyState, temporal,
    displacement, project, mallet, strike, modeEnergy, meanSquare, waveSpeed, frequencyHz, RTAB };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.DrumEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
