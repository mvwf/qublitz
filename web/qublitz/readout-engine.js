/* readout-engine.js: dispersive readout of a qubit through a resonator, for qublitz/readout.html.
 *
 * Model (Blais et al., Rev. Mod. Phys. 93, 025005 (2021), §V; Gambetta et al., PRA 77, 012112 (2008)):
 *   The qubit pulls the resonator to f_r − χ/2π in |0⟩ and f_r + χ/2π in |1⟩ (a full shift of 2χ; papers differ on
 *   whether they quote χ or 2χ, and Coherence Vault stores the two separately). The resonator is driven at f_r + δ with a fixed
 *   amplitude ε, and its field α obeys
 *       dα/dt = −(κ/2 + iΔ_s) α − iε,   Δ_s = ω_s − ω_d  (ω_0 = −χ, ω_1 = +χ, ω_d = 2πδ, all relative to f_r)
 *   so α relaxes to the pointer state α_s = −iε/(κ/2 + iΔ_s) at rate κ/2. ε is set by n̄, the photon number a drive at
 *   the midpoint (δ = 0) gives in either state: ε² = n̄ (κ²/4 + χ²).
 *   Each segment is solved exactly: α(t) = α_s + (α(t0) − α_s) e^{−λ(t−t0)}, λ = κ/2 + iΔ_s.
 *
 * Measurement: the field leaks out at κ (no internal loss) and is amplified with efficiency η. The record is the field
 *   averaged over the integration time τ, r = (1/τ)∫α dt + noise, where vacuum noise gives each quadrature the
 *   variance 1/(4ηκτ). η = 1: an ideal phase-sensitive amplifier along the separation; a quantum-limited
 *   phase-preserving amplifier (both quadratures, as in the IQ plane here) is η = ½. In steady state this gives
 *   SNR = |r̄_1 − r̄_0|/σ = 2√(ηκτ) |α_1 − α_0|, and the measurement dephases the qubit at Γ_d = 2χ Im(α_0 α_1*),
 *   which is κ|α_0 − α_1|²/2 for a midpoint drive.
 *
 * Errors: the record is classified by a line halfway between the two mean records (the ideal linear discriminator for
 *   equal Gaussian blobs). A qubit prepared in |1⟩ can decay during the readout (rate 1/T₁), after which its field heads
 *   to α_0; this is integrated exactly over the decay time. A leftover thermal population P_exc means "prepare |0⟩"
 *   gives |1⟩ with probability P_exc, and the π pulse that prepares |1⟩ then gives |0⟩ with the same probability.
 *   Assignment fidelity F = 1 − [P(1|0) + P(0|1)]/2.
 *
 * Transmon dispersive shift (Koch et al., PRA 76, 042319 (2007)): χ = g² α_q / (Δ(Δ + α_q)), Δ = ω_q − ω_r, with α_q the
 *   anharmonicity; the dispersive picture holds below the critical photon number n_crit = Δ²/(4g²).
 *
 * Units: frequencies as f/2π in MHz, times in µs, so rates are 2π·MHz = rad/µs. Loads as a classic <script>
 * (window.ReadoutEngine) or from Node (require) for web/qublitz/tests/readout-engine.test.js.
 */
(function (root) {
  'use strict';
  const TAU = 2 * Math.PI;

  // complex numbers as [re, im]
  const C = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1]],
    mul: (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
    div: (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; },
    scale: (a, s) => [a[0] * s, a[1] * s],
    exp: a => { const e = Math.exp(a[0]); return [e * Math.cos(a[1]), e * Math.sin(a[1])]; },
    abs: a => Math.hypot(a[0], a[1]),
    conj: a => [a[0], -a[1]],
  };

  // The defaults are one consistent transmon: g/2π = 91 MHz, Δ/2π = −1.25 GHz, α/2π = −220 MHz give |χ|/2π ≈ 0.99 MHz and
  // n_crit ≈ 47 (transmonChi below), with κ = 2χ, the steady-state optimum. CIRCUIT is what the page's calculator starts from.
  const CIRCUIT = { g: 91, Delta: -1250, alpha: -220 };
  const DEFAULTS = { chi: 1, kappa: 2, nbar: 2, tau: 0.3, eta: 0.5, T1: 100, pexc: 0, delta: 0, ncrit: 47 };

  // rates in rad/µs from a parameter set (MHz, µs)
  function rates(p) {
    const k = TAU * p.kappa, x = TAU * Math.abs(p.chi), wd = TAU * (p.delta || 0);
    const eps = Math.sqrt(Math.max(0, p.nbar) * (k * k / 4 + x * x));
    return { k, x, wd, eps, lam: [[k / 2, -x - wd], [k / 2, x - wd]] };   // λ_s = κ/2 + iΔ_s
  }
  // steady pointer state of qubit state s
  function pointer(p, s) { const r = rates(p); return C.div([0, -r.eps], r.lam[s]); }
  function photons(p, s) { const a = pointer(p, s); return a[0] * a[0] + a[1] * a[1]; }
  // photons in the resonator in state s when driven at f_r + d (MHz), for the spectrum plot
  function photonsAt(p, s, d) { return photons(Object.assign({}, p, { delta: d }), s); }

  // one exact segment: start at a0, state s, for time T → { a: α(T), I: ∫α dt }
  function segment(r, a0, s, T) {
    if (T <= 0) return { a: a0, I: [0, 0] };
    const lam = r.lam[s], ass = C.div([0, -r.eps], lam), e = C.exp(C.scale(lam, -T));
    const d = C.sub(a0, ass);
    return { a: C.add(ass, C.mul(d, e)), I: C.add(C.scale(ass, T), C.div(C.mul(d, C.sub([1, 0], e)), lam)) };
  }
  // mean record (1/τ)∫α for a qubit in state s that, if s = 1 and td < τ, decays to |0⟩ at td. Starts from an empty resonator.
  function record(p, s, td) {
    const r = rates(p), T = p.tau;
    if (s === 0 || !(td < T)) return C.scale(segment(r, [0, 0], s, T).I, 1 / T);
    const a = segment(r, [0, 0], 1, td), b = segment(r, a.a, 0, T - td);
    return C.scale(C.add(a.I, b.I), 1 / T);
  }
  // field α(t) on [0, τ] (n points) for the trajectory plot
  function trajectory(p, s, n) {
    const r = rates(p), out = [];
    for (let i = 0; i <= n; i++) out.push(segment(r, [0, 0], s, p.tau * i / n).a);
    return out;
  }
  const sigma = p => 1 / Math.sqrt(4 * Math.max(1e-9, p.eta) * TAU * p.kappa * p.tau);

  // erfc by the Chebyshev fit of Numerical Recipes (erfcc), fractional error below 1.2×10⁻⁷ everywhere, and Φ, the normal CDF
  function erfc(x) {
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  const Phi = x => 0.5 * erfc(-x / Math.SQRT2);   // standard normal CDF

  // The discriminator: unit vector u from the |0⟩ blob to the |1⟩ blob, and the threshold halfway (projected), moved by
  // p.thrSigma noise widths toward the |1⟩ blob (negative: toward |0⟩). The midpoint is only optimal for two equal Gaussians;
  // with T₁ decay (records of |1⟩ slide toward |0⟩) or a thermal population the best line sits elsewhere (bestThreshold).
  function discriminator(p) {
    const m0 = record(p, 0), m1 = record(p, 1, Infinity), d = C.sub(m1, m0), L = C.abs(d) || 1e-300;
    const u = C.scale(d, 1 / L), proj = z => z[0] * u[0] + z[1] * u[1];
    const mid = (proj(m0) + proj(m1)) / 2, shift = (p.thrSigma || 0) * sigma(p);
    return { m0, m1, u, sep: L, thr: mid + shift, mid, shift, proj };
  }

  // Everything the page reports, exact (no sampling).
  function analyze(p) {
    const s = sigma(p), D = discriminator(p), x0 = D.proj(D.m0), x1 = D.proj(D.m1);
    const e01 = 1 - Phi((D.thr - x0) / s);                     // |0⟩ read as 1 (Gaussian tail)
    const e10g = Phi((D.thr - x1) / s);                        // |1⟩ read as 0, no decay
    // decay during the readout: average the miss probability over td ~ Exp(T₁) on [0, τ] (Simpson, 400 panels)
    let e10 = e10g, pdec = 0;
    if (p.T1 > 0 && isFinite(p.T1)) {
      const N = 400, h = p.tau / N; let acc = 0;
      for (let i = 0; i <= N; i++) {
        const t = i * h, w = (i === 0 || i === N) ? 1 : (i % 2 ? 4 : 2);
        acc += w * Math.exp(-t / p.T1) / p.T1 * Phi((D.thr - D.proj(record(p, 1, t))) / s);
      }
      pdec = 1 - Math.exp(-p.tau / p.T1);
      e10 = Math.exp(-p.tau / p.T1) * e10g + acc * h / 3;
    }
    const pe = Math.min(1, Math.max(0, (p.pexc || 0) / 100));
    const p10 = (1 - pe) * e01 + pe * (1 - e10), p01 = (1 - pe) * e10 + pe * (1 - e01);
    const a0 = pointer(p, 0), a1 = pointer(p, 1), r = rates(p);
    const gd = Math.abs(2 * r.x * C.mul(a0, C.conj(a1))[1]);   // measurement-induced dephasing, rad/µs
    return {
      sigma: s, snr: D.sep / s, sep: D.sep, m0: D.m0, m1: D.m1, u: D.u, thr: D.thr, mid: D.mid, thrSigma: p.thrSigma || 0,
      eGauss: (e01 + e10g) / 2, pDecay: pdec, p10, p01, F: 1 - (p10 + p01) / 2,
      n0: photons(p, 0), n1: photons(p, 1), a0, a1, gammaD: gd, tRing: 2 / r.k,
      snrSteady: 2 * Math.sqrt(Math.max(1e-9, p.eta) * r.k * p.tau) * C.abs(C.sub(a1, a0)),
      overCrit: p.ncrit > 0 && Math.max(photons(p, 0), photons(p, 1)) > p.ncrit,
    };
  }

  // The threshold position (in noise widths from the midpoint) that maximizes F, by a golden-section search bracketed
  // on [lo, hi]; F(thrSigma) is unimodal for two Gaussian blobs plus the decay tail. Returns { thrSigma, F }.
  function bestThreshold(p, lo = -4, hi = 4) {
    const F = s => analyze(Object.assign({}, p, { thrSigma: s })).F, g = (Math.sqrt(5) - 1) / 2;
    let a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a), fc = F(c), fd = F(d);
    for (let i = 0; i < 40; i++) {
      if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = F(c); }
      else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = F(d); }
    }
    const s = (a + b) / 2;
    return { thrSigma: s, F: F(s) };
  }

  // F against integration time τ, for the "best listening time" plot
  function fidelityCurve(p, taus) { return taus.map(t => analyze(Object.assign({}, p, { tau: t })).F); }
  function bestTau(p, lo, hi) {
    let best = { tau: lo, F: -1 };
    for (let i = 0; i <= 120; i++) { const t = lo * Math.pow(hi / lo, i / 120), F = analyze(Object.assign({}, p, { tau: t })).F; if (F > best.F) best = { tau: t, F }; }
    return best;
  }

  // Simulated shots: n per prepared state. Returns { pts: [[I, Q, prepared, true state at start, decayed]], wrong: [count0, count1] }
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function shots(p, n, seed) {
    const R = rng(seed == null ? 1 : seed), s = sigma(p), D = discriminator(p), pe = (p.pexc || 0) / 100, pts = [], wrong = [0, 0];
    const gauss = () => { let u = 0; while (u === 0) u = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * R()); };
    for (let prep = 0; prep < 2; prep++) for (let i = 0; i < n; i++) {
      const flipped = R() < pe, start = flipped ? 1 - prep : prep;
      const td = start === 1 && p.T1 > 0 ? -p.T1 * Math.log(1 - R()) : Infinity;
      const m = record(p, start, td), I = m[0] + s * gauss(), Q = m[1] + s * gauss();
      const read = D.proj([I, Q]) > D.thr ? 1 : 0;
      if (read !== prep) wrong[prep]++;
      pts.push([I, Q, prep, start, td < p.tau ? 1 : 0, read]);
    }
    return { pts, wrong, thr: D.thr, u: D.u };
  }

  // Transmon dispersive shift and critical photon number from circuit parameters (all /2π in MHz)
  function transmonChi(g, Delta, alpha) {
    const d2 = Delta + alpha;
    if (!Delta || !d2) return { chi: NaN, ncrit: NaN };
    return { chi: g * g * alpha / (Delta * d2), ncrit: Delta * Delta / (4 * g * g) };
  }

  const api = { DEFAULTS, CIRCUIT, C, rates, pointer, photons, photonsAt, segment, record, trajectory, sigma, erfc, Phi, discriminator, analyze, bestThreshold, fidelityCurve, bestTau, shots, transmonChi };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ReadoutEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
