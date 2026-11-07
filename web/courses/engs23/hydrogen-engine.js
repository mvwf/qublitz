/* hydrogen-engine.js: the hydrogen atom of Griffiths, Introduction to Quantum Mechanics, ch. 4 (§4.2), plus
 * spontaneous emission and laser driving, for hydrogen.html. No dependencies. Loads as a classic <script>
 * (window.HydrogenEngine) or from Node (require), so tests/hydrogen-engine.test.js checks the code the page runs.
 *
 * States. ψ_nlm(r, θ, φ) = R_nl(r) Y_l^m(θ, φ), with
 *   R_nl(r) = √[(2/na)³ (n−l−1)! / (2n (n+l)!)] e^{−r/na} (2r/na)^l L^{2l+1}_{n−l−1}(2r/na)        (Griffiths §4.2)
 * (L^p_q is the associated Laguerre polynomial in the modern convention), and energies E_n = −hcR_H/n², 13.598 eV/n².
 * Wavefunction plots use the Bohr radius a; the matrix elements use the reduced-mass radius a_μ = a(1 + m_e/m_p) and
 * the wavelengths use R_H = R_∞/(1 + m_e/m_p) (vacuum wavelengths: Lyman-α 121.568 nm, H-α 656.47 nm, 656.28 nm in air).
 * Fine structure, the Lamb shift and hyperfine structure are left out, as in §4.2: they move these lines by ~0.002 nm
 * (Lyman-α is measured at 121.5668 and 121.5674 nm, its two fine-structure components; NIST ASD).
 *
 * Spontaneous emission. Fermi's golden rule with the free-space density of photon modes ρ(ω) = ω²V/(π²c³), summed
 * over both polarizations and all directions, gives the Einstein A coefficient (Griffiths §11.3; the golden-rule route is §11.4)
 *   A = ω³|⟨b|q r|a⟩|² / (3π ε₀ ħ c³)  =  (4αω³ / 3c²) |⟨b|r|a⟩|²,
 * and summing over the final m with the angular integrals of the Y_l^m gives, for nl → n'l' (l' = l ± 1),
 *   A(nl → n'l') = (4αω³ / 3c²) · max(l, l') / (2l + 1) · |∫ R_nl R_n'l' r³ dr|².
 * The lifetime is τ_nl = 1 / Σ A over every lower state. Only Δl = ±1 is allowed (electric dipole), so 2s cannot decay
 * to 1s this way: it lives 0.122 s by emitting two photons at once (rate 8.229 s⁻¹, a second-order process we quote,
 * not compute).
 *
 * Driving. A laser of intensity I and angular frequency ω_L near a transition g → e couples them with Rabi
 * frequency Ω = d E₀/ħ, E₀ = √(2I/(cε₀)), d = e |⟨e|z|g⟩| for light polarized along z (Δm = 0). The optical Bloch
 * equations of the two-level system, with the excited state's full decay rate Γ = 1/τ, give the excited population.
 *
 * Sources
 *   [1] D. J. Griffiths & D. F. Schroeter, Introduction to Quantum Mechanics, 3rd ed. (Cambridge, 2018): §4.2 the
 *       hydrogen atom (R_nl, Y_l^m, E_n, Rydberg formula); §11.3 spontaneous emission (Einstein A, lifetimes, selection rules).
 *   [2] H. A. Bethe & E. E. Salpeter, Quantum Mechanics of One- and Two-Electron Atoms (Springer, 1957): radial
 *       integrals, transition probabilities and lifetimes of hydrogen (checked in the tests).
 *   [3] W. L. Wiese & J. R. Fuhr, "Accurate atomic transition probabilities for hydrogen, helium, and lithium,"
 *       J. Phys. Chem. Ref. Data 38, 565 (2009): A coefficients used as test references.
 *   [4] E. Tiesinga, P. J. Mohr, D. B. Newell & B. N. Taylor, "CODATA recommended values of the fundamental physical
 *       constants: 2018," Rev. Mod. Phys. 93, 025010 (2021): every constant in K.
 *   [5] A. Kramida, Yu. Ralchenko, J. Reader & NIST ASD Team, NIST Atomic Spectra Database (physics.nist.gov/asd):
 *       measured H I wavelengths used as test references.
 *   [6] S. P. Goldman & G. W. F. Drake, Phys. Rev. A 24, 183 (1981): the 2s two-photon decay rate, 8.229 s⁻¹; first
 *       computed by L. Spitzer & J. L. Greenstein, Astrophys. J. 114, 407 (1951), 8.23 s⁻¹.
 *   [7] C. J. Foot, Atomic Physics (Oxford, 2005), ch. 7–8: optical Bloch equations, saturation intensity, power and
 *       Doppler broadening.
 *   [8] P. Thompson, D. E. Cox & J. B. Hastings, J. Appl. Cryst. 20, 79 (1987): the pseudo-Voigt approximation.
 */
(function (root) {
  'use strict';

  // ---- constants (CODATA 2018, SI; mH is the ¹H atomic mass from the atomic mass evaluation, unused since the atom table) ----
  const K = {
    c: 299792458, h: 6.62607015e-34, hbar: 1.054571817e-34, e: 1.602176634e-19, eps0: 8.8541878128e-12,
    kB: 1.380649e-23, alpha: 7.2973525693e-3, a0: 5.29177210903e-11, Rinf: 10973731.568160,
    meOverMp: 1 / 1836.15267343, mH: 1.00782503207 * 1.66053906660e-27,
  };
  K.RH = K.Rinf / (1 + K.meOverMp);            // m⁻¹
  K.amu = K.a0 * (1 + K.meOverMp);             // reduced-mass Bohr radius, m
  K.eV = K.e;
  K.Ry_eV = K.h * K.c * K.RH / K.eV;           // 13.5983 eV
  const NMAX = 7;
  const TWO_PHOTON_2S = 8.229;                 // s⁻¹, 2s → 1s + 2γ (Goldman & Drake 1981; Spitzer & Greenstein 1951 found 8.23)

  // ---- one-electron atoms: the same solution with nuclear charge Z and nuclear mass M ----
  // Every result scales exactly (Griffiths Problem 4.19 and §4.2): with the reduced mass μ = m_e M/(m_e + M),
  //   E_n = −Z² (μ/m_e) hcR_∞ / n²,  lengths in a_μ/Z with a_μ = a₀ m_e/μ,  ω ∝ Z² μ,  so A ∝ Z⁴ μ and τ ∝ 1/(Z⁴ μ);
  // the 2s two-photon rate scales as Z⁶ μ. Nuclear masses in u (CODATA 2018 / AME2016); the electron mass is
  // 5.48579909065×10⁻⁴ u. Fine structure grows as Z⁴α², so it matters more for He⁺ and Li²⁺ (see the page).
  const ME_U = 5.48579909065e-4, U = 1.66053906660e-27;
  const ATOMS = {
    H: { Z: 1, M: 1.007276466621, name: 'Hydrogen', sym: '¹H' },
    D: { Z: 1, M: 2.013553212745, name: 'Deuterium', sym: '²H (D)' },
    T: { Z: 1, M: 3.01550071621, name: 'Tritium', sym: '³H (T)' },
    He: { Z: 2, M: 4.001506179127, name: 'Helium ion', sym: '⁴He⁺' },
    Li: { Z: 3, M: 7.014357, name: 'Lithium ion', sym: '⁷Li²⁺' },
  };
  const AT = {};
  function setAtom(key) {
    const a = ATOMS[key] || ATOMS.H;
    AT.key = ATOMS[key] ? key : 'H'; AT.Z = a.Z; AT.name = a.name; AT.sym = a.sym;
    AT.muRatio = 1 / (1 + ME_U / a.M);                // μ / m_e
    AT.R = K.Rinf * AT.muRatio;                       // Rydberg constant for this nucleus, m⁻¹
    AT.a = K.a0 / AT.muRatio / AT.Z;                  // length unit of the wavefunctions, m
    AT.mass = (a.M + ME_U) * U;                       // mass of the moving atom or ion, kg
    AT.twoPhoton = TWO_PHOTON_2S * Math.pow(AT.Z, 6) * AT.muRatio / (1 / (1 + ME_U / ATOMS.H.M));
    AT.Ry_eV = K.h * K.c * AT.R / K.e;
    return AT;
  }
  setAtom('H');
  const L = ['s', 'p', 'd', 'f', 'g', 'h', 'i'];

  // ---- energies and wavelengths ----
  const energyEV = n => -AT.Z * AT.Z * AT.Ry_eV / (n * n);
  const waveNumber = (n1, n2) => AT.Z * AT.Z * AT.R * Math.abs(1 / (n1 * n1) - 1 / (n2 * n2));   // m⁻¹
  const wavelength = (n1, n2) => 1 / waveNumber(n1, n2);                             // m (vacuum)
  const omega = (n1, n2) => 2 * Math.PI * K.c * waveNumber(n1, n2);
  const label = (n, l) => `${n}${L[l]}`;

  // ---- radial functions ----
  function laguerre(q, p, x) {           // L^p_q(x), modern convention, by the three-term recurrence
    if (q === 0) return 1;
    let l0 = 1, l1 = 1 + p - x;
    for (let k = 1; k < q; k++) { const l2 = ((2 * k + 1 + p - x) * l1 - (k + p) * l0) / (k + 1); l0 = l1; l1 = l2; }
    return l1;
  }
  const fact = n => { let f = 1; for (let k = 2; k <= n; k++) f *= k; return f; };
  // R_nl(r) with r in units of the Bohr radius (so ∫ R² r² dr = 1 in those units)
  function radialR(n, l, r) {
    const x = 2 * r / n;
    const norm = Math.sqrt(Math.pow(2 / n, 3) * fact(n - l - 1) / (2 * n * fact(n + l)));
    return norm * Math.exp(-r / n) * Math.pow(x, l) * laguerre(n - l - 1, 2 * l + 1, x);
  }
  const meanR = (n, l) => (3 * n * n - l * (l + 1)) / 2;        // ⟨r⟩ in units of a
  const radialNodes = (n, l) => n - l - 1;

  // ---- spherical harmonics, |Y_l^m(θ)|² (independent of φ) ----
  function assocLegendre(l, m, x) {      // P_l^m(x), m ≥ 0, with the Condon–Shortley phase
    let pmm = 1;
    if (m > 0) { const s = Math.sqrt((1 - x) * (1 + x)); let f = 1; for (let i = 1; i <= m; i++) { pmm *= -f * s; f += 2; } }
    if (l === m) return pmm;
    let pm1 = x * (2 * m + 1) * pmm;
    if (l === m + 1) return pm1;
    let pll = 0;
    for (let ll = m + 2; ll <= l; ll++) { pll = (x * (2 * ll - 1) * pm1 - (ll + m - 1) * pmm) / (ll - m); pmm = pm1; pm1 = pll; }
    return pll;
  }
  function Ylm2(l, m, theta) {
    const am = Math.abs(m);
    const N = (2 * l + 1) / (4 * Math.PI) * fact(l - am) / fact(l + am);
    const P = assocLegendre(l, am, Math.cos(theta));
    return N * P * P;
  }

  // ---- radial integrals ∫ R_nl R_n'l' r³ dr (units of a), by Simpson's rule on a fine grid ----
  const cache = new Map();
  function radialIntegral(n1, l1, n2, l2) {
    const key = `${n1},${l1},${n2},${l2}`, rk = `${n2},${l2},${n1},${l1}`;
    if (cache.has(key)) return cache.get(key);
    const nm = Math.max(n1, n2), rmax = 8 * nm * nm + 40, N = Math.ceil(rmax / 0.01 / 2) * 2, h = rmax / N;
    let s = 0;
    for (let i = 0; i <= N; i++) {
      const r = i * h, w = i === 0 || i === N ? 1 : (i % 2 ? 4 : 2);
      s += w * radialR(n1, l1, r) * radialR(n2, l2, r) * r * r * r;
    }
    const v = s * h / 3;
    cache.set(key, v); cache.set(rk, v);
    return v;
  }

  // ---- spontaneous emission ----
  // A(nl → n'l'), s⁻¹, summed over final m (the same for every initial m).
  function einsteinA(n, l, n2, l2) {
    if (n2 >= n || Math.abs(l - l2) !== 1 || l2 >= n2) return 0;
    const w = omega(n, n2), I = radialIntegral(n, l, n2, l2) * AT.a;
    return 4 * K.alpha * w * w * w / (3 * K.c * K.c) * Math.max(l, l2) / (2 * l + 1) * I * I;
  }
  // every decay channel of nl, with rate, branching ratio and photon wavelength
  function decays(n, l) {
    const out = [];
    for (let n2 = 1; n2 < n; n2++) for (const l2 of [l - 1, l + 1]) {
      if (l2 < 0 || l2 >= n2) continue;
      const A = einsteinA(n, l, n2, l2);
      if (A > 0) out.push({ n: n2, l: l2, A, lambda: wavelength(n, n2) });
    }
    if (n === 2 && l === 0) out.push({ n: 1, l: 0, A: AT.twoPhoton, lambda: null, twoPhoton: true });
    const tot = out.reduce((s, d) => s + d.A, 0);
    for (const d of out) d.branch = d.A / tot;
    return out;
  }
  const totalRate = (n, l) => decays(n, l).reduce((s, d) => s + d.A, 0);
  const lifetime = (n, l) => (n === 1 ? Infinity : 1 / totalRate(n, l));

  // ---- cascades: where the atom ends up, and every photon it emits on the way, from nl ----
  function cascade(n, l) {
    const photons = new Map();            // key 'n,l→n',l'' → expected number of photons per excitation
    const pop = new Map([[`${n},${l}`, 1]]);
    for (let nn = n; nn >= 2; nn--) for (let ll = nn - 1; ll >= 0; ll--) {
      const p = pop.get(`${nn},${ll}`); if (!p) continue;
      for (const d of decays(nn, ll)) {
        const k = `${nn},${ll},${d.n},${d.l}`;
        photons.set(k, (photons.get(k) || 0) + p * d.branch);
        pop.set(`${d.n},${d.l}`, (pop.get(`${d.n},${d.l}`) || 0) + p * d.branch);
      }
    }
    return [...photons].map(([k, num]) => { const [a, b, c, d] = k.split(',').map(Number);
      return { from: [a, b], to: [c, d], num, lambda: a === 2 && b === 0 && c === 1 ? null : wavelength(a, c) }; })
      .sort((x, y) => y.num - x.num);
  }

  // ---- the decay board: every route from nl down to 1s, with its probability ----
  // A route is the list of states the atom visits; its probability is the product of the branching ratios A/ΣA at
  // each step (each decay is independent of the past: the atom keeps no memory of how it arrived). 2s → 1s is the
  // two-photon decay, the only way out of 2s. The probabilities of all routes add to 1.
  function routes(n, l) {
    const out = [], path = [[n, l]];
    (function rec(nn, ll, p) {
      if (nn === 1) { out.push({ path: path.map(s => s.slice()), p }); return; }
      for (const d of decays(nn, ll)) { path.push([d.n, d.l]); rec(d.n, d.l, p * d.branch); path.pop(); }
    })(n, l, 1);
    return out.sort((a, b) => b.p - a.p);
  }
  // probability that an atom put in nl at t = 0 has finished its one-photon cascade (reached 1s or the metastable 2s)
  // by each time in ts (s, ascending): the rate equations dP_i/dt = −Γ_i P_i + Σ_j A_{j→i} P_j, integrated by RK4 with a
  // step far below the shortest lifetime. Its derivative is the distribution of the cascade time: one exponential for a
  // single step, a sum of exponentials (a hypoexponential when the route is fixed) for longer cascades.
  function cascadeCDF(n, l, ts) {
    if (n === 1 || (n === 2 && l === 0)) return ts.map(() => 1);
    const S = []; for (let nn = 2; nn <= n; nn++) for (let ll = 0; ll < nn; ll++) if (!(nn === 2 && ll === 0)) S.push([nn, ll]);
    const ix = new Map(S.map((s, i) => [s.join(','), i]));
    const G = new Float64Array(S.length), feeds = S.map(() => []);
    S.forEach(([nn, ll], i) => { for (const d of decays(nn, ll)) { G[i] += d.A; const j = ix.get(d.n + ',' + d.l); if (j != null) feeds[j].push([i, d.A]); } });
    const f = P => P.map((p, i) => { let s = -G[i] * p; for (const [j, A] of feeds[i]) s += A * P[j]; return s; });
    let P = new Float64Array(S.length); P[ix.get(n + ',' + l)] = 1;
    const dt = 0.05 / Math.max(...G), out = []; let t = 0;
    for (const T of ts) {
      while (t < T) {
        const h = Math.min(dt, T - t), k1 = f(P), add = (k, c) => P.map((p, i) => p + c * k[i]);
        const k2 = f(add(k1, h / 2)), k3 = f(add(k2, h / 2)), k4 = f(add(k3, h));
        P = P.map((p, i) => p + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])); t += h;
      }
      out.push(Math.min(1, Math.max(0, 1 - P.reduce((a, b) => a + b, 0))));
    }
    return out;
  }

  // ---- driving ----
  // dipole matrix element |⟨n'l'0|z|n l 0⟩| in units of e·a_μ (π light, m = 0 → m = 0)
  function zElement(n, l, n2, l2) {
    if (Math.abs(l - l2) !== 1) return 0;
    const lm = Math.max(l, l2);             // ⟨l'0|cos θ|l0⟩ = lm / √((2lm−1)(2lm+1))
    return Math.abs(radialIntegral(n, l, n2, l2)) * lm / Math.sqrt((2 * lm - 1) * (2 * lm + 1));
  }
  const fieldFromIntensity = I => Math.sqrt(2 * I / (K.c * K.eps0));         // I in W/m² → E₀ in V/m
  function rabi(n, l, n2, l2, I) {                                          // Ω in rad/s
    return K.e * zElement(n, l, n2, l2) * AT.a * fieldFromIntensity(I) / K.hbar;
  }
  // Doppler FWHM (Hz) of a line at frequency nu0 for hydrogen atoms at temperature T
  const dopplerFWHM = (nu0, T) => nu0 * Math.sqrt(8 * K.kB * Math.max(T, 0) * Math.LN2 / (AT.mass * K.c * K.c));

  // Voigt profile (Lorentzian FWHM fL convolved with Gaussian FWHM fG) as an area-normalized density in 1/Hz, via the
  // pseudo-Voigt of Thompson, Cox & Hastings (1987), accurate to about 1%.
  const voigtFWHM = (fL, fG) => Math.pow(fG ** 5 + 2.69269 * fG ** 4 * fL + 2.42843 * fG ** 3 * fL ** 2
    + 4.47163 * fG ** 2 * fL ** 3 + 0.07842 * fG * fL ** 4 + fL ** 5, 0.2);
  function voigtDensity(dnu, fL, fG) {
    const lor = f => (2 / (Math.PI * f)) / (1 + 4 * dnu * dnu / (f * f));
    const gau = f => (2 * Math.sqrt(Math.LN2 / Math.PI) / f) * Math.exp(-4 * Math.LN2 * dnu * dnu / (f * f));
    if (fG <= 1e-9 * fL) return lor(fL);
    if (fL <= 1e-9 * fG) return gau(fG);
    const f = voigtFWHM(fL, fG), r = fL / f, eta = 1.36603 * r - 0.47719 * r * r + 0.11116 * r * r * r;
    return eta * lor(f) + (1 - eta) * gau(f);
  }

  // Every single-photon line from a starting state (1s or 2s) up to n = NMAX, π light (Δl = ±1, Δm = 0).
  function lines(start) {
    const [n0, l0] = start, out = [];
    for (let n = n0 + 1; n <= NMAX; n++) for (const l of [l0 - 1, l0 + 1]) {
      if (l < 0 || l >= n) continue;
      const lam = wavelength(n0, n), tau = lifetime(n, l);
      out.push({ n, l, lambda: lam, nu: K.c / lam, Gamma: 1 / tau, tau, d: zElement(n0, l0, n, l) });
    }
    return out;
  }
  // Steady-state excited fraction of one line for atoms at temperature T, laser at frequency nu (Hz), linewidth laserHz
  // (FWHM, treated as Gaussian), intensity I (W/m²). Each velocity class sees ρ_ee = (s/2)/(1 + s + (2Δ/Γ)²): a
  // power-broadened Lorentzian of FWHM fL = (Γ/2π)√(1+s) with area (s/2)/(1+s)·(π/2)fL. Averaging over the Maxwell
  // velocity distribution (and the laser's own spread) convolves it with a Gaussian, which keeps that area.
  function lineResponse(line, nu, I, T, laserHz, start) {
    const fL0 = line.Gamma / (2 * Math.PI);
    const Om = rabi(start[0], start[1], line.n, line.l, I);
    const s = 2 * Om * Om / (line.Gamma * line.Gamma);       // saturation parameter on resonance, s = I/I_sat
    const fL = fL0 * Math.sqrt(1 + s);
    const fD = dopplerFWHM(line.nu, T), fG = Math.sqrt(fD * fD + laserHz * laserHz);
    const area = (s / 2) / (1 + s) * Math.PI / 2 * fL;
    return { rho: area * voigtDensity(nu - line.nu, fL, fG), s, Om, fL, fD, fG, fwhm: voigtFWHM(fL, fG),
      peak: area * voigtDensity(0, fL, fG) };
  }
  // saturation intensity of a line, W/m²: I_sat = ħ²Γ²cε₀ / (4 d²)  (from s = 2Ω²/Γ² = I/I_sat)
  function Isat(line, start) {
    const d = K.e * zElement(start[0], start[1], line.n, line.l) * AT.a;
    return K.hbar * K.hbar * line.Gamma * line.Gamma * K.c * K.eps0 / (4 * d * d);
  }

  // Optical Bloch equations for a two-level atom: Bloch vector (u, v, w), w = ρ_ee − ρ_gg, decay Γ, detuning Δ.
  // Returns ρ_ee sampled at nOut points over [0, tEnd]. RK4 with a step well below 1/Ω, 1/Γ and 1/|Δ|.
  function bloch(Om, Delta, Gamma, tEnd, nOut) {
    const fastest = Math.max(Math.abs(Om), Math.abs(Delta), Gamma);
    // RK4 needs ω·dt below ~2.8 for every rate in the problem; far off resonance that would take too many steps,
    // and the answer is known: ρ_ee stays ≈ Ω²/(4Δ²) (the caller checks farOff() first).
    const steps = Math.max(nOut * 4, Math.ceil(tEnd * fastest / 0.05));
    if (steps > 2e6) throw new RangeError('bloch: detuning too large for direct integration; use the steady state');
    const dt = tEnd / steps, every = steps / (nOut - 1);
    let u = 0, v = 0, w = -1;
    const f = (u, v, w) => [-Gamma / 2 * u - Delta * v, Delta * u - Gamma / 2 * v - Om * w, Om * v - Gamma * (w + 1)];
    const t = [0], rho = [0];
    let next = every;
    for (let i = 1; i <= steps; i++) {
      const k1 = f(u, v, w), k2 = f(u + dt / 2 * k1[0], v + dt / 2 * k1[1], w + dt / 2 * k1[2]);
      const k3 = f(u + dt / 2 * k2[0], v + dt / 2 * k2[1], w + dt / 2 * k2[2]), k4 = f(u + dt * k3[0], v + dt * k3[1], w + dt * k3[2]);
      u += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
      v += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      w += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
      if (i >= next - 1e-9 || i === steps) { t.push(i * dt); rho.push((w + 1) / 2); next += every; }
    }
    return { t, rho };
  }

  // Bloch equations with a pulse sequence. Each pulse {t0, dur, Om, phase}: Rabi frequency Om (rad/s) about the axis
  // (cos φ, sin φ, 0) of the Bloch sphere; between pulses the atom only precesses at the detuning and decays.
  //   du/dt = −Γu/2 − Δv + Ω_v w,   dv/dt = Δu − Γv/2 − Ω_u w,   dw/dt = Ω_u v − Ω_v u − Γ(w + 1),
  // with (Ω_u, Ω_v) = Ω(cos φ, sin φ); φ = 0 gives the equations of bloch() above. Starts in the ground state.
  function blochPulses(pulses, Delta, Gamma, tEnd, nOut) {
    // integrate piece by piece between pulse edges, so every switch falls exactly on a step boundary
    const edges = new Set([0, tEnd]);
    for (const p of pulses) { if (p.t0 > 0 && p.t0 < tEnd) edges.add(p.t0); if (p.t0 + p.dur < tEnd) edges.add(p.t0 + p.dur); }
    const cuts = [...edges].sort((a, b) => a - b), out = { t: [0], rho: [0], u: [0], v: [0], w: [-1] };
    let u = 0, v = 0, w = -1;
    const sampleDt = tEnd / (nOut - 1); let nextSample = sampleDt;
    for (let s = 0; s < cuts.length - 1; s++) {
      const a = cuts[s], b = cuts[s + 1], mid = (a + b) / 2;
      const p = pulses.find(q => mid >= q.t0 && mid < q.t0 + q.dur);
      const Ou = p ? p.Om * Math.cos(p.phase || 0) : 0, Ov = p ? p.Om * Math.sin(p.phase || 0) : 0;
      const fastest = Math.max(Gamma, Math.abs(Delta), Math.hypot(Ou, Ov), 1e-30);
      const n = Math.max(4, Math.min(200000, Math.max(Math.ceil((b - a) * fastest / 0.02), Math.ceil((b - a) / sampleDt) * 4))), dt = (b - a) / n;
      const f = (u, v, w) => [-Gamma / 2 * u - Delta * v + Ov * w, Delta * u - Gamma / 2 * v - Ou * w, Ou * v - Ov * u - Gamma * (w + 1)];
      for (let i = 1; i <= n; i++) {
        const k1 = f(u, v, w), k2 = f(u + dt / 2 * k1[0], v + dt / 2 * k1[1], w + dt / 2 * k1[2]);
        const k3 = f(u + dt / 2 * k2[0], v + dt / 2 * k2[1], w + dt / 2 * k2[2]), k4 = f(u + dt * k3[0], v + dt * k3[1], w + dt * k3[2]);
        u += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); v += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); w += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
        const T = a + i * dt;
        if (T >= nextSample - 1e-15 || (s === cuts.length - 2 && i === n)) { out.t.push(T); out.rho.push((w + 1) / 2); out.u.push(u); out.v.push(v); out.w.push(w); nextSample += sampleDt; }
      }
    }
    return out;
  }

  // ---- colours for photons: real spectral colours in the visible, labelled tints outside it ----
  function photonColor(lambdaM) {
    const nm = lambdaM * 1e9;
    if (nm < 380) return { css: '#b48cff', band: 'UV' };
    if (nm > 750) return { css: '#ff7a6b', band: 'IR' };
    let r = 0, g = 0, b = 0;
    if (nm < 440) { r = -(nm - 440) / 60; b = 1; } else if (nm < 490) { g = (nm - 440) / 50; b = 1; }
    else if (nm < 510) { g = 1; b = -(nm - 510) / 20; } else if (nm < 580) { r = (nm - 510) / 70; g = 1; }
    else if (nm < 645) { r = 1; g = -(nm - 645) / 65; } else r = 1;
    const k = nm < 420 ? 0.4 + 0.6 * (nm - 380) / 40 : nm > 700 ? 0.4 + 0.6 * (750 - nm) / 50 : 1;
    const to = x => Math.round(255 * Math.pow(Math.max(0, x * k), 0.8));
    // lift dark violet/red so it reads on a dark page
    const lift = c => Math.min(255, Math.round(c * 0.75 + 64));
    return { css: `rgb(${lift(to(r))},${lift(to(g))},${lift(to(b))})`, band: 'visible' };
  }
  // series name for a transition ending on n1 (Lyman, Balmer, Paschen, …) and its Greek letter
  function seriesName(n1, n2) {
    const S = ['Lyman', 'Balmer', 'Paschen', 'Brackett', 'Pfund', 'Humphreys'], G = ['α', 'β', 'γ', 'δ', 'ε', 'ζ'];
    const lo = Math.min(n1, n2), hi = Math.max(n1, n2);
    return `${S[lo - 1] || 'n=' + lo}-${G[hi - lo - 1] || hi}`;
  }

  // far off resonance (|Δ| ≫ Ω, Γ), the Bloch solution is a tiny fast wiggle around ρ_ee ≈ Ω²/(4Δ²); no need to integrate
  const farOff = (Om, Delta, Gamma, tEnd) => tEnd * Math.max(Math.abs(Om), Math.abs(Delta), Gamma) / 0.05 > 2e6;
  const api = { K, NMAX, farOff, L, TWO_PHOTON_2S, ATOMS, setAtom, atom: () => AT, blochPulses, label, energyEV, waveNumber, wavelength, omega, laguerre, radialR, meanR,
    radialNodes, assocLegendre, Ylm2, radialIntegral, einsteinA, decays, totalRate, lifetime, cascade, routes, cascadeCDF, zElement,
    fieldFromIntensity, rabi, dopplerFWHM, voigtDensity, voigtFWHM, lines, lineResponse, Isat, bloch, photonColor, seriesName };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.HydrogenEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
