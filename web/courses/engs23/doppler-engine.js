/* doppler-engine.js: aircraft flyovers heard by a microphone, for the Doppler mode of spectrogram.html.
 * No dependencies. Loads as a classic <script> (window.DopplerEngine) or from Node (require), so
 * tests/doppler-engine.test.js checks the same code the page runs.
 *
 * Physics: a source moves along a straight, level track at constant velocity through still air with sound speed c;
 * the microphone is at rest. Sound heard at time t left the source at the emission time τ, the root of
 *
 *   t = τ + |s(τ) − m| / c          (s = source position, m = microphone)
 *
 * which is unique for v < c (the right side increases with τ). The signal at the microphone is the source's signal at
 * τ, delayed and spread over a sphere: p(t) = q(τ) / r(τ). A tone q = sin(2π f₀ τ) therefore arrives with frequency
 *
 *   f(t) = f₀ dτ/dt = f₀ / (1 + ṙ/c),   ṙ = dr/dτ = (s − m)·v / r,
 *
 * the Doppler formula for a moving source and a stationary listener. ṙ < 0 on approach (pitch above f₀), ṙ = 0 at the
 * closest point of approach (CPA), ṙ > 0 going away. Far away on either side f → f₀/(1 ∓ v/c), so
 *
 *   v/c = (f_app − f_rec)/(f_app + f_rec),   f₀ = 2 f_app f_rec/(f_app + f_rec),
 *
 * and near the CPA (slant distance D) the pitch falls fastest, at df/dt ≈ −f₀ v²/(c D) for v ≪ c.
 * render() builds the pressure wave sample by sample from the exact τ, so the spectrogram of the synthetic recording
 * shows real Doppler curves; nothing in it is drawn by hand.
 *
 * Left out (say so on the page): wind and temperature gradients (c varies with height), ground reflections (which
 * add interference fringes to real flyover spectrograms), air absorption of high frequencies, and the convective
 * amplification of a moving source's loudness.
 */
(function (root) {
  'use strict';
  const C_AIR = 343;   // m/s, dry air at 20 °C

  // A track: position p0 = [x, y, z] (m) at time t0 (s), constant velocity vel = [vx, vy, vz] (m/s).
  const posAt = (tr, tau) => [tr.p0[0] + tr.vel[0] * (tau - tr.t0), tr.p0[1] + tr.vel[1] * (tau - tr.t0), tr.p0[2] + tr.vel[2] * (tau - tr.t0)];
  const speed = tr => Math.hypot(tr.vel[0], tr.vel[1], tr.vel[2]);

  // Emission time for reception time t. For a straight track at constant velocity this is a quadratic in τ:
  // c²(t − τ)² = |d + v(τ − t0)|², d = p0 − m. Take the root with τ < t (sound travels forward in time).
  function emissionTime(tr, mic, t, c) {
    c = c || C_AIR;
    const d = [tr.p0[0] - mic[0], tr.p0[1] - mic[1], tr.p0[2] - mic[2]], v = tr.vel;
    // let u = τ − t0, T = t − t0:  (c² − v²) u² − 2 (c² T + d·v) u + (c² T² − d·d) = 0
    const T = t - tr.t0, vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2], dv = d[0] * v[0] + d[1] * v[1] + d[2] * v[2], dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    const a = c * c - vv, b = c * c * T + dv, cc = c * c * T * T - dd;
    const disc = Math.max(0, b * b - a * cc);
    const u = (b - Math.sqrt(disc)) / a;   // the smaller root: the one with τ ≤ t
    return tr.t0 + u;
  }
  // Everything the microphone hears at time t from one track: emission time, distance, radial speed, frequency factor.
  function heard(tr, mic, t, c) {
    c = c || C_AIR;
    const tau = emissionTime(tr, mic, t, c), s = posAt(tr, tau);
    const dx = s[0] - mic[0], dy = s[1] - mic[1], dz = s[2] - mic[2], r = Math.hypot(dx, dy, dz);
    const rdot = (dx * tr.vel[0] + dy * tr.vel[1] + dz * tr.vel[2]) / r;
    return { tau, r, rdot, factor: 1 / (1 + rdot / c) };
  }
  // the closest point of approach: emission time and slant distance (v ≠ 0)
  function cpa(tr, mic) {
    const d = [tr.p0[0] - mic[0], tr.p0[1] - mic[1], tr.p0[2] - mic[2]], v = tr.vel, vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const u = -(d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / vv, s = posAt(tr, tr.t0 + u);
    return { tau: tr.t0 + u, D: Math.hypot(s[0] - mic[0], s[1] - mic[1], s[2] - mic[2]) };
  }
  // The textbook flyover with its own parameters: speed v, slant CPA distance D, CPA (emission) time t0, tone f0.
  // Received frequency at time t, exactly (same physics as heard(), written for the fitting sliders).
  function flyoverFreq(t, f0, v, D, t0, c) {
    c = c || C_AIR;
    const tr = { p0: [0, D, 0], vel: [v, 0, 0], t0 };
    return f0 * heard(tr, [0, 0, 0], t, c).factor;
  }
  // the two numbers a student reads off the curve, and what they give back
  const fApproach = (f0, v, c) => f0 / (1 - v / (c || C_AIR));
  const fRecede = (f0, v, c) => f0 / (1 + v / (c || C_AIR));
  const speedFrom = (fa, fr, c) => (c || C_AIR) * (fa - fr) / (fa + fr);
  const sourceFrom = (fa, fr) => 2 * fa * fr / (fa + fr);
  const cpaSlope = (f0, v, D, c) => -f0 * v * v / ((c || C_AIR) * D);   // Hz/s at the CPA, v ≪ c

  // A propeller's tone: the blade-pass frequency f0 and its harmonics. amps[k-1] for harmonic k.
  // render(sources, {fs, dur, mic, noise, rng}) → Float32Array, peak-normalized to 0.9.
  // sources: [{ track, f0, amps, level }]. Phase is integrated from the source's own clock at τ, so the Doppler
  // shift comes out of the propagation delay itself.
  function render(sources, o) {
    const fs = o.fs, n = Math.round(o.dur * fs), x = new Float32Array(n), c = o.c || C_AIR, R = o.rng || Math.random;
    const TAU = 2 * Math.PI;
    for (const S of sources) {
      const ph0 = S.amps.map(() => R() * TAU);
      for (let i = 0; i < n; i++) {
        const h = heard(S.track, o.mic, i / fs, c), g = (S.level || 1) * 300 / Math.max(h.r, 50);
        let v = 0;
        for (let k = 0; k < S.amps.length; k++) v += S.amps[k] * Math.sin(TAU * (k + 1) * S.f0 * h.tau + ph0[k]);
        x[i] += g * v;
      }
    }
    // background: brown-ish noise (wind and distant traffic, strongest at low frequency) plus a little white noise
    const nz = o.noise == null ? 0.06 : o.noise;
    if (nz > 0) { let b = 0; for (let i = 0; i < n; i++) { const w = R() * 2 - 1; b = 0.995 * b + 0.05 * w; x[i] += nz * (4 * b + 0.15 * w); } }
    let pk = 0; for (const v of x) pk = Math.max(pk, Math.abs(v));
    if (pk > 0) for (let i = 0; i < n; i++) x[i] *= 0.9 / pk;
    return x;
  }

  // ---- matching a model to a measured spectrogram ----
  // spec = { data (dB, frame-major), nFrames, nBins, hop, fs, fftSize }. bandFloor is each frame's median level in a band.
  function bandFloor(spec, fLo, fHi) {
    const { data, nFrames, nBins, fftSize, fs } = spec, b0 = Math.max(1, Math.floor(fLo * fftSize / fs)), b1 = Math.min(nBins - 1, Math.ceil(fHi * fftSize / fs));
    const out = new Float32Array(nFrames), tmp = [];
    for (let f = 0; f < nFrames; f++) { tmp.length = 0; for (let b = b0; b <= b1; b++) tmp.push(data[f * nBins + b]); tmp.sort((a, b) => a - b); out[f] = tmp[tmp.length >> 1]; }
    return out;
  }
  // curve(t) → fundamental frequency at time t (or null outside the event); harmonics 1..H are scored.
  // A point is "on a line" when the loudest of the 3 nearest bins is within 3 dB of the loudest bin in ±8 bins
  // (so it sits on the ridge, not on its flank) and stands 15 dB above the band's median. The score is the fraction of
  // points on a line, 0 to 1: a model that follows the real curves everywhere scores near 1.
  function scoreCurve(spec, floor, curve, H, tFrom, tTo) {
    const { data, nFrames, nBins, hop, fftSize, fs } = spec;
    let s = 0, cnt = 0;
    for (let f = 0; f < nFrames; f++) {
      const t = (f * hop + fftSize / 2) / fs; if (t < tFrom || t > tTo) continue;
      const f1 = curve(t); if (!(f1 > 0)) continue;
      for (let k = 1; k <= H; k++) {
        const b = Math.round(k * f1 * fftSize / fs); if (b < 9 || b >= nBins - 9) continue;
        const base = f * nBins, v = Math.max(data[base + b - 1], data[base + b], data[base + b + 1]);
        let loc = -Infinity; for (let j = b - 8; j <= b + 8; j++) loc = Math.max(loc, data[base + j]);
        if (v >= loc - 3 && v - floor[f] >= 15) s++;
        cnt++;
      }
    }
    return cnt ? s / cnt : 0;
  }

  // ---- scenes ----
  // The single flyover: a light propeller plane (two-blade prop at 2400 rpm → 80 Hz blade-pass), 55 m/s,
  // closest 350 m away at t = 9 s, in a 20 s clip. randomFlyover gives a new plane of the same kind.
  const PROP_AMPS = [1, 0.7, 0.5, 0.35, 0.25];
  function flyoverScene(p) { p = p || {}; return { f0: p.f0 || 80, v: p.v || 55, D: p.D || 350, t0: p.t0 == null ? 9 : p.t0, dur: p.dur || 20 }; }
  function randomFlyover(R) { R = R || Math.random; return flyoverScene({ f0: 70 + 30 * R(), v: 40 + 50 * R(), D: 200 + 500 * R(), t0: 7 + 5 * R(), dur: 20 }); }
  function flyoverSources(sc) { return [{ track: { p0: [0, sc.D, 0], vel: [sc.v, 0, 0], t0: sc.t0 }, f0: sc.f0, amps: PROP_AMPS, level: 1 }]; }

  // The mystery: two aircraft on known tracks (as ADS-B broadcasts give them), one microphone somewhere on an
  // 8 km × 8 km map (origin at the map's centre, z up). Each track is placed so its CPA to the microphone falls inside
  // the clip; the two headings differ by at least 50°, so the two left–right ambiguities cross at one point.
  const MAP = 4000;   // half-width, m
  const PLANES = [
    { name: 'Light plane', f0: 80, speed: 55, alt: 450, amps: PROP_AMPS, level: 1, tc: [8, 12] },
    { name: 'Turboprop', f0: 68, speed: 110, alt: 900, amps: [1, 0.8, 0.6, 0.45, 0.3], level: 1.6, tc: [26, 31] },
  ];
  function mysteryScene(R) {
    R = R || Math.random;
    const mic = [(R() * 2 - 1) * 2500, (R() * 2 - 1) * 2500, 0];
    const h1 = R() * 2 * Math.PI, h2 = h1 + (R() < 0.5 ? 1 : -1) * (50 + 80 * R()) * Math.PI / 180;
    const tracks = PLANES.map((P, k) => {
      const hd = k ? h2 : h1, u = [Math.cos(hd), Math.sin(hd)], nrm = [-u[1], u[0]];
      const off = (R() < 0.5 ? -1 : 1) * (250 + 1500 * R()), tc = P.tc[0] + (P.tc[1] - P.tc[0]) * R();
      // the plane is at mic + off·n̂ (horizontally) at time tc, at its altitude
      return { name: P.name, f0: P.f0, amps: P.amps, level: P.level, track: { p0: [mic[0] + off * nrm[0], mic[1] + off * nrm[1], P.alt], vel: [P.speed * u[0], P.speed * u[1], 0], t0: tc } };
    });
    return { mic, tracks, dur: 40, map: MAP };
  }
  function mysterySources(sc) { return sc.tracks.map(T => ({ track: T.track, f0: T.f0, amps: T.amps, level: T.level })); }
  // predicted fundamental at time t for a candidate microphone position (on the ground)
  function predictedCurve(T, mic, c) { return t => T.f0 * heard(T.track, mic, t, c).factor; }
  // the time window in which an event is loud enough to score: within ±12 s of its CPA (as heard)
  function eventWindow(T, mic) { const q = cpa(T.track, mic), tc = q.tau + q.D / C_AIR; return [tc - 12, tc + 12]; }
  // score every candidate on an n × n grid over the map; which = 0, 1 or 'both'
  function scoreMap(spec, floor, sc, n, which) {
    const out = new Float32Array(n * n), step = 2 * sc.map / n;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const mic = [-sc.map + (i + 0.5) * step, -sc.map + (j + 0.5) * step, 0];
      let s = 0;
      sc.tracks.forEach((T, k) => { if (which !== 'both' && which !== k) return; const w = eventWindow(T, mic); s += scoreCurve(spec, floor, predictedCurve(T, mic), 3, w[0], w[1]); });
      out[j * n + i] = s;
    }
    return out;
  }

  // xorshift32, so scenes are reproducible from a seed
  function rng(seed) { let x = (seed >>> 0) || 0x9e3779b9; return () => { x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0; return x / 4294967296; }; }

  const api = { C_AIR, posAt, speed, emissionTime, heard, cpa, flyoverFreq, fApproach, fRecede, speedFrom, sourceFrom, cpaSlope, render,
    bandFloor, scoreCurve, flyoverScene, randomFlyover, flyoverSources, mysteryScene, mysterySources, predictedCurve, eventWindow, scoreMap, PLANES, MAP, rng };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.DopplerEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
