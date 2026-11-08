// Checks the flyover physics in web/courses/engs23/doppler-engine.js (the Doppler mode of the Spectrogram Explorer).
// Usage: node web/courses/engs23/tests/doppler-engine.test.js
'use strict';
const D = require('../doppler-engine.js');

let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
const c = D.C_AIR;

// ---- the emission time solves t = τ + r(τ)/c, and τ < t ----
{
  const tr = { p0: [-800, 300, 450], vel: [50, 20, 0], t0: 3 }, mic = [100, -200, 0];
  let worst = 0, causal = true;
  for (let t = -20; t <= 40; t += 0.37) {
    const tau = D.emissionTime(tr, mic, t), s = D.posAt(tr, tau), r = Math.hypot(s[0] - mic[0], s[1] - mic[1], s[2] - mic[2]);
    worst = Math.max(worst, Math.abs(tau + r / c - t)); if (tau > t) causal = false;
  }
  check('emission time solves t = τ + r/c', worst < 1e-9, `max residual ${worst.toExponential(1)} s`);
  check('sound is heard after it is emitted (τ < t)', causal);
  // the frequency factor is dτ/dt (phase of the source clock), checked by a central difference
  let w2 = 0;
  for (let t = -10; t <= 30; t += 1.3) {
    const h = 1e-5, num = (D.emissionTime(tr, mic, t + h) - D.emissionTime(tr, mic, t - h)) / (2 * h);
    w2 = Math.max(w2, Math.abs(num - D.heard(tr, mic, t).factor));
  }
  check('f/f0 = dτ/dt = 1/(1 + ṙ/c)', w2 < 1e-6, `max error ${w2.toExponential(1)}`);
}

// ---- the textbook flyover: asymptotes, CPA, slope, inversions ----
{
  const f0 = 80, v = 55, Dd = 350, t0 = 9;
  const far = D.flyoverFreq(-300, f0, v, Dd, t0), late = D.flyoverFreq(400, f0, v, Dd, t0);
  check('long before the CPA: f → f0/(1 − v/c)', Math.abs(far - D.fApproach(f0, v)) < 0.01, `${far.toFixed(3)} vs ${D.fApproach(f0, v).toFixed(3)} Hz`);
  check('long after: f → f0/(1 + v/c)', Math.abs(late - D.fRecede(f0, v)) < 0.01, `${late.toFixed(3)} vs ${D.fRecede(f0, v).toFixed(3)} Hz`);
  check('f = f0 when the CPA sound arrives (t0 + D/c)', Math.abs(D.flyoverFreq(t0 + Dd / c, f0, v, Dd, t0) - f0) < 1e-9);
  check('speed back from the two plateaus', Math.abs(D.speedFrom(D.fApproach(f0, v), D.fRecede(f0, v)) - v) < 1e-9);
  check('source frequency back from the two plateaus', Math.abs(D.sourceFrom(D.fApproach(f0, v), D.fRecede(f0, v)) - f0) < 1e-9);
  const tc = t0 + Dd / c, h = 1e-3, slope = (D.flyoverFreq(tc + h, f0, v, Dd, t0) - D.flyoverFreq(tc - h, f0, v, Dd, t0)) / (2 * h);
  const approx = D.cpaSlope(f0, v, Dd);
  check('steepest fall at the CPA ≈ −f0 v²/(c D)', Math.abs(slope / approx - 1) < 0.05, `${slope.toFixed(2)} vs ${approx.toFixed(2)} Hz/s`);
  // monotone fall: an approaching then receding source only ever drops in pitch
  let mono = true, prev = Infinity; for (let t = -30; t < 50; t += 0.1) { const f = D.flyoverFreq(t, f0, v, Dd, t0); if (f > prev + 1e-12) mono = false; prev = f; }
  check('the pitch only falls during a straight flyover', mono);
}

// ---- the rendered audio really carries the Doppler shift ----
// A short-time FFT (Hann, as on the page) of the synthetic recording; the loudest bin near the fundamental is compared
// with the predicted curve at several times.
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len >> 1; k++) { const p = i + k, q = p + (len >> 1), tr = re[q] * cr - im[q] * ci, ti = re[q] * ci + im[q] * cr;
      re[q] = re[p] - tr; im[q] = im[p] - ti; re[p] += tr; im[p] += ti; const nc = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nc; } } }
}
function stft(x, fs, N) {
  const hop = N >> 2, nBins = N >> 1, nFrames = 1 + Math.floor((x.length - N) / hop), data = new Float32Array(nFrames * nBins), re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f < nFrames; f++) { for (let i = 0; i < N; i++) { re[i] = x[f * hop + i] * 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1))); im[i] = 0; } fft(re, im);
    for (let b = 0; b < nBins; b++) data[f * nBins + b] = 20 * Math.log10(Math.hypot(re[b], im[b]) / N + 1e-12); }
  return { data, nFrames, nBins, hop, fs, fftSize: N };
}
const FS = 11025;
{
  const sc = D.flyoverScene(), x = D.render(D.flyoverSources(sc), { fs: FS, dur: sc.dur, mic: [0, 0, 0], noise: 0.02, rng: D.rng(7) });
  // peak of the 3rd harmonic (better resolved) from a parabolic fit, versus 3·f(t)
  const S = stft(x, FS, 4096), bw = FS / 4096;
  let worst = 0;
  for (const t of [2, 5, 14, 18]) {
    const fr = Math.round((t * FS - 2048) / S.hop), f3 = 3 * D.flyoverFreq(t, sc.f0, sc.v, sc.D, sc.t0);
    let bb = -1, bv = -Infinity; for (let b = Math.floor((f3 - 25) / bw); b <= Math.ceil((f3 + 25) / bw); b++) { const v = S.data[fr * S.nBins + b]; if (v > bv) { bv = v; bb = b; } }
    const a = S.data[fr * S.nBins + bb - 1], m = S.data[fr * S.nBins + bb + 1], d = 0.5 * (a - m) / (a - 2 * bv + m);
    worst = Math.max(worst, Math.abs((bb + d) * bw - f3));
  }
  check('the synthetic recording\'s 3rd harmonic follows 3 f(t)', worst < 1.5, `max error ${worst.toFixed(2)} Hz (bins ${bw.toFixed(2)} Hz, the window spans a changing pitch)`);
  // the fit score: the true parameters beat a wrong speed, a wrong distance and a wrong tone
  const floor = D.bandFloor(S, 30, 600), sco = (f0, v, Dd, t0) => D.scoreCurve(S, floor, t => D.flyoverFreq(t, f0, v, Dd, t0), 3, 0, sc.dur);
  const best = sco(sc.f0, sc.v, sc.D, sc.t0);
  const wrong = [sco(sc.f0, sc.v + 15, sc.D, sc.t0), sco(sc.f0, sc.v, sc.D * 3, sc.t0), sco(sc.f0 * 1.06, sc.v, sc.D, sc.t0), sco(sc.f0, sc.v, sc.D, sc.t0 + 2)];
  check('the true curve scores higher than wrong speed, distance, tone or time', wrong.every(w => w < 0.8 * best), `best ${best.toFixed(2)}, others ${wrong.map(w => w.toFixed(2)).join(', ')}`);
}

// ---- the mystery: two flyovers pin down the microphone; one alone leaves a mirror ambiguity ----
{
  let found = 0, mirror = 0; const trials = 3;
  for (let s = 1; s <= trials; s++) {
    const sc = D.mysteryScene(D.rng(100 + s)), x = D.render(D.mysterySources(sc), { fs: FS, dur: sc.dur, mic: sc.mic, rng: D.rng(s) });
    const S = stft(x, FS, 4096), floor = D.bandFloor(S, 30, 600), n = 40;
    const both = D.scoreMap(S, floor, sc, n, 'both');
    let bi = 0; for (let i = 1; i < both.length; i++) if (both[i] > both[bi]) bi = i;
    const step = 2 * sc.map / n, gx = -sc.map + ((bi % n) + 0.5) * step, gy = -sc.map + (Math.floor(bi / n) + 0.5) * step;
    const err = Math.hypot(gx - sc.mic[0], gy - sc.mic[1]);
    if (err < 400) found++;
    // with plane A alone, the reflection of the microphone across A's ground track scores about as well
    const T = sc.tracks[0].track, u = [T.vel[0] / D.speed(T), T.vel[1] / D.speed(T)], rel = [sc.mic[0] - T.p0[0], sc.mic[1] - T.p0[1]];
    const along = rel[0] * u[0] + rel[1] * u[1], foot = [T.p0[0] + along * u[0], T.p0[1] + along * u[1]], img = [2 * foot[0] - sc.mic[0], 2 * foot[1] - sc.mic[1], 0];
    const w = D.eventWindow(sc.tracks[0], sc.mic), sTrue = D.scoreCurve(S, floor, D.predictedCurve(sc.tracks[0], sc.mic), 3, w[0], w[1]);
    const sImg = D.scoreCurve(S, floor, D.predictedCurve(sc.tracks[0], img), 3, w[0], w[1]);
    if (Math.abs(sImg - sTrue) < 0.05 * sTrue && sTrue > 0.5) mirror++;
    console.log(`      scene ${s}: best grid cell ${err.toFixed(0)} m from the microphone; plane A alone: true ${sTrue.toFixed(2)}, mirror image ${sImg.toFixed(2)}`);
  }
  check('two flyovers locate the microphone (best of a 200 m grid within 400 m)', found === trials, `${found}/${trials}`);
  check('one flyover cannot tell the microphone from its mirror image across the track', mirror === trials, `${mirror}/${trials}`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
