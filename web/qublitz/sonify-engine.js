/* sonify-engine.js: browser port of the signal path in pages/Sonify.py and
 * utils/sound_to_image.py. No dependencies. Loads as a classic <script>
 * (window.SonifyEngine) or from Node (require), so the same file is what the page
 * runs and what tests/sonify-engine.test.js checks against Python.
 *
 * Model, same as the Python page:
 *   a grayscale image is a matrix of envelopes in [0, 1], one per row; the columns
 *   span 0 … duration seconds (numpy linspace without the endpoint)
 *   modulated row = envelope · cos(2π f t),  f = the modulation frequency
 *   audio: the row is linearly interpolated to 22050 Hz, multiplied by the carrier,
 *   given 2 % linear fades at each end, and normalized to a peak of 1
 *   sound → image: first channel, block-averaged to at most 16384 samples,
 *   normalized to [0, 1] and tiled into 64 rows
 */
(function (root) {
  'use strict';
  const DEFAULT_SR = 22050;

  // numpy.linspace(a, b, n, endpoint=False)
  function timeAxis(n, a, b) {
    const out = new Float64Array(n), step = (b - a) / n;
    for (let i = 0; i < n; i++) out[i] = a + i * step;
    return out;
  }

  // numpy.linspace(a, b, n) with the endpoint
  function linspace(a, b, n) {
    const out = new Float64Array(n);
    if (n === 1) { out[0] = a; return out; }
    const step = (b - a) / (n - 1);
    for (let i = 0; i < n; i++) out[i] = a + i * step;
    out[n - 1] = b;
    return out;
  }

  function modulate(env, t, f) {
    const out = new Float64Array(env.length), w = 2 * Math.PI * f;
    for (let i = 0; i < env.length; i++) out[i] = env[i] * Math.cos(w * t[i]);
    return out;
  }

  // ---- FFT: radix-2 in place, Bluestein for other lengths ----
  function fft2(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inverse ? 2 : -2) * Math.PI / len, half = len >> 1;
      for (let k = 0; k < half; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        for (let i = k; i < n; i += len) {
          const j = i + half, xr = re[j] * wr - im[j] * wi, xi = re[j] * wi + im[j] * wr;
          re[j] = re[i] - xr; im[j] = im[i] - xi; re[i] += xr; im[i] += xi;
        }
      }
    }
  }

  function fft(re, im) {
    const n = re.length;
    if ((n & (n - 1)) === 0) { fft2(re, im, false); return; }
    // Bluestein: express the length-n DFT as a power-of-two convolution.
    let m = 1; while (m < 2 * n - 1) m <<= 1;
    const cr = new Float64Array(n), ci = new Float64Array(n);
    for (let k = 0; k < n; k++) { const a = Math.PI * ((k * k) % (2 * n)) / n; cr[k] = Math.cos(a); ci[k] = -Math.sin(a); }
    const ar = new Float64Array(m), ai = new Float64Array(m), br = new Float64Array(m), bi = new Float64Array(m);
    for (let k = 0; k < n; k++) {
      ar[k] = re[k] * cr[k] - im[k] * ci[k]; ai[k] = re[k] * ci[k] + im[k] * cr[k];
      br[k] = cr[k]; bi[k] = -ci[k];
      if (k) { br[m - k] = cr[k]; bi[m - k] = -ci[k]; }
    }
    fft2(ar, ai, false); fft2(br, bi, false);
    for (let k = 0; k < m; k++) { const t = ar[k] * br[k] - ai[k] * bi[k]; ai[k] = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = t; }
    fft2(ar, ai, true);
    for (let k = 0; k < n; k++) {
      const xr = ar[k] / m, xi = ai[k] / m;
      re[k] = xr * cr[k] - xi * ci[k]; im[k] = xr * ci[k] + xi * cr[k];
    }
  }

  // |numpy.fft.rfft(x)|
  function rfftMag(x) {
    const n = x.length, re = Float64Array.from(x), im = new Float64Array(n);
    fft(re, im);
    const out = new Float64Array((n >> 1) + 1);
    for (let k = 0; k < out.length; k++) out[k] = Math.hypot(re[k], im[k]);
    return out;
  }

  // numpy.fft.rfftfreq(n, d)
  function rfftFreq(n, d) {
    const val = 1 / (n * d), out = new Float64Array((n >> 1) + 1);
    for (let k = 0; k < out.length; k++) out[k] = k * val;
    return out;
  }

  // numpy.interp for increasing xp, clamped at both ends.
  function interp(x, xp, fp) {
    const out = new Float64Array(x.length), last = xp.length - 1;
    let j = 0;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      if (v <= xp[0]) { out[i] = fp[0]; continue; }
      if (v >= xp[last]) { out[i] = fp[last]; continue; }
      while (xp[j + 1] <= v) j++;
      while (xp[j] > v) j--;
      const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
      out[i] = slope * (v - xp[j]) + fp[j];
    }
    return out;
  }

  // generate_audio(): one row of the image as a waveform at `sr`, as Float32Array.
  function generateAudio(env, tEnv, f, duration, sr = DEFAULT_SR) {
    const N = Math.ceil(duration * sr), t = timeAxis(N, 0, duration);
    const t0 = tEnv[0], xp = Float64Array.from(tEnv, v => v - t0);
    const audio = interp(t, xp, env);
    const w = 2 * Math.PI * f;
    for (let i = 0; i < N; i++) audio[i] *= Math.cos(w * t[i]);
    const L = Math.max(1, Math.trunc(0.02 * N)), up = linspace(0, 1, L), down = linspace(1, 0, L);
    for (let i = 0; i < L; i++) { audio[i] *= up[i]; audio[N - L + i] *= down[i]; }
    let peak = 0; for (let i = 0; i < N; i++) peak = Math.max(peak, Math.abs(audio[i]));
    const out = new Float32Array(N), s = peak + 1e-12;
    for (let i = 0; i < N; i++) out[i] = audio[i] / s;
    return out;
  }

  // utils/sound_to_image.audio_to_image_array, returning { rows, cols, data } (row-major).
  function audioToImage(audio, nRows = 64, maxCols = 16384) {
    let a = Float64Array.from(audio);
    if (a.length > maxCols) {
      const per = Math.ceil(a.length / maxCols), out = new Float64Array(maxCols);
      for (let k = 0; k < maxCols; k++) {
        let s = 0;
        for (let i = 0; i < per; i++) s += a[Math.min(k * per + i, a.length - 1)];   // edge padding
        out[k] = s / per;
      }
      a = out;
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = hi - lo, row = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) row[i] = span > 0 ? (a[i] - lo) / (span + 1e-12) : 0.5;
    const data = new Float64Array(nRows * a.length);
    for (let r = 0; r < nRows; r++) data.set(row, r * a.length);
    return { rows: nRows, cols: a.length, data };
  }

  // PIL Image.convert("L") on RGBA pixels (alpha ignored), scaled to [0, 1].
  function lumaFromRGBA(px, w, h) {
    const data = new Float64Array(w * h);
    for (let i = 0, k = 0; k < w * h; i += 4, k++) {
      data[k] = ((px[i] * 19595 + px[i + 1] * 38470 + px[i + 2] * 7471 + 0x8000) >> 16) / 255;
    }
    return { rows: h, cols: w, data };
  }

  // Minimal WAV reader (PCM 8/16/24/32-bit, float 32/64), scaled like soundfile.read.
  function parseWav(buf) {
    const v = new DataView(buf), tag = o => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
    if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('not a WAV file');
    let fmt = null, off = 12;
    while (off + 8 <= v.byteLength) {
      const id = tag(off), size = v.getUint32(off + 4, true), body = off + 8;
      if (id === 'fmt ') {
        let format = v.getUint16(body, true);
        if (format === 0xfffe) format = v.getUint16(body + 24, true);         // WAVE_FORMAT_EXTENSIBLE
        fmt = { format, channels: v.getUint16(body + 2, true), rate: v.getUint32(body + 4, true), bits: v.getUint16(body + 14, true) };
      } else if (id === 'data' && fmt) {
        const bps = fmt.bits / 8, frames = Math.floor(Math.min(size, v.byteLength - body) / (bps * fmt.channels));
        const first = new Float64Array(frames);
        for (let i = 0; i < frames; i++) {
          const o = body + i * bps * fmt.channels;
          let s;
          if (fmt.format === 3) s = fmt.bits === 64 ? v.getFloat64(o, true) : v.getFloat32(o, true);
          else if (fmt.bits === 8) s = (v.getUint8(o) - 128) / 128;
          else if (fmt.bits === 16) s = v.getInt16(o, true) / 32768;
          else if (fmt.bits === 24) s = ((v.getUint8(o) | (v.getUint8(o + 1) << 8) | (v.getInt8(o + 2) << 16))) / 8388608;
          else if (fmt.bits === 32) s = v.getInt32(o, true) / 2147483648;
          else throw new Error(`unsupported WAV sample size (${fmt.bits} bits)`);
          first[i] = s;
        }
        return { sampleRate: fmt.rate, samples: first };
      }
      off = body + size + (size & 1);
    }
    throw new Error('WAV file has no audio data');
  }

  // 16-bit PCM mono WAV, for the page's audio player and download.
  function toWav(samples, sr) {
    const n = samples.length, buf = new ArrayBuffer(44 + 2 * n), v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, 'RIFF'); v.setUint32(4, 36 + 2 * n, true); str(8, 'WAVE');
    str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, 2 * sr, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, 2 * n, true);
    for (let i = 0; i < n; i++) v.setInt16(44 + 2 * i, Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767))), true);
    return buf;
  }

  const api = { DEFAULT_SR, timeAxis, linspace, modulate, fft, rfftMag, rfftFreq, interp, generateAudio,
    audioToImage, lumaFromRGBA, parseWav, toWav };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.SonifyEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
