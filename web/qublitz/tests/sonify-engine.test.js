// Checks web/qublitz/sonify-engine.js against the Python original.
// Reference data: sonify-reference.json, written by scripts/make_sonify_reference.py.
// Usage: node web/qublitz/tests/sonify-engine.test.js
'use strict';
const fs = require('fs');
const path = require('path');
const S = require('../sonify-engine.js');

const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'sonify-reference.json'), 'utf8'));
let failures = 0;
function check(label, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}
// Largest error over two arrays, relative to `scale` (the array's own peak by default).
function err(got, want, scale) {
  if (got.length !== want.length) return Infinity;
  const s = scale ?? Math.max(...want.map(Math.abs), 1e-300);
  let e = 0;
  for (let i = 0; i < want.length; i++) e = Math.max(e, Math.abs(got[i] - want[i]) / s);
  return e;
}

check('sample rate', S.DEFAULT_SR === ref.sr, `${S.DEFAULT_SR}`);
for (const c of ref.rows) {
  const t = S.timeAxis(c.cols, 0, c.duration);
  const mod = S.modulate(c.env, t, c.f);
  const audio = S.generateAudio(c.env, t, c.f, c.duration);
  const e = Math.max(err(t, c.t), err(mod, c.mod), err(S.rfftMag(mod), c.mag),
    err(S.rfftFreq(c.cols, c.duration / c.cols), c.freqs));
  // The audio is float32 in both, so compare at float32 precision.
  const ea = audio.length === c.audio_len ? err(audio.filter((_, i) => i % ref.audio_stride === 0), c.audio, 1) : Infinity;
  check(`row of ${c.cols} columns, ${c.duration} s at ${c.f} Hz`, e < 1e-11 && ea < 1e-6,
    `signal/FFT error ${e.toExponential(1)}, audio error ${ea.toExponential(1)}`);
}
for (const c of ref.images) {
  // Same input formula as scripts/make_sonify_reference.py.
  const audio = Float64Array.from({ length: c.n }, (_, i) =>
    c.n === 300 ? 0.3 : Math.sin(i * 0.013) * Math.exp(-i / c.n) + 0.05 * Math.sin(i * i * 0.0007));
  const img = S.audioToImage(audio);
  let same = true;
  for (let r = 1; r < img.rows && same; r++) for (let i = 0; i < img.cols; i++) if (img.data[r * img.cols + i] !== img.data[i]) { same = false; break; }
  const row = img.data.subarray(0, img.cols).filter((_, i) => i % ref.image_stride === 0);
  const e = img.rows === c.shape[0] && img.cols === c.shape[1] ? err(row, c.row, 1) : Infinity;
  check(`sound to image, ${c.n} samples`, e < 1e-9 && same === c.rows_equal, `${img.rows}×${img.cols}, error ${e.toExponential(1)}`);
}

// WAV round trip: toWav writes 16-bit PCM, parseWav reads it back like soundfile.
const tone = Float32Array.from({ length: 1000 }, (_, i) => Math.sin(i / 10) * 0.8);
const back = S.parseWav(S.toWav(tone, 22050));
check('WAV write/read round trip', back.sampleRate === 22050 && err(back.samples, tone, 1) < 1 / 16000,
  `max error ${err(back.samples, tone, 1).toExponential(1)}`);

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
