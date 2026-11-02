/* livespec.js: a live, scrolling spectrogram of whatever a page plays through Web Audio, plus the
 * instantaneous spectrum (the FFT of the newest window) underneath. Same colormap (inferno) and the same
 * "newest column enters on the right" display as the Spectrogram Explorer. No dependencies;
 * scripts/build_standalone.py inlines it into the single-file downloads.
 *
 *   const ls = LiveSpec.create({ spec: canvas, line: canvas, fmax: () => 2000, markers: () => [{f, label, color}] });
 *   source.connect(ls.input(audioContext));   // instead of source.connect(audioContext.destination)
 *
 * The input node is an AnalyserNode that passes the sound on to the speakers. The display scrolls only
 * while sound is coming through, at a fixed rate (opts.seconds of history across the width, 8 by default).
 * The spectrum line is live while the sound plays, then holds the loudest moment of that sound.
 * The level scale is fixed (dB relative to full scale), so loud and quiet partials keep their colours.
 *
 * LiveSpec.fft(re, im) is the radix-2 FFT the pages use for their static spectra.
 */
(function (root) {
  'use strict';
  const INFERNO = [[0,0,4],[12,8,38],[36,12,79],[66,10,104],[96,19,110],[127,29,103],[158,41,86],[186,54,69],
    [212,72,66],[233,91,47],[246,119,23],[251,152,6],[249,186,40],[241,219,94],[252,255,164]];
  function inferno(t) {
    if (t <= 0) return INFERNO[0];
    if (t >= 1) return INFERNO[INFERNO.length - 1];
    const x = t * (INFERNO.length - 1), i = Math.floor(x), f = x - i, a = INFERNO[i], b = INFERNO[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  const M = { l: 52, r: 12, t: 8, b: 26 };
  const HI = -20, RANGE = 70;            // dBFS shown from HI − RANGE (black) to HI (pale yellow)

  function create(opts) {
    const spec = opts.spec, line = opts.line, FFT = opts.fftSize || 8192;
    let an = null, off = null, raf = 0, quiet = 0, data = null, held = null;   // held: the spectrum line on show
    let loud = null, loudPk = -Infinity, lastT = 0, acc = 0;   // loudest frame of the current sound; scroll bookkeeping
    const WINDOW = opts.seconds || 8;                           // seconds of history across the spectrogram
    const fmaxOf = () => Math.max(100, typeof opts.fmax === 'function' ? opts.fmax() : (opts.fmax || 2000));
    const markersOf = () => (opts.markers ? opts.markers() : []);

    function size(cv, h) {
      const dpr = Math.min(root.devicePixelRatio || 1, 2), w = cv.clientWidth || 600;
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      return { W: cv.width, H: cv.height, dpr };
    }
    function geom() {
      const g = size(spec, opts.specHeight || 230), d = g.dpr;
      const m = { l: M.l * d, r: M.r * d, t: M.t * d, b: M.b * d };
      return Object.assign(g, { m, pw: Math.max(1, (g.W - m.l - m.r) | 0), ph: Math.max(1, (g.H - m.t - m.b) | 0) });
    }
    function ensureOff(g) {
      if (!off || off.width !== g.pw || off.height !== g.ph) {
        const o = document.createElement('canvas'); o.width = g.pw; o.height = g.ph;
        const c = o.getContext('2d'); c.fillStyle = '#000'; c.fillRect(0, 0, g.pw, g.ph);
        if (off) c.drawImage(off, g.pw - off.width, 0);
        off = o;
      }
    }

    function niceStep(fmax) { const raw = fmax / 6, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
      return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; }

    function paintSpec(g) {
      const ctx = spec.getContext('2d'), d = g.dpr, m = g.m, fmax = fmaxOf();
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, g.W, g.H);
      ctx.imageSmoothingEnabled = false; ctx.drawImage(off, m.l, m.t);
      ctx.font = (11 * d) + 'px Helvetica, Arial, sans-serif'; ctx.fillStyle = '#5a6b63';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const step = niceStep(fmax);
      for (let f = 0; f <= fmax + 1e-9; f += step) {
        const y = m.t + (1 - f / fmax) * g.ph;
        ctx.fillText(f >= 1000 ? (f / 1000) + 'k' : '' + Math.round(f), m.l - 5 * d, y);
        ctx.strokeStyle = '#c9d6cf'; ctx.beginPath(); ctx.moveTo(m.l - 3 * d, y); ctx.lineTo(m.l, y); ctx.stroke();
      }
      ctx.save(); ctx.translate(12 * d, m.t + g.ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
      ctx.fillText('frequency (Hz)', 0, 0); ctx.restore();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('← earlier        time        now →', m.l + g.pw / 2, m.t + g.ph + 6 * d);
      // predicted frequencies as ticks on the right edge
      markersOf().forEach(k => { if (k.f > fmax) return; const y = m.t + (1 - k.f / fmax) * g.ph;
        ctx.strokeStyle = k.color || '#9bd'; ctx.lineWidth = 2 * d; ctx.beginPath();
        ctx.moveTo(m.l + g.pw - 6 * d, y); ctx.lineTo(m.l + g.pw, y); ctx.stroke(); ctx.lineWidth = 1; });
    }

    function paintLine() {
      if (!line) return;
      const g = size(line, opts.lineHeight || 170), d = g.dpr, ctx = line.getContext('2d'), fmax = fmaxOf();
      const m = { l: M.l * d, r: M.r * d, t: 14 * d, b: 30 * d }, pw = g.W - m.l - m.r, ph = g.H - m.t - m.b;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#f5f9f6'; ctx.fillRect(0, 0, g.W, g.H);
      ctx.font = (11 * d) + 'px Helvetica, Arial, sans-serif'; ctx.lineWidth = 1;
      const fx = f => m.l + f / fmax * pw, fy = db => m.t + (1 - Math.max(0, Math.min(1, (db - (HI - RANGE)) / RANGE))) * ph;
      const step = niceStep(fmax);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let f = 0; f <= fmax + 1e-9; f += step) { const x = fx(f);
        ctx.strokeStyle = '#e4ede8'; ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke();
        ctx.fillStyle = '#5a6b63'; ctx.fillText(f >= 1000 ? (f / 1000) + 'k' : '' + Math.round(f), x, m.t + ph + 4 * d); }
      ctx.fillText('frequency (Hz)', m.l + pw / 2, m.t + ph + 17 * d);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let k = 0; k <= 4; k++) { const db = HI - RANGE * k / 4, y = m.t + k / 4 * ph;
        ctx.strokeStyle = '#e4ede8'; ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
        ctx.fillStyle = '#5a6b63'; ctx.fillText(Math.round(db) + ' dB', m.l - 5 * d, y); }
      // predicted lines, labelled
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      let lastX = -1e9;
      markersOf().forEach(k => { if (k.f > fmax) return; const x = fx(k.f);
        ctx.strokeStyle = k.color || '#9bb0a6'; ctx.setLineDash([3 * d, 3 * d]); ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ph); ctx.stroke(); ctx.setLineDash([]);
        if (k.label && x - lastX > 34 * d) { ctx.fillStyle = k.color || '#5a6b63'; ctx.fillText(k.label, x, m.t + 1); lastX = x; } });
      if (!held || !an) { ctx.fillStyle = '#8a9a92'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('press ♪ to hear it — the live FFT runs while the sound plays', m.l + pw / 2, m.t + ph / 2); return; }
      const sr = an.context.sampleRate, n = held.length;
      ctx.beginPath();
      for (let ox = 0; ox <= pw; ox++) { const f = ox / pw * fmax; let b = Math.round(f * FFT / sr); if (b > n - 1) b = n - 1;
        const y = fy(held[b]); ox ? ctx.lineTo(m.l + ox, y) : ctx.moveTo(m.l + ox, y); }
      ctx.strokeStyle = '#00693e'; ctx.lineWidth = 1.6 * d; ctx.lineJoin = 'round'; ctx.stroke();
      ctx.lineTo(m.l + pw, m.t + ph); ctx.lineTo(m.l, m.t + ph); ctx.closePath(); ctx.fillStyle = 'rgba(0,105,62,0.10)'; ctx.fill();
    }

    // scroll left by k pixels and paint the newest spectrum into the k new columns at the right
    function column(g, k) {
      const c = off.getContext('2d'), sr = an.context.sampleRate, fmax = fmaxOf(), n = data.length;
      c.globalCompositeOperation = 'copy'; c.drawImage(off, -k, 0); c.globalCompositeOperation = 'source-over';
      const col = c.createImageData(1, g.ph);
      for (let y = 0; y < g.ph; y++) {
        const f = (1 - y / (g.ph - 1)) * fmax; let b = Math.round(f * FFT / sr); if (b > n - 1) b = n - 1;
        const rgb = inferno((data[b] - (HI - RANGE)) / RANGE);
        col.data[4 * y] = rgb[0]; col.data[4 * y + 1] = rgb[1]; col.data[4 * y + 2] = rgb[2]; col.data[4 * y + 3] = 255;
      }
      for (let j = 1; j <= k; j++) c.putImageData(col, g.pw - j, 0);
    }

    function loop(ts) {
      raf = 0;
      if (!an) return;
      an.getFloatFrequencyData(data);
      let pk = -Infinity; for (let i = 0; i < data.length; i++) if (data[i] > pk) pk = data[i];
      const g = geom(); ensureOff(g);
      acc += Math.min(0.1, Math.max(0, (ts - lastT) / 1000)) * g.pw / WINDOW; lastT = ts;
      const k = Math.floor(acc); acc -= k;
      if (pk > HI - RANGE) {
        quiet = 0; held = data.slice();                       // live while there is sound
        if (pk > loudPk) { loudPk = pk; loud = held; }
        if (k > 0) column(g, k);
      } else { quiet++; if (loud) held = loud; }              // then show the loudest moment of that sound
      paintSpec(g); paintLine();
      if (quiet < 90) raf = requestAnimationFrame(loop);     // stop scrolling ~1.5 s after the sound ends
    }

    function input(actx) {
      if (!an || an.context !== actx) {
        an = actx.createAnalyser(); an.fftSize = FFT; an.smoothingTimeConstant = 0;
        an.minDecibels = HI - RANGE - 10; an.maxDecibels = HI;
        an.connect(actx.destination); data = new Float32Array(an.frequencyBinCount);
      }
      quiet = 0; loud = null; loudPk = -Infinity;
      if (!raf) { lastT = performance.now(); raf = requestAnimationFrame(loop); }
      return an;
    }
    function redraw() { const g = geom(); ensureOff(g); paintSpec(g); paintLine(); }
    function clear() { off = null; redraw(); }
    requestAnimationFrame(redraw);   // first draw after the page's own script has finished setting up
    return { input, redraw, clear };
  }

  // In-place iterative radix-2 Cooley-Tukey FFT (re, im: arrays of the same power-of-two length).
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len >> 1; k++) {
          const a = i + k, b = a + (len >> 1), tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  const api = { create, inferno, fft };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LiveSpec = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
