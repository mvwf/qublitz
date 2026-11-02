/* livespec.js: a live, scrolling spectrogram plus the instantaneous spectrum (the FFT of the newest window)
 * underneath. Same colormap (inferno) and the same "newest column enters on the right" display as the
 * Spectrogram Explorer. No dependencies; scripts/build_standalone.py inlines it into the single-file downloads.
 *
 *   const ls = LiveSpec.create({ spec: canvas, line: canvas, fmax: () => 2000,
 *                                markers: () => [{f, label, color}],
 *                                signal: (t0, fs, n) => Float64Array,   // optional, see below
 *                                running: () => bool });
 *   source.connect(ls.input(audioContext));   // instead of source.connect(audioContext.destination)
 *
 * Two feeds:
 *   sound   ls.input() is an AnalyserNode that passes the sound on to the speakers; while sound comes
 *           through, the display shows exactly what is heard.
 *   signal  otherwise, if the page gives signal(t0, fs, n) (n samples of its signal at rate fs from time t0 s)
 *           (and running(), if given, is true), the display runs on that signal silently, so it is live without
 *           sound. It only computes while the spectrogram is on screen.
 * The display scrolls at a fixed rate (opts.seconds of history across the width, 8 by default) and holds still
 * when neither feed is on. The level scale is fixed for sound (dBFS); the silent feed is scaled so its loudest
 * line sits near the top of the scale.
 *
 * Both plots also show the mean frequency ⟨f⟩ = Σ f·P(f) / Σ P(f), the power-weighted average (the centre of mass
 * of the power spectrum P = |X|², over 0..fmax): a white trace across the spectrogram and a line on the spectrum.
 *
 * LiveSpec.fft(re, im) is the radix-2 FFT the pages also use for their static spectra, and
 * LiveSpec.meanFrequency(freqs, amps) is ⟨f⟩ for a spectrum given as amplitudes.
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
  const HI = -20, RANGE = 70, FLOOR = HI - RANGE;   // dB shown from FLOOR (black) to HI (pale yellow)
  const NSIG = 2048;                                // samples per silent-feed window

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

  // ⟨f⟩ = Σ f a² / Σ a² for a spectrum given as frequencies and amplitudes (NaN when it is empty)
  function meanFrequency(freqs, amps) {
    let num = 0, den = 0;
    for (let i = 0; i < amps.length; i++) { const p = amps[i] * amps[i]; num += freqs[i] * p; den += p; }
    return den > 0 ? num / den : NaN;
  }
  const fmtHz = f => f >= 1000 ? (f / 1000).toFixed(2) + ' kHz' : Math.round(f) + ' Hz';

  function niceStep(fmax) { const raw = fmax / 6, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; }

  function create(opts) {
    const spec = opts.spec, line = opts.line, FFT = opts.fftSize || 8192, WINDOW = opts.seconds || 8;
    const fmaxOf = () => Math.max(100, typeof opts.fmax === 'function' ? opts.fmax() : (opts.fmax || 2000));
    const markersOf = () => (opts.markers ? opts.markers() : []);
    const running = () => !!opts.signal && (!opts.running || opts.running());
    // A frame is a spectrum: db[k] at frequency k·binHz.
    let an = null, adata = null, off = null, raf = 0, lastT = 0, acc = 0, visible = true;
    let held = null, feed = '';                        // the frame on show and where it came from ('sound' or 'signal')
    let loud = null, loudPk = -Infinity, soundOn = false, quiet = 0;
    let sigBuf = null, sigT = -1;

    if (root.IntersectionObserver) new IntersectionObserver(es => { visible = es.some(e => e.isIntersecting); if (visible) kick(); }).observe(spec);

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
    function frameMean(fr) {
      if (fr.mean !== undefined) return fr.mean;
      const fmax = fmaxOf(); let num = 0, den = 0;
      for (let k = 1; k < fr.db.length && k * fr.binHz <= fmax; k++) {
        if (fr.db[k] <= FLOOR) continue;
        const p = Math.pow(10, fr.db[k] / 10); num += k * fr.binHz * p; den += p;
      }
      return (fr.mean = den > 0 ? num / den : NaN);
    }
    const dbAt = (fr, f) => { let b = Math.round(f / fr.binHz); if (b > fr.db.length - 1) b = fr.db.length - 1; return fr.db[b]; };

    // The silent feed: FFT (Hann) of NSIG samples of the page's signal, sampled at about 2.6·fmax.
    function signalFrame(nowS) {
      const fmax = fmaxOf(), fs = 2.6 * fmax, n = NSIG;
      const x = opts.signal(nowS - n / fs, fs, n), re = new Float64Array(n), im = new Float64Array(n);
      let sw = 0;
      for (let i = 0; i < n; i++) { const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); re[i] = w * x[i]; sw += w; }
      fft(re, im);
      const db = new Float32Array(n / 2); let pk = 0;
      for (let k = 0; k < n / 2; k++) { const a = 2 * Math.hypot(re[k], im[k]) / sw; db[k] = a; if (a > pk) pk = a; }
      for (let k = 0; k < n / 2; k++) db[k] = pk > 0 && db[k] > 0 ? HI - 4 + 20 * Math.log10(db[k] / pk) : FLOOR - 10;
      return { db, binHz: fs / n, silent: pk === 0 };
    }

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
      // which feed is on
      const tag = feed === 'sound' ? '♪ what you hear' : running() ? 'live signal, silent (press ♪ to hear it)' : 'press ♪ to hear it';
      const mf = held ? frameMean(held) : NaN;
      ctx.textAlign = 'left'; ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillText(tag + (mf > 0 ? '   ·   white trace: mean frequency ⟨f⟩ = ' + fmtHz(mf) : ''), m.l + 6 * d, m.t + 5 * d);
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
      const fx = f => m.l + f / fmax * pw, fy = db => m.t + (1 - Math.max(0, Math.min(1, (db - FLOOR) / RANGE))) * ph;
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
      if (!held) { ctx.fillStyle = '#8a9a92'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('press ♪ to hear it — the live FFT runs while the sound plays', m.l + pw / 2, m.t + ph / 2); return; }
      ctx.beginPath();
      for (let ox = 0; ox <= pw; ox++) { const y = fy(dbAt(held, ox / pw * fmax)); ox ? ctx.lineTo(m.l + ox, y) : ctx.moveTo(m.l + ox, y); }
      ctx.strokeStyle = '#00693e'; ctx.lineWidth = 1.6 * d; ctx.lineJoin = 'round'; ctx.stroke();
      ctx.lineTo(m.l + pw, m.t + ph); ctx.lineTo(m.l, m.t + ph); ctx.closePath(); ctx.fillStyle = 'rgba(0,105,62,0.10)'; ctx.fill();
      const mf = frameMean(held);
      if (mf > 0 && mf <= fmax) { const x = fx(mf);
        ctx.strokeStyle = '#1b2a24'; ctx.lineWidth = 1.5 * d; ctx.beginPath(); ctx.moveTo(x, m.t + 12 * d); ctx.lineTo(x, m.t + ph); ctx.stroke();
        ctx.fillStyle = '#1b2a24'; ctx.font = 'bold ' + (11 * d) + 'px Helvetica, Arial, sans-serif'; ctx.textBaseline = 'top';
        ctx.textAlign = x > m.l + pw * 0.75 ? 'right' : 'left';
        ctx.fillText('⟨f⟩ = ' + fmtHz(mf), x + (x > m.l + pw * 0.75 ? -5 : 5) * d, m.t + 14 * d); }
    }

    // scroll left by k pixels and paint frame fr into the k new columns at the right
    function column(g, k, fr) {
      const c = off.getContext('2d'), fmax = fmaxOf();
      c.globalCompositeOperation = 'copy'; c.drawImage(off, -k, 0); c.globalCompositeOperation = 'source-over';
      const col = c.createImageData(1, g.ph);
      for (let y = 0; y < g.ph; y++) {
        const rgb = inferno((dbAt(fr, (1 - y / (g.ph - 1)) * fmax) - FLOOR) / RANGE);
        col.data[4 * y] = rgb[0]; col.data[4 * y + 1] = rgb[1]; col.data[4 * y + 2] = rgb[2]; col.data[4 * y + 3] = 255;
      }
      for (let j = 1; j <= k; j++) c.putImageData(col, g.pw - j, 0);
      const mf = frameMean(fr);   // the ⟨f⟩ trace, painted into the image so it scrolls with it
      if (mf > 0 && mf <= fmax) { c.fillStyle = 'rgba(255,255,255,0.95)'; c.fillRect(g.pw - k, Math.round((1 - mf / fmax) * (g.ph - 1)) - 1, k, 2); }
    }

    function loop(ts) {
      raf = 0;
      const g = geom(); ensureOff(g);
      const step = Math.min(0.1, Math.max(0, (ts - lastT) / 1000)); lastT = ts;
      let fr = null;
      if (soundOn) {
        an.getFloatFrequencyData(adata);
        let pk = -Infinity; for (let i = 0; i < adata.length; i++) if (adata[i] > pk) pk = adata[i];
        if (pk > FLOOR) { quiet = 0; fr = { db: adata.slice(), binHz: an.context.sampleRate / FFT }; feed = 'sound';
          if (pk > loudPk) { loudPk = pk; loud = fr; } }
        else if (++quiet > 30) { soundOn = false; feed = ''; if (loud) held = loud; }   // the sound has ended
      }
      if (!soundOn && running()) {
        if (ts - sigT > 33 || !sigBuf) { sigBuf = signalFrame(ts / 1000); sigT = ts; }   // at most ~30 FFTs a second
        fr = sigBuf; feed = 'signal';
      }
      if (fr) { held = fr; acc += step * g.pw / WINDOW; const k = Math.floor(acc); acc -= k; if (k > 0) column(g, k, fr); }
      paintSpec(g); paintLine();
      if (visible && !document.hidden && (soundOn || running())) raf = requestAnimationFrame(loop);
    }
    function kick() { if (!raf) { lastT = performance.now(); raf = requestAnimationFrame(loop); } }

    function input(actx) {
      if (!an || an.context !== actx) {
        an = actx.createAnalyser(); an.fftSize = FFT; an.smoothingTimeConstant = 0;
        an.minDecibels = FLOOR - 10; an.maxDecibels = HI;
        an.connect(actx.destination); adata = new Float32Array(an.frequencyBinCount);
      }
      soundOn = true; quiet = 0; loud = null; loudPk = -Infinity; kick();
      return an;
    }
    // redraw: repaint now and (re)start the silent feed if the page is running
    function redraw() { sigBuf = null; const g = geom(); ensureOff(g); paintSpec(g); paintLine(); if (running() || soundOn) kick(); }
    function clear() { off = null; held = null; redraw(); }
    document.addEventListener('visibilitychange', () => { if (!document.hidden) kick(); });
    requestAnimationFrame(redraw);   // first draw after the page's own script has finished setting up
    return { input, redraw, clear };
  }

  const api = { create, inferno, fft, meanFrequency };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LiveSpec = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
