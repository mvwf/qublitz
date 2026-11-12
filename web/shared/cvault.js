/* cvault.js: reads Coherence Vault (https://mvwf.github.io/CoherenceVault/), the MARQUIS database of published coherence data for
 * superconducting qubits and resonators, so Qublitz pages can set up real devices and wiring. Coherence Vault publishes its tables
 * as CSV with open cross-origin access; nothing is copied into this repository, and a page fetches them only when the
 * visitor asks (a button), so pages stay fast and work offline without it.
 *
 * Every value in Coherence Vault is the number the paper printed, with where it appears. At the time of writing no person has yet
 * checked them against the papers, so pages show NOTE next to anything they use, with a link to the source.
 *
 * Loads as a classic <script> (window.CVault) or from Node (require) for web/shared/tests/cvault.test.js.
 */
(function (root) {
  'use strict';
  const SITE = 'https://mvwf.github.io/CoherenceVault/', CSV = SITE + 'data/csv/';
  const NOTE = 'Transcribed from the paper by Coherence Vault and not yet checked by a curator: follow the link to check it.';

  // RFC 4180 CSV: quoted fields may hold commas, quotes ("") and line breaks
  function parseCSV(text) {
    const rows = []; let row = [], f = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"') q = true;
      else if (c === ',') { row.push(f); f = ''; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); rows.push(row); row = []; f = ''; }
      else f += c;
    }
    if (f !== '' || row.length) { row.push(f); rows.push(row); }
    const head = rows.shift() || [];
    return rows.filter(r => r.length > 1 || (r[0] || '') !== '').map(r => Object.fromEntries(head.map((h, k) => [h, r[k] == null ? '' : r[k]])));
  }
  const cache = {};
  const get = name => (typeof fetch === 'function' ? fetch(CSV + name, { cache: 'force-cache' }) : Promise.reject(new Error('no fetch')))
    .then(r => (r.ok ? r.text() : Promise.reject(new Error('HTTP ' + r.status))));
  // The tables were named qvault_*.csv while the database was called Q-Vault. Until every deployed copy of the site
  // serves the cvault_ names, a table that is missing under its new name is read under the old one.
  const OLD = name => name.replace(/^cvault_/, 'qvault_');
  function load(name) {
    if (!cache[name]) cache[name] = get(name).catch(e => (OLD(name) !== name ? get(OLD(name)) : Promise.reject(e))).then(parseCSV)
      .catch(e => { delete cache[name]; throw e; });
    return cache[name];
  }

  // ── citations ──
  function sourceMap(sources) { const m = {}; (sources || []).forEach(s => { m[s.citekey] = s; }); return m; }
  function cite(s, citekey) {
    if (!s) return citekey;
    const a = String(s.first_author || '').replace(/^and\s+/i, '').trim().split(/\s+/).pop() || citekey;
    return `${a}${s.year ? ' ' + s.year : ''}`;
  }
  const doiURL = s => (s && s.doi ? 'https://doi.org/' + s.doi : s && s.arxiv ? 'https://arxiv.org/abs/' + s.arxiv : '');
  const deviceURL = (citekey, dev, q) => `${SITE}#/compare?pins=${encodeURIComponent(citekey)}~${encodeURIComponent(dev || 'D1')}${q ? '&q=' + encodeURIComponent(q) : ''}`;

  // ── cryostat wiring (cvault_setup_parts.csv) ──
  // The attenuators on each drive line, top to bottom, with every dB stated. mappable: every attenuator sits on a stage the
  // Fridge Noise Explorer has (50 K, 4 K, still, mixing chamber), so the chain can be loaded into its five rows exactly.
  const ROLES = { input: 1, drive: 1, readout_input: 1 };
  const FRIDGE_STAGES = ['50K', '4K', 'still', 'mxc'];
  function chains(parts) {
    const by = {};
    (parts || []).forEach(p => {
      if (!ROLES[p.role] || p.part !== 'attenuator') return;
      const k = [p.citekey, p.setup_id, p.line].join('|');
      (by[k] = by[k] || { citekey: p.citekey, setup: p.setup_id, line: p.line, role: p.role, label: p.line_label || '', base_mK: +p.base_temperature_mK || null, parts: [], bad: false }).parts
        .push({ pos: +p.position || 0, stage: p.stage, dB: p.dB === '' ? NaN : +p.dB, count: +p.count || 1, loc: p.data_location });
    });
    return Object.values(by).map(c => {
      c.parts.sort((a, b) => a.pos - b.pos);
      c.bad = c.parts.some(p => !isFinite(p.dB) || p.stage === 'not_stated' || p.stage === 'sample' || p.stage === '');
      c.total = c.parts.reduce((s, p) => s + p.dB * p.count, 0);
      c.mappable = !c.bad && c.parts.every(p => FRIDGE_STAGES.includes(p.stage));
      c.perStage = {}; c.parts.forEach(p => { c.perStage[p.stage] = (c.perStage[p.stage] || 0) + p.dB * p.count; });
      return c;
    }).filter(c => !c.bad).sort((a, b) => (a.citekey < b.citekey ? -1 : a.citekey > b.citekey ? 1 : (a.setup + a.line < b.setup + b.line ? -1 : 1)));
  }
  // Thermal photons leaving the last attenuator of a chain, with the same model as fridge-engine.js: each attenuator of loss
  // L at temperature T passes n/L and adds (1 − 1/L)·n̄(T), starting from room-temperature noise. temps: { stage: kelvin }.
  function chainNoise(chain, F, temps, fGHz) {
    const T = [temps.room_temperature], A = [0];
    chain.parts.forEach(p => { for (let k = 0; k < p.count; k++) { T.push(temps[p.stage]); A.push(p.dB); } });
    const r = F.propagateChain([fGHz], T, A);
    return r.nEff[r.nEff.length - 1][0];
  }

  // ── resonator quality factors (cvault_measurements.csv) ──
  const QI_NAMES = { Qi_SP: 'single-photon Qᵢ', Qi_LP: 'low-power Qᵢ', Qi_HP: 'high-power Qᵢ' };
  function resonatorQs(meas) {
    return (meas || []).filter(m => /^resonator/.test(m.type) && QI_NAMES[m.quantity] && (m.status || 'current') === 'current' && isFinite(+m.value) && +m.value > 0)
      .map(m => ({ citekey: m.citekey, device: m.device_id, kind: m.quantity, value: +m.value, statistic: m.statistic, qualifier: m.qualifier || '',
        f: +m.freq_GHz || null, metal: m.base_metal, substrate: m.substrate, loc: m.data_location }))
      .sort((a, b) => b.value - a.value);
  }

  // ── links from Coherence Vault ("Qublitz" chips on its Parameters and Setups pages) ──
  // qv=citekey~device~label names one row of cvault_parameters.csv (or citekey~setup for a wiring). Only the ids are
  // trusted from the link; the values are read again from the live CSV, so a stale or edited link cannot put wrong numbers
  // on the page under a paper's name.
  function parseQV(s) {
    const [citekey, device, ...rest] = String(s || '').split('~');
    if (!/^[A-Za-z0-9_.:-]{2,100}$/.test(citekey || '')) return null;
    return { citekey, device: device || '', label: rest.join('~') };
  }
  function findParam(rows, qv) {
    if (!qv) return null;
    const same = r => r.citekey === qv.citekey && (!qv.device || r.device_id === qv.device);
    return (rows || []).find(r => same(r) && (r.label || '') === (qv.label || '')) || (qv.label ? null : (rows || []).find(same)) || null;
  }
  const paperURL = citekey => `${SITE}#/paper/${encodeURIComponent(citekey)}`;
  const paramsURL = (x, y, kind) => `${SITE}#/parameters?` + [x && 'x=' + x, y && 'y=' + y, kind && 'kind=' + kind].filter(Boolean).join('&');
  const num = v => (v === '' || v == null || !isFinite(+v) ? null : +v);

  // A real qubit in the Virtual Qubit Lab. The lab solves every GHz oscillation, so it runs on nanoseconds: a real qubit is
  // loaded with its own frequency and its own T2/T1, and with time running TIME_SCALE times faster (1 µs real = 1 ns here).
  const TIME_SCALE = 1000;
  function labQubit(p) {
    const f01 = num(p.f01), t1 = num(p.t1), t2 = num(p.t2), notes = [];
    if (!f01 || !t1 || t1 <= 0) return { ok: false, why: 'it needs both a qubit frequency and a T₁' };
    if (f01 < 1 || f01 > 10) return { ok: false, why: `its frequency, ${f01} GHz, is outside the lab's 1–10 GHz range${f01 < 1 ? ' (a fluxonium sits below 1 GHz: driven at the lab’s Rabi rates it would not behave like a two-level qubit)' : ''}` };
    let T1 = t1 * 1e3 / TIME_SCALE, T2;
    if (T1 > 1e4) { notes.push(`T₁ is capped at the lab's 10 µs (${t1} µs would be ${T1} ns here)`); T1 = 1e4; }
    if (t2 && t2 > 0) {
      T2 = t2 * 1e3 / TIME_SCALE * (T1 / (t1 * 1e3 / TIME_SCALE));
      if (T2 > 2 * T1) { notes.push(`the paper's T₂ (${t2} µs) is more than 2T₁, the most a qubit can have, so it is set to 2T₁`); T2 = 2 * T1; }
    } else { T2 = 2 * T1; notes.push('Coherence Vault lists no T₂ for it, so T₂ = 2T₁ (no pure dephasing)'); }
    const r = v => +v.toPrecision(4);
    return { ok: true, omegaQ: r(f01), T1: r(T1), T2: r(T2), notes };
  }

  // ── dispersive readout (Readout Lab): qubits with both a readout linewidth κ and a dispersive shift ──
  // χ is half the resonator's shift between |0⟩ and |1⟩: chi_MHz as printed, or two_chi_MHz / 2. When Coherence Vault's note on that
  // value says the paper does not state which convention it used, unsure is set and the page says so.
  function readoutRows(params, meas) {
    const unsure = new Set((meas || []).filter(m => (m.quantity === 'chi' || m.quantity === 'two_chi') && /does not say whether|may be th/i.test(m.notes || ''))
      .map(m => m.citekey + '|' + m.device_id));
    return (params || []).map(r => {
      const k = num(r.kappa_MHz), c = num(r.chi_MHz), c2 = num(r.two_chi_MHz);
      const chi = c != null ? Math.abs(c) : c2 != null ? Math.abs(c2) / 2 : null;
      if (!(k > 0) || !(chi > 0)) return null;
      return { citekey: r.citekey, device: r.device_id, label: r.label || '', type: r.type, kappa: k, chi, chiFrom: c != null ? 'chi' : 'two_chi',
        unsure: unsure.has(r.citekey + '|' + r.device_id), T1: num(r.T1_us), fr: num(r.f_readout_GHz), f01: num(r.f01_GHz), pexc: num(r.P_excited_percent) };
    }).filter(Boolean).sort((a, b) => (a.citekey < b.citekey ? -1 : a.citekey > b.citekey ? 1 : String(a.label).localeCompare(String(b.label), undefined, { numeric: true })));
  }

  // ── mystery qubits: every Virtual Qubit Lab code is a real published transmon ──
  // The pool: transmons with f₀₁ in the lab's default sweep window (3.5–6.5 GHz) and 30 µs ≤ T₁ ≤ 600 µs, so that on the
  // 1000× clock a π pulse stays short next to T₁ and a full T₁ decay fits in a 2000 µs run. Each code picks one by a hash
  // (sorted by id, so the same code gives the same qubit while Coherence Vault's table is unchanged) and a drive, since papers do not
  // list Rabi rates: Ω_R/2π between max(100, 5×10⁴/T₁[µs]) and 300 kHz on the chip (π pulse at most 1% of T₁).
  // Instructor links carry the chosen device as a token (the id XOR-ed with the code, base64url), so an answer key keeps
  // naming the same device as the table grows. It hides the device from a casual look, not from a determined student.
  const MYSTERY = { f01: [3.5, 6.5], T1: [30, 600] };
  function fnv(str) { let h = 0x811c9dc5; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h >>> 0; }
  function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const rowId = r => [r.citekey, r.device_id, r.label || ''].join('~');
  function mysteryPool(params) {
    return (params || []).filter(r => r.type === 'transmon' && num(r.f01_GHz) != null && num(r.T1_us) != null
      && +r.f01_GHz >= MYSTERY.f01[0] && +r.f01_GHz <= MYSTERY.f01[1] && +r.T1_us >= MYSTERY.T1[0] && +r.T1_us <= MYSTERY.T1[1])
      .sort((a, b) => (rowId(a) < rowId(b) ? -1 : rowId(a) > rowId(b) ? 1 : 0));
  }
  // The lab qubit for one pool row and code: f₀₁ exact, T₁ and T₂ in lab ns (= µs on the chip), the paper's T₂ (Ramsey, else
  // echo, else 2T₁), and the drive. Returns null if the row cannot run.
  function mysteryQubit(row, code) {
    const t2r = num(row.T2R_us), t2e = num(row.T2E_us), t2 = t2r > 0 ? t2r : t2e > 0 ? t2e : null;
    const L = labQubit({ f01: row.f01_GHz, t1: row.T1_us, t2 });
    if (!L.ok) return null;
    const r = mulberry(fnv('qublitz-real-drive:' + code)), lo = Math.min(300, Math.max(100, 5e4 / L.T1));
    const rabiMHz = Math.round(10 * (lo + (300 - lo) * r())) / 10;
    return { code, omegaQ: L.omegaQ, omegaRabi: rabiMHz / 1000, T1: L.T1, T2: L.T2,
      real: { citekey: row.citekey, device: row.device_id, label: row.label || '', t1: +row.T1_us, t2, kind: t2 ? (t2r > 0 ? 'Ramsey' : 'echo') : '', metal: row.base_metal || '', substrate: row.substrate || '' } };
  }
  function mysteryPick(pool, code) {
    if (!pool || !pool.length) return null;
    return pool[Math.floor(mulberry(fnv('qublitz-real:' + String(code).trim().toUpperCase()))() * pool.length)];
  }
  const b64u = { enc: bytes => (typeof btoa === 'function' ? btoa(String.fromCharCode(...bytes)) : Buffer.from(bytes).toString('base64')).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''),
    dec: s => { const t = s.replace(/-/g, '+').replace(/_/g, '/'); const bin = typeof atob === 'function' ? atob(t) : Buffer.from(t, 'base64').toString('binary'); return Array.from(bin, c => c.charCodeAt(0)); } };
  function mysteryToken(row, code) {
    const id = unescape(encodeURIComponent(rowId(row))), r = mulberry(fnv('qublitz-token:' + String(code).trim().toUpperCase()));
    return b64u.enc(Array.from(id, c => c.charCodeAt(0) ^ Math.floor(r() * 256)));
  }
  function mysteryFromToken(pool, token, code) {
    try {
      const r = mulberry(fnv('qublitz-token:' + String(code).trim().toUpperCase()));
      const id = decodeURIComponent(escape(String.fromCharCode(...b64u.dec(String(token)).map(b => b ^ Math.floor(r() * 256)))));
      return (pool || []).find(x => rowId(x) === id) || null;
    } catch (e) { return null; }
  }

  // ── real quantities: the spread of published values behind a number on a Qublitz page ──
  // src 'p': a column of cvault_parameters.csv (one row per qubit or resonator) for the listed device types.
  // src 'm': a quantity of cvault_measurements.csv (current rows; bounds such as "< 0.1 %" are left out, and rows whose notes match
  // skip, such as repetition-code Λ next to surface-code Λ).
  const RES = ['resonator_cpw', 'resonator_lumped', 'resonator_stripline'];
  const REAL = {
    'transmon.T1': { src: 'p', types: ['transmon'], col: 'T1_us', unit: 'µs', sym: 'T₁', who: 'transmons', x: 'f01_GHz', kind: 'qubits' },
    'transmon.T2E': { src: 'p', types: ['transmon'], col: 'T2E_us', unit: 'µs', sym: 'T₂ (echo)', who: 'transmons', x: 'T1_us', kind: 'qubits' },
    'transmon.T2R': { src: 'p', types: ['transmon'], col: 'T2R_us', unit: 'µs', sym: 'T₂* (Ramsey)', who: 'transmons', x: 'T1_us', kind: 'qubits' },
    'transmon.f01': { src: 'p', types: ['transmon'], col: 'f01_GHz', unit: 'GHz', sym: 'f₀₁', who: 'transmons', y: 'T1_us', kind: 'qubits' },
    'transmon.alpha': { src: 'p', types: ['transmon'], col: 'anharmonicity_MHz', unit: 'MHz', sym: '|α|/2π', who: 'transmons', abs: true, x: 'f01_GHz', kind: 'qubits' },
    'transmon.g': { src: 'p', types: ['transmon'], col: 'g_qr_MHz', unit: 'MHz', sym: 'g/2π', who: 'transmons', x: 'f_readout_GHz', kind: 'qubits' },
    // χ: half the resonator's shift between |0⟩ and |1⟩. Coherence Vault stores papers that print the full shift as two_chi; halve those.
    'transmon.chi': { src: 'p', types: ['transmon'], col: 'chi_MHz', alt: ['two_chi_MHz', 0.5], unit: 'MHz', sym: '|χ|/2π', who: 'transmons', abs: true, x: 'kappa_MHz', kind: 'qubits' },
    'transmon.kappa': { src: 'p', types: ['transmon'], col: 'kappa_MHz', unit: 'MHz', sym: 'κ/2π', who: 'readout resonators', x: 'f_readout_GHz', kind: 'qubits' },
    'transmon.fr': { src: 'p', types: ['transmon'], col: 'f_readout_GHz', unit: 'GHz', sym: 'readout frequency', who: 'transmons', x: 'f01_GHz', kind: 'qubits' },
    'fluxonium.T1': { src: 'p', types: ['fluxonium'], col: 'T1_us', unit: 'µs', sym: 'T₁', who: 'fluxoniums', x: 'f01_GHz', kind: 'qubits' },
    'resonator.Qi': { src: 'p', types: RES, col: 'Qi_SP_x', unit: '', sym: 'single-photon Qᵢ', who: 'planar resonators', x: 'f_res_GHz', kind: 'resonators' },
    'resonator.fr': { src: 'p', types: RES, col: 'f_res_GHz', unit: 'GHz', sym: 'frequency', who: 'planar resonators', y: 'Qi_SP_x', kind: 'resonators' },
    'cavity.T1': { src: 'p', types: ['cavity_3d'], col: 'T1_us', unit: 'µs', sym: 'photon lifetime', who: '3D cavities', x: 'f_res_GHz', kind: 'resonators' },
    'qubit.Teff': { src: 'm', quantity: 'T_eff', unit: 'mK', sym: 'effective temperature', who: 'qubits' },
    'qubit.Pexc': { src: 'm', quantity: 'P_excited', unit: '%', sym: 'leftover excited population', who: 'qubits' },
    'qubit.F1Q': { src: 'm', quantity: 'F1Q', unit: '%', sym: 'single-qubit gate fidelity', who: 'qubits' },
    'qubit.F2Q': { src: 'm', quantity: 'F2Q', unit: '%', sym: 'two-qubit gate fidelity', who: 'qubits' },
    'qubit.readout': { src: 'm', quantity: 'readout_error', unit: '%', sym: 'readout error', who: 'qubits' },
    'qec.Lambda': { src: 'm', quantity: 'QEC_Lambda', unit: '', sym: 'Λ (logical error suppression per distance step)', who: 'surface-code experiments', skip: /repetition code/i },
    'loss.tan': { src: 'm', quantity: 'tan_delta', unit: '', sym: 'loss tangent tan δ', who: 'devices' },
  };
  const FILE = { p: 'cvault_parameters.csv', m: 'cvault_measurements.csv' };
  function realValues(rows, spec) {
    if (spec.src === 'p') {
      const val = r => (num(r[spec.col]) != null ? +r[spec.col] : spec.alt && num(r[spec.alt[0]]) != null ? +r[spec.alt[0]] * spec.alt[1] : null);
      return (rows || []).filter(r => spec.types.includes(r.type) && val(r) != null)
        .map(r => ({ v: spec.abs ? Math.abs(val(r)) : val(r), citekey: r.citekey, device: r.device_id, label: r.label || '' }));
    }
    return (rows || []).filter(r => r.quantity === spec.quantity && (r.status || 'current') === 'current' && num(r.value) != null && !/[<>]/.test(r.qualifier || '') && !(spec.skip && spec.skip.test(r.notes || '')))
      .map(r => ({ v: +r.value, citekey: r.citekey, device: r.device_id, label: '' }));
  }
  // median and the middle half (25th to 75th percentile, linear interpolation), with how many values and papers
  function realStats(vals) {
    const v = vals.map(x => x.v).filter(isFinite).sort((a, b) => a - b), n = v.length;
    if (!n) return null;
    const q = p => { const i = (n - 1) * p, k = Math.floor(i); return k + 1 < n ? v[k] + (v[k + 1] - v[k]) * (i - k) : v[k]; };
    return { n, papers: new Set(vals.map(x => x.citekey)).size, median: q(0.5), lo: q(0.25), hi: q(0.75), min: v[0], max: v[n - 1] };
  }
  function fmt(v, unit) {
    if (!isFinite(v)) return '–';
    const a = Math.abs(v);
    const s = a >= 1e5 ? v.toExponential(1).replace(/e\+?(-?\d+)/, (m, e) => '×10' + String(e).replace(/-/, '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]))
      : a > 0 && a < 1e-2 ? v.toExponential(1).replace(/e(-?\d+)/, (m, e) => '×10' + e.replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]))
        : String(+v.toPrecision(unit === '%' && a >= 90 ? 4 : a >= 10 ? 3 : 2));   // fidelities need their nines
    return s + (unit ? (unit === '%' ? '%' : ' ' + unit) : '');
  }
  function realText(key, st) {
    const s = REAL[key];
    if (!st) return `No published ${s.sym} in Coherence Vault yet.`;
    // "3.6–4.7 GHz": the unit once, unless a power of ten would make the pair hard to read
    const pair = (a, b) => (/×/.test(fmt(a, '') + fmt(b, '')) ? `${fmt(a, s.unit)} to ${fmt(b, s.unit)}` : `${fmt(a, '')}–${fmt(b, s.unit)}`);
    const mid = st.n >= 4 ? `, middle half ${pair(st.lo, st.hi)}` : st.n > 1 ? `, range ${pair(st.min, st.max)}` : '';
    return `${st.n > 1 ? 'median ' : ''}${fmt(st.median, s.unit)}${mid} · ${st.n} value${st.n > 1 ? 's' : ''} from ${st.papers} paper${st.papers > 1 ? 's' : ''}`;
  }
  const realURL = key => { const s = REAL[key]; return s && s.src === 'p' ? paramsURL(s.x || s.col, s.y || s.col, s.kind) : `${SITE}#/explore`; };

  // Chips: <span data-qv-real="transmon.T1"></span> anywhere on a page becomes "📚 Real transmons: T₁ median 103 µs, …
  // · Coherence Vault ↗". The CSVs load once, when the first chip scrolls near the screen; offline, the chip is a plain link.
  const CHIP_CSS = '.qv-real{display:inline-flex;flex-wrap:wrap;gap:2px 6px;align-items:baseline;font-size:12.5px;line-height:1.4;' +
    'background:rgba(0,105,62,.07);border:1px solid rgba(0,105,62,.25);border-radius:7px;padding:2px 8px;margin:2px 0;color:inherit;max-width:100%}' +
    '.qv-real b{font-weight:600}.qv-real a{font-weight:600;white-space:nowrap;display:inline-block;padding:4px 2px;margin:-2px 0}.qv-real .qv-w{opacity:.8}';
  function chip(el, key, st) {
    const s = REAL[key]; if (!s) return;
    el.classList.add('qv-real');
    el.innerHTML = `<span class="qv-w">📚 Real ${s.who}:</span> <span><b>${s.sym}</b> ${st === undefined ? '…' : realText(key, st)}</span> ` +
      `<a href="${realURL(key)}" target="_blank" rel="noopener" title="See every value, with the paper it came from, in Coherence Vault">Coherence Vault ↗</a>`;
  }
  const valsCache = {};
  function realFor(key) {
    const s = REAL[key];
    if (!s) return Promise.reject(new Error('unknown quantity ' + key));
    return (valsCache[key] = valsCache[key] || load(FILE[s.src]).then(rows => realStats(realValues(rows, s))));
  }
  function mountChips(scope) {
    const doc = root.document; if (!doc) return;
    if (!doc.getElementById('qv-real-css')) { const st = doc.createElement('style'); st.id = 'qv-real-css'; st.textContent = CHIP_CSS; doc.head.append(st); }
    const els = [...(scope || doc).querySelectorAll('[data-qv-real]:not([data-qv-done])')];
    const fill = el => { const key = el.dataset.qvReal; realFor(key).then(st => chip(el, key, st)).catch(() => {
      el.classList.add('qv-real'); el.innerHTML = `<span class="qv-w">📚 Real ${REAL[key] ? REAL[key].who : 'devices'}:</span> <a href="${realURL(key)}" target="_blank" rel="noopener">see published values in Coherence Vault ↗</a>`; }); };
    const io = 'IntersectionObserver' in root ? new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.unobserve(e.target); fill(e.target); } }), { rootMargin: '300px' }) : null;
    els.forEach(el => { el.dataset.qvDone = '1'; chip(el, el.dataset.qvReal, undefined); if (io) io.observe(el); else fill(el); });
  }
  if (root.document) {
    const start = () => {
      mountChips();
      // pages that draw panels later (tabs, depth cards) get their chips too
      let t = 0; new MutationObserver(() => { clearTimeout(t); t = setTimeout(() => mountChips(), 60); }).observe(root.document.body, { childList: true, subtree: true });
    };
    if (root.document.readyState === 'loading') root.document.addEventListener('DOMContentLoaded', start); else start();
  }

  // The source line under anything loaded from Coherence Vault: paper, Coherence Vault page, the caveat.
  function sourceHTML(s, citekey, extra) {
    const url = doiURL(s);
    return `${cite(s, citekey)}${s && s.title ? `, <i>${String(s.title).replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' })[c]).slice(0, 120)}</i>` : ''}${extra ? ' · ' + extra : ''}. ` +
      `${url ? `<a href="${url}" target="_blank" rel="noopener">Paper</a> · ` : ''}<a href="${paperURL(citekey)}" target="_blank" rel="noopener">In Coherence Vault</a>. ${NOTE}`;
  }

  const api = { SITE, CSV, NOTE, parseCSV, load, sourceMap, cite, doiURL, deviceURL, chains, chainNoise, resonatorQs, QI_NAMES, FRIDGE_STAGES,
    parseQV, findParam, paperURL, paramsURL, labQubit, readoutRows, MYSTERY, mysteryPool, mysteryQubit, mysteryPick, mysteryToken, mysteryFromToken, TIME_SCALE, REAL, realValues, realStats, realText, realFor, realURL, fmt, mountChips, sourceHTML };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CVault = api;
})(typeof window !== 'undefined' ? window : globalThis);
