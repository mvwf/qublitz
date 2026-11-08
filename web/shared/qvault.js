/* qvault.js: reads Q-Vault (https://mvwf.github.io/MARQUIS_Q_VAULT/), the MARQUIS database of published coherence data for
 * superconducting qubits and resonators, so Qublitz pages can set up real devices and wiring. Q-Vault publishes its tables
 * as CSV with open cross-origin access; nothing is copied into this repository, and a page fetches them only when the
 * visitor asks (a button), so pages stay fast and work offline without it.
 *
 * Every value in Q-Vault is the number the paper printed, with where it appears. At the time of writing no person has yet
 * checked them against the papers, so pages show NOTE next to anything they use, with a link to the source.
 *
 * Loads as a classic <script> (window.QVault) or from Node (require) for web/shared/tests/qvault.test.js.
 */
(function (root) {
  'use strict';
  const SITE = 'https://mvwf.github.io/MARQUIS_Q_VAULT/', CSV = SITE + 'data/csv/';
  const NOTE = 'Transcribed from the paper by Q-Vault and not yet checked by a curator: follow the link to check it.';

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
  function load(name) {
    if (!cache[name]) cache[name] = (typeof fetch === 'function' ? fetch(CSV + name, { cache: 'force-cache' }) : Promise.reject(new Error('no fetch')))
      .then(r => (r.ok ? r.text() : Promise.reject(new Error('HTTP ' + r.status)))).then(parseCSV)
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

  // ── cryostat wiring (qvault_setup_parts.csv) ──
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

  // ── resonator quality factors (qvault_measurements.csv) ──
  const QI_NAMES = { Qi_SP: 'single-photon Qᵢ', Qi_LP: 'low-power Qᵢ', Qi_HP: 'high-power Qᵢ' };
  function resonatorQs(meas) {
    return (meas || []).filter(m => /^resonator/.test(m.type) && QI_NAMES[m.quantity] && (m.status || 'current') === 'current' && isFinite(+m.value) && +m.value > 0)
      .map(m => ({ citekey: m.citekey, device: m.device_id, kind: m.quantity, value: +m.value, statistic: m.statistic, qualifier: m.qualifier || '',
        f: +m.freq_GHz || null, metal: m.base_metal, substrate: m.substrate, loc: m.data_location }))
      .sort((a, b) => b.value - a.value);
  }

  const api = { SITE, CSV, NOTE, parseCSV, load, sourceMap, cite, doiURL, deviceURL, chains, chainNoise, resonatorQs, QI_NAMES, FRIDGE_STAGES };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.QVault = api;
})(typeof window !== 'undefined' ? window : globalThis);
