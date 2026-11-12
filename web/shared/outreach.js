/* outreach.js: anonymous usage counts, learning panels and end-of-activity quizzes for every page.
 * No dependencies; scripts/build_standalone.py inlines it into the offline downloads (which never send anything).
 *
 * Include, after outreach-config.js and ../learning/manifest.js:
 *   <script src="../shared/outreach-config.js"></script>
 *   <script src="../learning/manifest.js"></script>
 *   <script src="../shared/outreach.js"></script>
 *   <script src="../learning/<id>.js"></script>      (the page's objectives, instructions and quiz)
 *
 * WHAT IS SENT, AND WHEN. Nothing at all unless QL_CONFIG.endpoint is set, the page is served over http(s) (not
 * the offline download), the visitor has not opted out, and the browser does not send Do Not Track or Global
 * Privacy Control. Then only:
 *   - events: { game, ev, level?, value?, stars?, dur_s?, code?, dev, day }, e.g. "beamline opened",
 *     "beamline level 'half' solved with 3 stars", "rider left after 412 s". No cookies, no identifiers, no IP
 *     address (Apps Script never sees it), no timestamps finer than the day.
 *   - quiz: which choice was picked for each question, whether it was right, and two optional 1-5 ratings (fun,
 *     learned). There is NO free text: nothing a visitor types is collected, so no one can hand over a name,
 *     school or email this way. See web/privacy.html.
 *   `code` is an optional outreach-session label from the link (?event=SCIFAIR26), so a dashboard can split
 *   counts by event. It is kept for the browser tab only (sessionStorage).
 *
 * API (window.QL):
 *   QL.event(name, data)        name: lower_snake (open, start, level_start, level_complete, win, lose, ...)
 *                               data: { level, value, stars, dur_s } (all optional), plus ANY other keys a game
 *                               wants to log (e.g. { theta_pi: 0.5, beam: 'x', hinted: true }). Extra keys go to
 *                               a flexible `x` object: up to 12 lower_snake keys, each a number, true/false or a
 *                               single-word label (letters, digits, _ . : | + -, at most 40 characters, no spaces).
 *                               Anything else is dropped, so no sentence or typed input can be stored. Never pass
 *                               anything a visitor typed. New event names and keys need no collector change.
 *   Every event and question record also carries `gv`, the version of the page that sent it (the date and commit
 *   of the last change to its own files), stamped automatically at deploy time, so analyses can compare a game
 *   before and after an update.
 *   QL.learning(spec)           register this page's learning material:
 *       { id, title, tagline, fun, objectives: [{ id, text }], howTo: [text], goals: [text],
 *         quiz: [{ id, objective, q, choices: [text], answer: index, explain }] }
 *   QL.panel(el, opts)          render the "What you'll learn / How to play / Goals" panel into el
 *   QL.context({ level, step }) where the player is now; the "leave" event reports the last place, so the
 *                               dashboard can show where people stop (set it on every level/step change)
 *   QL.answer(itemId, pick)     record an answer given INSIDE the game (fly through a ring, shoot a target...)
 *                               for quiz item itemId; returns { ok, answer, choice, explain } so the game can
 *                               react. Items answered in-game are skipped by the debrief.
 *   QL.addItems([item, ...])    add question items at runtime (e.g. a guided step's prediction), same shape as
 *                               spec.quiz entries; duplicates by id are ignored
 *   QL.answered(itemId)         true once that item has been answered in this page load
 *   QL.showQuiz()               open the debrief: a few unanswered items and two 1-5 ratings (no free text)
 *   QL.offerQuiz(message)       a small prompt that opens the debrief (once per page load)
 *   QL.labLink(query)           URL of the Virtual Qubit Lab with ?from=<this page> (plus your query)
 *   QL.toLab(anchor, query)     the same, set on a link, counting the click as a to_lab event
 *   QL.challenges(list, opts)   live goals for an explore page: a small tracker ("Challenges 1/3") that ticks each
 *                               one off from the page's own state. list = [{ id, text, hint?, check?() }], where
 *                               check() returns true once done (polled ~3×/s, errors count as not done; omit it and
 *                               call QL.challengeDone(id) instead). Done ones are remembered in this browser only.
 *                               The current one sends level_start and each completion a level_complete, both
 *                               with level 'c:<id>' (value = how many are done), so the dashboard's level table shows how far people get. The hint
 *                               appears after opts.hintAfter seconds (default 40) on the current challenge.
 *                               opts: { title, hintAfter, done(message) } — the final one offers the debrief.
 *   QL.challengeDone(id)        mark a challenge done from page code (for things a check() cannot see, e.g. a sound played)
 *   QL.submitDesign(rec)       send an ErrorSweeper code design to the frontier board: rec = { mission, family
 *                               ('lattice'|'bb'), n, k, d, dExact, score, spec }; numbers and a layout string only.
 *                               Returns true if sent (never from test hosts, opted-out or offline copies)
 *   QL.frontier(mission)        Promise of the board for a mission: [{ spec, family, n, k, d, dExact, score, day,
 *                               count }], best first; [] when there is no collector or on any error
 *   QL.submitMap({ edges })    send a concept map the visitor chose to share: 'a>b:rel;…' with ids and relations from
 *                               web/learning/concepts.js only (anything else is dropped); returns true if sent
 *   QL.mapBoard()               Promise of the combined map from everyone's shared maps, or null
 *   QL.enabled(), QL.optOut(on), QL.code()
 *
 * DEPTH. One site-wide choice of how deep to go: curious · student (default) · engineer · researcher, picked in the site bar,
 *   by a link (?depth=engineer, remembered) or by QL.setDepth(d). It is stored in this browser only (localStorage
 *   'qublitz_depth', shared with Q-Vault on the same origin) and sent with each event as the one-word label x.depth.
 *   Tag any element with data-depth="engineer+" (that depth and deeper), "-student" (up to student), "curious" (only
 *   there) or several tokens separated by spaces; untagged content shows at every depth. Learning-file objectives,
 *   quiz items, challenges, and howTo/goals entries written as { text, depth } take the same tag in a `depth` field.
 *   QL.depth() → current depth · QL.setDepth(d) · QL.onDepth(fn) (called with the new depth) · QL.atDepth(tag) → bool
 *   QL.progress(id) → { explored, done, total } for a page id, from this browser's own records (for the course map)
 */
(function (root) {
  'use strict';
  const V = 1;
  const MAX_TEXT = 600, MAX_EVENTS = 300;
  const ID = /^[a-z0-9-]{1,32}$/, EV = /^[a-z_]{1,32}$/, CODE = /^[A-Za-z0-9-]{1,24}$/;

  // ── Sanitising (pure; also used by the collector and the tests) ──────────────────────────────────────
  function redact(text) {
    if (typeof text !== 'string') return '';
    return text
      .replace(/[\w.+-]+@[\w-]+(\.[\w-]+)+/g, '[removed]')                     // email addresses
      .replace(/(\+?\d[\d\s().-]{7,}\d)/g, '[removed]')                          // phone-like digit runs
      .replace(/https?:\/\/\S+/g, '[link]')
      .replace(/\s+/g, ' ').trim().slice(0, MAX_TEXT);
  }
  const num = (v, lo, hi) => (typeof v === 'number' && isFinite(v) && v >= lo && v <= hi) ? Math.round(v * 1000) / 1000 : undefined;
  const str = (v, n) => (typeof v === 'string' && v.length) ? v.replace(/[^\w .:/|+-]/g, '').slice(0, n) : undefined;
  // Extra, game-specific fields: numbers, booleans and single-word labels only (never free text).
  const XKEY = /^[a-z][a-z0-9_]{0,31}$/, XTOK = /^[A-Za-z0-9_.:|+-]{1,40}$/, GV = /^[A-Za-z0-9_.:+-]{1,40}$/;
  const RESERVED = { level: 1, value: 1, stars: 1, dur_s: 1 };
  // Anything that could identify a person is refused outright: identifier-like field names, and labels that look
  // like an IP address or a long number (a phone or student number).
  const XBADKEY = /(^|_)(ids?|uid|uuid|guid|ip|user|username|name|email|mail|phone|session|token|device|fingerprint|address|school|student)(_|$)/, XBADVAL = /^\d{1,3}(\.\d{1,3}){3}$|\d{6,}/;
  function cleanExtra(d) {
    const x = {}; let n = 0;
    for (const k of Object.keys(d || {})) {
      if (RESERVED[k] || !XKEY.test(k) || XBADKEY.test(k) || n >= 12) continue;
      const v = d[k];
      if (typeof v === 'boolean') x[k] = v;
      else if (typeof v === 'number' && isFinite(v) && Math.abs(v) <= 1e9) x[k] = Math.round(v * 1e4) / 1e4;
      else if (typeof v === 'string' && XTOK.test(v) && !XBADVAL.test(v)) x[k] = v;
      else continue;
      n++;
    }
    return n ? x : null;
  }
  function cleanEvent(game, ev, data, ctx) {
    if (!ID.test(game || '') || !EV.test(ev || '')) return null;
    const d = data || {}, c = ctx || {};
    const out = { game, ev, day: c.day, dev: c.dev === 'touch' ? 'touch' : 'desktop' };
    const level = str(d.level == null ? undefined : String(d.level), 40); if (level) out.level = level;
    const value = num(d.value, -1e6, 1e6); if (value !== undefined) out.value = value;
    const stars = num(d.stars, 0, 3); if (stars !== undefined) out.stars = Math.round(stars);
    const dur = num(d.dur_s, 0, 86400); if (dur !== undefined) out.dur_s = Math.round(dur);
    const x = cleanExtra(d); if (x) out.x = x;
    if (c.code && CODE.test(c.code)) out.code = c.code;
    if (c.gv && GV.test(c.gv)) out.gv = c.gv;
    return out;
  }
  function cleanQuiz(game, rec, ctx) {
    if (!ID.test(game || '') || !rec) return null;
    const c = ctx || {};
    const answers = (rec.answers || []).filter(a => a && typeof a.pick === 'number' && a.pick >= 0).slice(0, 20).map(a => ({
      q: str(String(a.q), 40), obj: str(String(a.obj || ''), 40), pick: num(a.pick, -1, 20), ok: !!a.ok }));
    const out = { kind: 'quiz', game, day: c.day, dev: c.dev === 'touch' ? 'touch' : 'desktop', answers,
      score: answers.filter(a => a.ok).length, n: answers.length,
      fun: num(rec.fun, 1, 5), learned: num(rec.learned, 1, 5),
      src: rec.src === 'ingame' ? 'ingame' : 'debrief' };   // no free-text fields, by design
    if (c.code && CODE.test(c.code)) out.code = c.code;
    if (c.gv && GV.test(c.gv)) out.gv = c.gv;
    return out;
  }

  // A code design for the frontier board: numbers, a mission id and a compact layout string only (no free text).
  //   rec = { mission, family: 'lattice'|'bb', n, k, d, dExact, score, spec }
  const MISSION = /^[a-z0-9-]{1,32}$/, SPEC = /^[0-9A-Za-z.,;:|+=-]{1,2000}$/;
  const int = (v, lo, hi) => (typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi) ? v : undefined;
  function cleanDesign(rec, ctx) {
    const r = rec || {}, c = ctx || {};
    if (!MISSION.test(r.mission || '') || (r.family !== 'lattice' && r.family !== 'bb')) return null;
    const n = int(r.n, 1, 2000); if (n === undefined) return null;
    const k = int(r.k, 0, n); if (k === undefined) return null;
    const d = int(r.d, 1, 200); if (d === undefined && k > 0) return null;
    if (typeof r.score !== 'number' || !isFinite(r.score) || r.score < 0 || r.score > 1e6) return null;
    if (typeof r.spec !== 'string' || !SPEC.test(r.spec)) return null;
    const out = { mission: r.mission, family: r.family, n, k, dExact: !!r.dExact, score: Math.round(r.score * 1e4) / 1e4, spec: r.spec,
      day: c.day, dev: c.dev === 'touch' ? 'touch' : 'desktop' };
    if (d !== undefined && k > 0) out.d = d;
    if (c.gv && GV.test(c.gv)) out.gv = c.gv;
    return out;
  }

  // A concept map (web/concepts.html): links between ids from the fixed vocabulary in web/learning/concepts.js, each with a
  // relation from its fixed list. Nothing typed can get in: unknown ids and relations are dropped. edges = 'a>b:rel;…'.
  const MAX_MAP_EDGES = 80, MAP_ID = /^[a-z0-9-]{1,24}$/;
  function cleanMap(rec, ctx, ids, rels) {
    const r = rec || {}, c = ctx || {}, idSet = new Set(ids || []), relSet = new Set(rels || []);
    if (typeof r.edges !== 'string' || r.edges.length > 4000) return null;
    const seen = new Set(), out = [];
    for (const tok of r.edges.split(';')) {
      const m = /^([a-z0-9-]{1,24})>([a-z0-9-]{1,24}):([a-z-]{1,12})$/.exec(tok);
      if (!m || m[1] === m[2] || !idSet.has(m[1]) || !idSet.has(m[2]) || !relSet.has(m[3])) continue;
      const key = m[1] < m[2] ? m[1] + '|' + m[2] : m[2] + '|' + m[1];
      if (seen.has(key)) continue;
      seen.add(key); out.push(`${m[1]}>${m[2]}:${m[3]}`);
      if (out.length >= MAX_MAP_EDGES) break;
    }
    if (!out.length) return null;
    const res = { edges: out.join(';'), n: out.length, day: c.day, dev: c.dev === 'touch' ? 'touch' : 'desktop' };
    if (DEPTHS.includes(r.depth)) res.depth = r.depth;
    if (c.code && CODE.test(c.code)) res.code = c.code;
    if (c.gv && GV.test(c.gv)) res.gv = c.gv;
    return res;
  }

  // Test traffic never counts: pages served from this machine or driven by an automated browser (tests, crawlers that
  // announce themselves) send nothing, so the statistics are real visitors only.
  function isTestHost(hostname, webdriver) {
    const h = String(hostname || '').toLowerCase();
    return !!webdriver || h === 'localhost' || h === '[::1]' || h === '::1' || /^127\./.test(h) || h === '0.0.0.0' || /\.local$/.test(h) || h === '';
  }
  // ── Depth (pure) ───────────────────────────────────────────────────────────────────────────────────────
  const DEPTHS = ['curious', 'student', 'engineer', 'researcher'];
  // tag: tokens separated by spaces: 'engineer' (only), 'engineer+' (and deeper), '-student' (up to student). No tag
  // (or no valid token) means every depth, so untagged content never disappears.
  function depthMatch(tag, d) {
    const i = DEPTHS.indexOf(d); if (i < 0 || tag == null || tag === '') return true;
    const toks = (Array.isArray(tag) ? tag : String(tag).trim().split(/\s+/)).map(t => /^(-?)(curious|student|engineer|researcher)(\+?)$/.exec(t)).filter(Boolean);
    if (!toks.length) return true;
    return toks.some(m => { const j = DEPTHS.indexOf(m[2]); return m[1] ? i <= j : m[3] ? i >= j : i === j; });
  }

  // The debrief asks a few questions, not all of them: unanswered items first, one per learning objective before
  // a second on any objective, in random order. rand() in [0, 1).
  function pickQuiz(items, n, skip, rand) {
    const R = rand || Math.random, pool = items.filter(q => !(skip && skip.has(q.id)));
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    const out = [], seen = new Set();
    for (const q of pool) if (out.length < n && !seen.has(q.objective)) { out.push(q); seen.add(q.objective); }
    for (const q of pool) if (out.length < n && !out.includes(q)) out.push(q);
    return out;
  }

  if (typeof root.document === 'undefined') {           // Node: export the pure parts for the tests
    if (typeof module !== 'undefined') module.exports = { redact, cleanEvent, cleanQuiz, cleanExtra, cleanDesign, cleanMap, isTestHost, pickQuiz, depthMatch, DEPTHS };
    return;
  }

  // ── Browser ──────────────────────────────────────────────────────────────────────────────────────────
  const doc = root.document;
  const CFG = root.QL_CONFIG || {};
  const store = {
    get(k) { try { return root.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { root.localStorage.setItem(k, v); } catch (e) {} },
  };
  const sess = {
    get(k) { try { return root.sessionStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { root.sessionStorage.setItem(k, v); } catch (e) {} },
  };
  const offline = !!root.QUBLITZ_STANDALONE || location.protocol === 'file:';
  const dnt = () => navigator.doNotTrack === '1' || root.doNotTrack === '1' || navigator.globalPrivacyControl === true;
  const optedOut = () => store.get('ql_optout') === '1';
  const testHost = isTestHost(location.hostname, navigator.webdriver);
  const enabled = () => !!CFG.endpoint && !offline && !testHost && !optedOut() && !dnt();

  // Outreach-session label from ?event=, remembered for this tab so it follows the visitor across pages.
  (function () {
    const m = /[?&]event=([A-Za-z0-9-]{1,24})/.exec(location.search);
    if (m) sess.set('ql_code', m[1]);
  })();
  const code = () => sess.get('ql_code') || '';
  const day = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; };
  const dev = () => (root.matchMedia && root.matchMedia('(pointer: coarse)').matches) ? 'touch' : 'desktop';
  // Page version, stamped into outreach-config.js by the Pages build (scripts/stamp_build.py): the date and
  // commit of the last change to this page's own files, e.g. '2026-11-05+1a2b3c4'. Falls back to the site commit.
  let pagePath = '';
  const gv = () => {
    const B = root.QUBLITZ_BUILD; if (!B) return '';
    return (B.pages && B.pages[pagePath]) || B.sha || '';
  };
  const ctx = () => ({ day: day(), dev: dev(), code: code(), gv: gv() });

  // ── Depth: how deep this visitor wants to go (see DEPTH above) ──────────────────────────────────────────
  const DEPTH_KEY = 'qublitz_depth';
  const DEPTH_LABEL = { curious: '🌱 Curious', student: '🎓 Student', engineer: '🔧 Engineer', researcher: '🔬 Researcher' };
  let curDepth = (() => {
    const m = /[?&]depth=(curious|student|engineer|researcher)(?![a-z])/.exec(location.search);
    if (m) { store.set(DEPTH_KEY, m[1]); return m[1]; }
    const v = store.get(DEPTH_KEY); return DEPTHS.includes(v) ? v : 'student';
  })();
  const depthSubs = [];
  doc.documentElement.setAttribute('data-depth', curDepth);
  (function () {
    const toks = []; DEPTHS.forEach(d => toks.push(d, d + '+', '-' + d));
    let css = '';
    DEPTHS.forEach(d => toks.forEach(t => { if (!depthMatch(t, d)) css += `html[data-depth="${d}"] [data-depth="${t}"]{display:none !important}`; }));
    css += '.ql-depth{display:inline-flex;align-items:center;gap:5px;font:inherit;color:inherit}.ql-depth select{font:inherit;font-size:12.5px;border:1px solid currentColor;border-radius:999px;padding:2px 8px;background:transparent;color:inherit;cursor:pointer}' +
      '.ql-depth select option{color:#1b2a24;background:#fff}@media (pointer: coarse){.ql-depth select{min-height:36px}}';
    const st = doc.createElement('style'); st.id = 'ql-depth-css'; st.textContent = css; (doc.head || doc.documentElement).appendChild(st);
  })();
  function depth() { return curDepth; }
  function atDepth(tag) { return depthMatch(tag, curDepth); }
  function onDepth(fn) { if (typeof fn === 'function') depthSubs.push(fn); }
  function setDepth(d, quiet) {
    if (!DEPTHS.includes(d) || d === curDepth) return;
    curDepth = d; doc.documentElement.setAttribute('data-depth', d);
    if (!quiet) store.set(DEPTH_KEY, d);
    doc.querySelectorAll('.ql-depth select').forEach(s => { s.value = d; });
    if (panelBox) panel(panelHost, Object.assign({}, panelOpts, { _swap: true }));
    if (chAll.length) { chList = chAll.filter(c => depthMatch(c.depth, curDepth)); chDraw(); }
    depthSubs.forEach(f => { try { f(d); } catch (e) { /* a page's handler must not break the others */ } });
    if (!quiet) event('depth', { level: d });
  }
  // another tab (or Q-Vault) changed it
  root.addEventListener('storage', e => { if (e.key === DEPTH_KEY && DEPTHS.includes(e.newValue)) setDepth(e.newValue, true); });
  // the picker sits in the site bar, next to the theme switch
  function mountDepthPicker() {
    const bar = doc.querySelector('.qsitebar'); if (!bar || bar.querySelector('.ql-depth')) return;
    const lab = doc.createElement('label'); lab.className = 'ql-depth'; lab.title = 'How deep to go: pages show more equations, data and open problems at greater depth';
    lab.innerHTML = '<span>Depth</span><select aria-label="Depth">' + DEPTHS.map(d => `<option value="${d}"${d === curDepth ? ' selected' : ''}>${DEPTH_LABEL[d]}</option>`).join('') + '</select>';
    lab.querySelector('select').addEventListener('change', e => setDepth(e.target.value));
    const theme = bar.querySelector('.rt-wrap');
    if (theme) theme.before(lab); else bar.append(lab);
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', () => setTimeout(mountDepthPicker, 0)); else setTimeout(mountDepthPicker, 0);
  root.addEventListener('load', mountDepthPicker);
  // progress on a page, from this browser's own records: opened it? how many of its challenges are done?
  function progress(id) {
    let done = 0, total = 0;
    try { const a = JSON.parse(store.get(`ql_ch:${id}`) || '[]'); done = Array.isArray(a) ? a.length : 0; } catch (e) { /* none */ }
    total = +store.get(`ql_ch:${id}:n`) || 0;
    return { explored: explored().includes(id), done, total };
  }

  // Which page is this? Match the page path against the manifest (the hub is '').
  function detectGame() {
    const me = doc.currentScript, list = root.QL_GAMES || [];
    let path = root.QUBLITZ_PATH;
    if (path == null && me && me.src) {
      const base = me.src.replace(/shared\/outreach\.js(?:[?#].*)?$/, '');
      path = location.href.startsWith(base) ? location.href.slice(base.length).replace(/[?#].*$/, '') : '';
    }
    path = (path || '').replace(/^index\.html$/, '');
    pagePath = path;
    const hit = list.find(g => g.path === path);
    return hit ? hit.id : null;
  }
  let game = detectGame(), spec = null, queue = [], sentCount = 0;

  // Which activities this browser has opened, for the hub's "explored" badges. Kept in this browser only
  // (localStorage); never sent anywhere.
  const EXPLORED = 'qublitz_explored';
  function explored() { try { const a = JSON.parse(store.get(EXPLORED) || '[]'); return Array.isArray(a) ? a.filter(x => ID.test(x)) : []; } catch (e) { return []; } }
  function markExplored(id) {
    if (!id || id === 'hub' || !ID.test(id)) return;
    const a = explored(); if (!a.includes(id)) { a.push(id); store.set(EXPLORED, JSON.stringify(a)); }
  }
  setTimeout(() => markExplored(game), 0);

  function send(payload, beacon) {
    if (!enabled()) return false;
    const body = JSON.stringify(payload);
    try {
      if (beacon && navigator.sendBeacon) return navigator.sendBeacon(CFG.endpoint, new Blob([body], { type: 'text/plain' }));
      fetch(CFG.endpoint, { method: 'POST', mode: 'no-cors', keepalive: true, headers: { 'Content-Type': 'text/plain' }, body }).catch(() => {});
      return true;
    } catch (e) { return false; }
  }
  function flush(beacon) {
    if (!queue.length) return;
    const batch = queue.splice(0, 50);
    send({ v: V, events: batch }, beacon);
    if (queue.length) flush(beacon);
  }
  // Where the player is (for the quit map) and how far they got this visit.
  let place = '', completedThisVisit = 0;
  function context(c) {
    const d = c || {};
    place = str([d.level, d.step].filter(v => v != null && v !== '').map(String).join('|'), 40) || '';
  }
  function event(name, data) {
    // A new level resets the place, but a page that already set this level's step keeps it.
    if (name === 'level_start' && data && data.level != null && place.split('|')[0] !== String(data.level)) context({ level: data.level });
    if (name === 'level_complete') completedThisVisit++;
    if (!enabled() || !game || sentCount >= MAX_EVENTS) return;
    const e = cleanEvent(game, name, Object.assign({ depth: curDepth }, data), ctx());
    if (!e) return;
    queue.push(e); sentCount++;
    if (queue.length >= 20) flush(false);
  }
  setInterval(() => flush(false), 15000);

  // Active time on the page (paused while the tab is hidden), reported once when the visitor leaves.
  let activeMs = 0, since = doc.visibilityState === 'hidden' ? null : performance.now(), leftSent = false;
  doc.addEventListener('visibilitychange', () => {
    if (doc.visibilityState === 'hidden') {
      if (since != null) { activeMs += performance.now() - since; since = null; }
      flush(true);
    } else since = performance.now();
  });
  root.addEventListener('pagehide', () => {
    if (since != null) { activeMs += performance.now() - since; since = null; }
    if (!leftSent) { leftSent = true; event('leave', { dur_s: activeMs / 1000, level: place || 'menu', value: completedThisVisit }); }
    flush(true);
  });

  // ── Styles ───────────────────────────────────────────────────────────────────────────────────────────
  const css = doc.createElement('style');
  css.textContent = `
.ql-panel{border:1px solid #d8e2dc;border-radius:10px;background:#f5f9f6;color:#1b2a24;padding:12px 16px;margin:10px 0;font:14px/1.5 Helvetica,"Helvetica Neue",Arial,sans-serif}
.ql-panel .ql-h{margin:0 0 4px;font-size:15px;font-weight:700;text-transform:none;letter-spacing:normal;color:#00693e}.ql-panel .ql-tag{color:#5a6b63;margin:0 0 8px}
.ql-panel .ql-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:6px 22px}
.ql-panel .ql-sh{margin:6px 0 2px;font-size:12.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#5a6b63}
.ql-panel ul,.ql-panel ol{margin:0;padding-left:20px}.ql-panel li{margin:2px 0}
.ql-panel .ql-fun{margin:8px 0 0;color:#5a6b63}
.ql-btn{font:600 13.5px Helvetica,Arial,sans-serif;border:1px solid #00693e;background:#00693e;color:#fff;border-radius:7px;padding:7px 14px;cursor:pointer}
.ql-btn.ql-ghost{background:#fff;color:#00693e}.ql-btn:focus-visible{outline:3px solid #ffd166;outline-offset:2px}
.ql-veil{position:fixed;inset:0;background:rgba(11,18,32,.55);display:flex;align-items:flex-start;justify-content:center;z-index:99999;overflow:auto;padding:24px 12px}
.ql-dlg{background:#fff;color:#1b2a24;max-width:640px;width:100%;border-radius:12px;padding:18px 20px;box-shadow:0 12px 40px rgba(0,0,0,.35);font:14.5px/1.5 Helvetica,"Helvetica Neue",Arial,sans-serif}
.ql-dlg h2{margin:0 0 2px;font-size:19px;color:#00693e}.ql-dlg .ql-sub{color:#5a6b63;margin:0 0 12px;font-size:13px}
.ql-q{border-top:1px solid #e3ebe6;padding:10px 0}.ql-q p{margin:0 0 6px;font-weight:600}
.ql-q label{display:block;padding:4px 8px;border-radius:6px;cursor:pointer}.ql-q label:hover{background:#f1f6f3}
.ql-q .ql-why{margin:6px 0 0;padding:6px 10px;border-radius:6px;font-weight:400;font-size:13.5px}
.ql-q .ql-why.ok{background:#e6f5ec}.ql-q .ql-why.no{background:#fdeceb}
.ql-rate{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:4px 0 8px}.ql-rate button{min-width:36px;padding:5px 8px;border:1px solid #c9d6cf;background:#fff;border-radius:6px;cursor:pointer;font:600 13px Helvetica,Arial,sans-serif}
.ql-rate button[aria-pressed="true"]{background:#00693e;color:#fff;border-color:#00693e}
.ql-dlg textarea{width:100%;box-sizing:border-box;min-height:56px;border:1px solid #c9d6cf;border-radius:6px;padding:6px 8px;font:14px Helvetica,Arial,sans-serif}
.ql-dlg .ql-row{display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap;margin-top:12px}
.ql-note{font-size:12px;color:#5a6b63}
.ql-offer{position:fixed;right:16px;bottom:16px;z-index:99998;background:#fff;color:#1b2a24;border:1px solid #00693e;border-radius:10px;padding:10px 12px;box-shadow:0 8px 24px rgba(0,0,0,.2);font:14px Helvetica,Arial,sans-serif;max-width:320px}
.ql-offer .ql-row{margin-top:8px}
.ql-privacy{font:12px/1.5 Helvetica,Arial,sans-serif;color:#5a6b63;text-align:center;margin:24px 12px 10px}.qfund .ql-privacy{margin:6px 0 0}
.ql-privacy button{font:inherit;color:#00693e;background:none;border:0;padding:0;text-decoration:underline;cursor:pointer}
.ql-fb{font:13px/1.5 Helvetica,Arial,sans-serif;color:#5a6b63;text-align:center;margin:20px 12px 4px;display:flex;flex-wrap:wrap;gap:6px 8px;justify-content:center;align-items:center}.qfund .ql-fb{margin:8px 0 0}
.ql-fb button{font:600 13px Helvetica,Arial,sans-serif;min-height:32px;min-width:44px;padding:4px 12px;border:1px solid #00693e;background:#fff;color:#00693e;border-radius:6px;cursor:pointer}.ql-fb button:focus-visible{outline:3px solid #ffd166;outline-offset:2px}
.ql-fb a{color:#00693e}.ql-fb b{color:#1b2a24}
.ql-ch{position:fixed;left:14px;bottom:14px;z-index:99990;max-width:min(340px,calc(100vw - 28px));background:#fff;color:#1b2a24;border:1px solid #00693e;border-radius:10px;box-shadow:0 6px 22px rgba(0,0,0,.18);font:13.5px/1.45 Helvetica,"Helvetica Neue",Arial,sans-serif}
.ql-ch>button.ql-ch-head{all:unset;box-sizing:border-box;display:flex;gap:8px;align-items:center;width:100%;padding:8px 12px;cursor:pointer;font-weight:700;color:#00693e}
.ql-ch>button.ql-ch-head:focus-visible{outline:3px solid #ffd166;outline-offset:2px;border-radius:9px}
.ql-ch-bar{flex:1;height:6px;border-radius:3px;background:#e3ebe6;overflow:hidden;min-width:40px}.ql-ch-bar i{display:block;height:100%;background:#00693e;transition:width .4s}
.ql-ch ol{list-style:none;margin:0;padding:0 12px 6px}.ql-ch li{display:flex;gap:8px;padding:4px 0;border-top:1px solid #eef3f0}
.ql-ch li .ql-ck{flex:0 0 18px;height:18px;border-radius:50%;border:2px solid #9fb3a9;box-sizing:border-box;margin-top:1px;text-align:center;font:700 11px/14px Helvetica,Arial,sans-serif;color:#fff}
.ql-ch li.done .ql-ck{background:#00693e;border-color:#00693e}.ql-ch li.done .ql-ct{color:#5a6b63;text-decoration:line-through;text-decoration-color:#9fb3a9}
.ql-ch li.now .ql-ct{font-weight:600}.ql-ch li.later{opacity:.75}
.ql-ch .ql-hint{margin:3px 0 0;padding:4px 8px;border-radius:6px;background:#fff6d6;font-size:12.5px;font-weight:400;text-decoration:none}
.ql-ch .ql-ch-foot{display:flex;justify-content:space-between;gap:8px;padding:2px 12px 8px;font-size:12px;color:#5a6b63}
.ql-ch .ql-ch-foot button{font:inherit;color:#00693e;background:none;border:0;padding:0;text-decoration:underline;cursor:pointer}
.ql-ch.ql-min ol,.ql-ch.ql-min .ql-ch-foot{display:none}
.ql-ch:not(.ql-all) li.done,.ql-ch:not(.ql-all) li.later{display:none}.ql-ch:not(.ql-all) li.now{border-top:0}
.ql-ch.ql-pop{animation:qlpop .6s}@keyframes qlpop{0%{transform:scale(1)}30%{transform:scale(1.06)}100%{transform:scale(1)}}
@media print{.ql-ch{display:none}}`;
  doc.head.appendChild(css);

  const el = (tag, attrs, ...kids) => {
    const n = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') n.className = v; else if (k === 'html') n.innerHTML = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
    }
    for (const c of kids) if (c != null) n.append(c.nodeType ? c : doc.createTextNode(String(c)));
    return n;
  };

  // ── Learning panel ───────────────────────────────────────────────────────────────────────────────────
  function learning(s) {
    spec = s || null;
    if (spec && spec.id && ID.test(spec.id)) game = spec.id;
  }
  let panelHost = null, panelOpts = null, panelBox = null;
  function panel(target, opts) {
    const host = typeof target === 'string' ? doc.querySelector(target) : target;
    if (!host || !spec) return null;
    const o = opts || {};
    // entries are strings, or { text, depth } shown only at matching depths
    const pickT = x => (typeof x === 'string' ? x : x && depthMatch(x.depth, curDepth) ? x.text : null);
    const list = (items, ordered) => { const l = el(ordered ? 'ol' : 'ul'); (items || []).map(pickT).filter(Boolean).forEach(t => l.append(el('li', { html: t }))); return l; };
    // the panel's heading sits one level below the heading before it (h1 -> h2), never deeper than h3; subheads one below
    const at = o._swap && panelBox && panelBox.isConnected ? panelBox : host;
    const inHost = at === host && !o.replace;
    let prev = 0;
    for (const h of doc.querySelectorAll('h1,h2,h3,h4,h5,h6')) {
      if (panelBox && panelBox.contains(h)) continue;
      const pos = at.compareDocumentPosition(h); // 2 = preceding, 16 = contained by
      if (((pos & 2) && !h.contains(at)) || (inHost && (pos & 16))) prev = +h.tagName[1];
    }
    const hl = Math.min(Math.max(prev + 1, 2), 3);
    const box = el('section', { class: 'ql-panel', 'aria-label': 'What you will learn and how to play' },
      el('h' + hl, { class: 'ql-h' }, o.title || spec.title || 'What you will learn'),
      spec.tagline ? el('p', { class: 'ql-tag', html: spec.tagline }) : null,
      el('div', { class: 'ql-cols' },
        el('div', {}, el('h' + (hl + 1), { class: 'ql-sh' }, 'You will learn to'), list((spec.objectives || []).filter(x => depthMatch(x.depth, curDepth)).map(x => x.text))),
        spec.howTo ? el('div', {}, el('h' + (hl + 1), { class: 'ql-sh' }, 'How to play'), list(spec.howTo, true)) : null,
        spec.goals ? el('div', {}, el('h' + (hl + 1), { class: 'ql-sh' }, 'Goals'), list(spec.goals)) : null),
      spec.fun ? el('p', { class: 'ql-fun', html: '<b>Why it is fun:</b> ' + spec.fun.charAt(0).toUpperCase() + spec.fun.slice(1) }) : null,
      o.noDebrief ? null : el('div', { class: 'ql-row', style: 'margin-top:8px;justify-content:flex-start' },
        el('button', { class: 'ql-btn', type: 'button', onclick: () => showQuiz() }, 'Debrief: what did you learn? (1 min) ▸')));
    if (o._swap && panelBox && panelBox.isConnected) panelBox.replaceWith(box);
    else if (o.replace) host.replaceChildren(box); else host.append(box);
    panelHost = host; panelOpts = Object.assign({}, o, { _swap: false }); panelBox = box;
    return box;
  }

  // ── In-game answers ──────────────────────────────────────────────────────────────────────────────────
  const answered = new Set();
  function addItems(list) {
    if (!spec) return;
    spec.quiz = spec.quiz || [];
    for (const it of list || []) if (it && it.id && !spec.quiz.some(q => q.id === it.id)) spec.quiz.push(it);
  }
  function answer(itemId, pick) {
    const items = (spec && spec.quiz) || [], q = items.find(x => x.id === itemId);
    if (!q) return null;
    const ok = pick === q.answer;
    answered.add(itemId);
    const rec = cleanQuiz(game, { answers: [{ q: q.id, obj: q.objective, pick, ok }], src: 'ingame' }, ctx());
    if (rec) send({ v: V, quiz: rec }, false);
    return { ok, answer: q.answer, choice: q.choices[q.answer], explain: q.explain || '' };
  }

  // ── Lab links: every activity can send people on to the Virtual Qubit Lab ────────────────────────────
  const kitBase = (() => { const me = doc.currentScript; return me && me.src ? me.src.replace(/shared\/outreach\.js(?:[?#].*)?$/, '') : ''; })();
  function labLink(query) {
    const q = (query ? String(query).replace(/^\?/, '') + '&' : '') + 'from=' + encodeURIComponent(game || 'site');
    const base = offline ? 'https://mvwf.github.io/qublitz/' : kitBase;
    return base + 'qublitz/qubit-simulator.html?' + q;
  }
  function toLab(a, query) {
    if (!a) return;
    a.href = labLink(query);
    a.addEventListener('click', () => { event('to_lab'); flush(true); });
  }

  // ── Debrief (the end-of-activity card) ───────────────────────────────────────────────────────────────
  let quizOpen = false, rounds = 0;      // later rounds ("3 more questions") skip the ratings
  function showQuiz() {
    if (!spec || quizOpen) return;
    quizOpen = true;
    event('quiz_open');
    // a short debrief: QUIZ_N questions (CFG.quizN, default 3), with an offer of more at the end
    const rest = (spec.quiz || []).filter(q => !answered.has(q.id) && depthMatch(q.depth, curDepth)), items = pickQuiz(rest, CFG.quizN || 3);
    const picks = items.map(() => -1), rate = { fun: 0, learned: 0 };
    const prevFocus = doc.activeElement;
    const veil = el('div', { class: 'ql-veil' });
    const dlg = el('div', { class: 'ql-dlg', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'ql-title' });
    const close = () => { veil.remove(); quizOpen = false; doc.removeEventListener('keydown', onKey); if (prevFocus && prevFocus.focus) prevFocus.focus(); };
    const onKey = e => { if (e.key === 'Escape') close(); };
    doc.addEventListener('keydown', onKey);

    dlg.append(el('h2', { id: 'ql-title' }, `${spec.title || 'Activity'}: debrief`),
      el('p', { class: 'ql-sub' }, items.length
        ? `${items.length} quick question${items.length > 1 ? 's' : ''}, then tell us how it went. You see the answers right away.`
        : 'Tell us how it went: it helps us make this more fun and clearer.'));
    const qs = items.map((q, qi) => {
      const box = el('fieldset', { class: 'ql-q', style: 'border:0;margin:0' }, el('legend', { style: 'padding:0' }, el('p', { html: `${qi + 1}. ${q.q}` })));
      // Shown in a shuffled order (recorded by the original index), so no file can leave every answer first.
      const order = q.choices.map((c, ci) => ci);
      for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      order.forEach(ci => {
        const c = q.choices[ci];
        const id = `ql-${qi}-${ci}`;
        box.append(el('label', { for: id }, el('input', { type: 'radio', name: `ql-${qi}`, id, onchange: () => { picks[qi] = ci; } }), ' ', el('span', { html: c })));
      });
      dlg.append(box); return box;
    });
    const rateRow = (key, label) => {
      const row = el('div', { class: 'ql-rate', role: 'group', 'aria-label': label }, el('span', { style: 'margin-right:6px' }, label));
      for (let v = 1; v <= 5; v++) row.append(el('button', { type: 'button', 'aria-pressed': 'false', onclick: e => {
        rate[key] = v; row.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === e.currentTarget))); } }, String(v)));
      return row;
    };
    if (!rounds) dlg.append(el('div', { class: 'ql-q' },
      rateRow('fun', 'How fun was it? (1 = not at all, 5 = very)'),
      rateRow('learned', 'How much did you learn? (1 = nothing new, 5 = a lot)'),
      el('p', { class: 'ql-note' }, 'Nothing you type is collected: only which answers you picked and the two ratings. ' +
        (enabled() ? `Answers are sent anonymously to ${CFG.owner || 'the site owners'} to improve these activities.` : 'Nothing is sent from this copy; your answers stay in this browser.'))));
    const msg = el('p', { class: 'ql-note', 'aria-live': 'polite' });
    const btnCheck = el('button', { class: 'ql-btn', type: 'button' }, items.length ? 'Check my answers' : 'Send');
    const btnClose = el('button', { class: 'ql-btn ql-ghost', type: 'button', onclick: close }, 'Close');
    dlg.append(msg, el('div', { class: 'ql-row' }, btnClose, btnCheck));
    let checked = false;
    btnCheck.addEventListener('click', () => {
      if (!checked) {
        checked = true;
        let score = 0;
        items.forEach((q, qi) => {
          const ok = picks[qi] === q.answer; if (ok) score++;
          qs[qi].querySelectorAll('input').forEach(i => { i.disabled = true; });
          qs[qi].append(el('div', { class: 'ql-why ' + (ok ? 'ok' : 'no'), html: (ok ? '✓ Right. ' : `✗ The answer is: ${q.choices[q.answer]}. `) + (q.explain || '') }));
        });
        // Only questions the visitor actually answered: a blank is not a wrong answer.
        const rec = cleanQuiz(game, { answers: items.map((q, qi) => ({ q: q.id, obj: q.objective, pick: picks[qi], ok: picks[qi] === q.answer })).filter(a => a.pick >= 0),
          fun: rate.fun || undefined, learned: rate.learned || undefined }, ctx());
        const sent = rec && send({ v: V, quiz: rec }, false);
        store.set(`ql_quiz_done:${game}`, String(score));
        items.forEach((q, qi) => { if (picks[qi] >= 0) answered.add(q.id); }); rounds++;
        const more = (spec.quiz || []).filter(q => !answered.has(q.id)).length;
        msg.textContent = (items.length ? `You got ${score} of ${items.length}. ` : '') + (sent ? 'Thank you, your answers were sent anonymously.' : 'Thank you!');
        btnCheck.textContent = 'Done'; btnClose.remove();
        if (more) msg.after(el('div', { class: 'ql-row', style: 'justify-content:flex-start' },
          el('button', { class: 'ql-btn ql-ghost', type: 'button', onclick: () => { close(); showQuiz(); } }, `${Math.min(more, CFG.quizN || 3)} more question${Math.min(more, CFG.quizN || 3) > 1 ? 's' : ''} (optional)`)));
        if (game !== 'qubit-lab') {
          const a = el('a', { class: 'ql-btn ql-ghost', style: 'text-decoration:none' }, 'Now try it on a lab-style qubit: the Virtual Qubit Lab ↗');
          toLab(a); msg.after(el('div', { class: 'ql-row', style: 'justify-content:flex-start' }, a));
        }
      } else close();
    });
    veil.addEventListener('click', e => { if (e.target === veil) close(); });
    veil.append(dlg); doc.body.append(veil);
    const first = dlg.querySelector('input'); if (first) first.focus();
  }

  let offered = false;
  function offerQuiz(message) {
    if (offered || !spec || quizOpen) return;
    offered = true;
    const box = el('div', { class: 'ql-offer', role: 'status' },
      el('div', { html: message || 'Nice work! A 1-minute debrief helps us make this better.' }),
      el('div', { class: 'ql-row' },
        el('button', { class: 'ql-btn ql-ghost', type: 'button', onclick: () => box.remove() }, 'Later'),
        el('button', { class: 'ql-btn', type: 'button', onclick: () => { box.remove(); showQuiz(); } }, 'Debrief')));
    doc.body.append(box);
    setTimeout(() => box.remove(), 20000);
  }

  // Privacy line at the end of the page, only while collection is on.
  function privacyNote() {
    if (!CFG.endpoint || offline || doc.querySelector('.ql-privacy')) return;
    const p = el('p', { class: 'ql-privacy' });
    const draw = () => {
      p.replaceChildren();
      if (dnt()) { p.append('Your browser asks not to be tracked, so this page sends nothing.'); return; }
      p.append(optedOut() ? 'You opted out: this page sends nothing. ' : `This page sends anonymous usage counts (no cookies, no personal data) to ${CFG.owner || 'its owners'}. `,
        el('button', { type: 'button', onclick: () => { optOut(!optedOut()); draw(); } }, optedOut() ? 'Opt back in' : 'Opt out'),
        ' · ', el('a', { href: kitBase + 'privacy.html' }, 'Privacy'));
    };
    draw();
    // inside a landmark: the page's footer (the site bar's funding footer, or the hub's own), else a footer of its own
    const foot = doc.querySelector('footer.qfund') || Array.from(doc.querySelectorAll('body > footer')).pop();
    if (foot) foot.append(p); else doc.body.append(el('footer', {}, p));
  }
  function optOut(on) { store.set('ql_optout', on ? '1' : '0'); if (on) queue = []; }

  // ── Feedback where it happens ─────────────────────────────────────────────────────────────────────────────
  // "Was this page useful?" at the end of every page. Yes / No is an anonymous event of this page (helpful_yes,
  // helpful_no), counted like any other while collection is on. "Suggest a change" opens the public Feedback issue
  // form with this page's link and version filled in, so a report says what the visitor saw; email is offered too.
  // Nothing typed is ever sent from here. scripts/feedback_digest.js ranks these with the open issues each week.
  const FEEDBACK_REPO = 'https://github.com/mvwf/qublitz', FEEDBACK_EMAIL = 'mattias.w.fitzpatrick@dartmouth.edu';
  function feedbackURL(extra) {
    const what = [`Page: ${location.href}`, gv() ? `Page version: ${gv()}` : '', `Screen: ${root.innerWidth}×${root.innerHeight}`, '',
      extra ? extra + '\n' : '', 'What I tried, what happened, and what would have helped:', ''].filter((l, i, a) => l !== '' || a[i - 1] !== '').join('\n');
    const title = `Feedback (${game || pagePath || 'site'}): `;
    return `${FEEDBACK_REPO}/issues/new?template=feedback.yml&title=${encodeURIComponent(title)}&what=${encodeURIComponent(what)}`;
  }
  let fbVoted = '';
  function pageFeedback() {
    if (doc.querySelector('.ql-fb') || /(^|\/)(privacy|outreach\/)/.test(pagePath)) return;
    const box = el('div', { class: 'ql-fb', role: 'region', 'aria-label': 'Feedback on this page' });
    const mail = () => el('a', { href: `mailto:${FEEDBACK_EMAIL}?subject=${encodeURIComponent('Qublitz feedback: ' + (game || pagePath || 'site'))}&body=${encodeURIComponent('Page: ' + location.href + '\n\n')}` }, 'email');
    const link = text => el('a', { href: feedbackURL(), target: '_blank', rel: 'noopener' }, text);
    const draw = () => {
      box.replaceChildren();
      if (!fbVoted) {
        box.append(el('span', {}, 'Was this page useful?'),
          el('button', { type: 'button', onclick: () => vote('yes') }, 'Yes'), el('button', { type: 'button', onclick: () => vote('no') }, 'No'),
          el('span', {}, '· ', link('Suggest a change'), ' (GitHub) or ', mail()));
        return;
      }
      box.append(el('span', { role: 'status' }, el('b', {}, 'Thank you. '),
        fbVoted === 'no' ? 'What was missing or confusing? One line helps: ' : 'Anything you would change? ', link(fbVoted === 'no' ? 'tell us on GitHub' : 'Suggest it'), ' or by ', mail(), '.'));
    };
    const vote = v => { fbVoted = v; event('helpful_' + v); draw(); };
    draw();
    const foot = doc.querySelector('footer.qfund') || Array.from(doc.querySelectorAll('body > footer')).pop();
    if (foot) foot.prepend(box); else doc.body.append(el('footer', {}, box));
  }
  // A script error counts once per page load as js_error. The message never leaves the browser.
  let jsErrorSent = false;
  root.addEventListener('error', () => { if (!jsErrorSent) { jsErrorSent = true; event('js_error'); } });

  // ── Challenges: live goals that tick off from the page's own state ───────────────────────────────────
  let chAll = [], chList = [], chBox = null, chDone = new Set(), chEarly = new Set(), chTimer = 0, chOpts = {}, chSince = 0, chCur = null, chStarted = new Set();
  const chKey = () => `ql_ch:${game || 'page'}`;
  function challenges(list, opts) {
    chAll = (list || []).filter(c => c && ID.test(c.id || '') && c.text);
    chList = chAll.filter(c => depthMatch(c.depth, curDepth));
    store.set(chKey() + ':n', String(chAll.length));
    chOpts = opts || {};
    try { const a = JSON.parse(store.get(chKey()) || '[]'); chDone = new Set(Array.isArray(a) ? a.filter(id => chList.some(c => c.id === id)) : []); } catch (e) { chDone = new Set(); }
    const mount = () => { if (chBox) chBox.remove(); chBox = el('aside', { class: 'ql-ch', 'aria-label': 'Challenges' }); doc.body.append(chBox);
      // Phones start folded so the tracker never covers the page; first-time desktop visitors see the list.
      const touch = root.matchMedia && root.matchMedia('(max-width: 640px)').matches;
      if (touch || store.get(chKey() + ':min') === '1') chBox.classList.add('ql-min');
      chDraw(); clearInterval(chTimer); chTimer = setInterval(chPoll, 320); };
    if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', mount); else mount();
    return { done: challengeDone, reset: chReset };
  }
  function chNext() { return chList.find(c => !chDone.has(c.id)) || null; }
  function chDraw() {
    if (!chBox) return;
    const n = chDone.size, N = chList.length, cur = chNext();
    if (!cur || !chCur || cur.id !== chCur.id) {
      chCur = cur; chSince = performance.now();
      // One start per challenge per page load, so the dashboard's level table reads as a funnel and the quit map
      // shows which challenge people stop on.
      if (cur && !chStarted.has(cur.id)) { chStarted.add(cur.id); event('level_start', { level: 'c:' + cur.id, value: chDone.size }); }
    }
    const hintAfter = (chOpts.hintAfter != null ? chOpts.hintAfter : 40) * 1000;
    const showHint = cur && cur.hint && performance.now() - chSince > hintAfter;
    const min = chBox.classList.contains('ql-min');
    const head = el('button', { class: 'ql-ch-head', type: 'button', 'aria-expanded': String(!min), onclick: () => {
      chBox.classList.toggle('ql-min'); store.set(chKey() + ':min', chBox.classList.contains('ql-min') ? '1' : '0'); chDraw(); } },
      el('span', {}, (n === N ? '🏆 ' : '🎯 ') + (chOpts.title || 'Challenges')),
      el('span', { class: 'ql-ch-bar', 'aria-hidden': 'true' }, el('i', { style: `width:${N ? 100 * n / N : 0}%` })),
      el('span', {}, `${n}/${N}`), el('span', { 'aria-hidden': 'true' }, min ? '▴' : '▾'));
    const ol = el('ol');
    chList.forEach(c => {
      const done = chDone.has(c.id), now = cur && c.id === cur.id;
      ol.append(el('li', { class: done ? 'done' : now ? 'now' : 'later' },
        el('span', { class: 'ql-ck', 'aria-hidden': 'true' }, done ? '✓' : ''),
        el('span', { class: 'ql-ct' }, el('span', { html: c.text }), done ? el('span', { class: 'ql-sr', style: 'position:absolute;left:-9999px' }, ' (done)') : null,
          now && showHint ? el('div', { class: 'ql-hint', html: '💡 ' + c.hint }) : null)));
    });
    if (n === N) chBox.classList.add('ql-all');
    const all = chBox.classList.contains('ql-all');
    const foot = el('div', { class: 'ql-ch-foot' },
      n === N ? el('span', {}, 'All done. Nice work!')
        : el('button', { type: 'button', 'aria-expanded': String(all), onclick: () => { chBox.classList.toggle('ql-all'); chDraw(); } }, all ? 'Show just the next one' : `See all ${N}`),
      n ? el('button', { type: 'button', onclick: chReset }, 'Start over') : el('span', {}, 'Ticks off as you explore'));
    chBox.replaceChildren(head, ol, foot);
    chBox._sig = `${n}|${cur && cur.id}|${showHint}|${min}`;
  }
  function chPoll() {
    if (!chBox || !chBox.isConnected) return;
    let changed = false;
    for (const c of chList) {
      if (chDone.has(c.id) || (typeof c.check !== 'function' && !chEarly.has(c.id))) continue;
      let ok = chEarly.has(c.id); if (!ok) try { ok = c.check() === true; } catch (e) { ok = false; }
      // In order only: a later challenge can be met by accident, but it ticks once the earlier ones are done.
      if (ok && c === chNext()) { chMark(c.id); changed = true; } else if (ok) chEarly.add(c.id);
    }
    const cur = chNext(), hintAfter = (chOpts.hintAfter != null ? chOpts.hintAfter : 40) * 1000;
    const sig = `${chDone.size}|${cur && cur.id}|${!!(cur && cur.hint && performance.now() - chSince > hintAfter)}|${chBox.classList.contains('ql-min')}`;
    if (changed || sig !== chBox._sig) chDraw();
  }
  function chMark(id) {
    if (chDone.has(id) || !chList.some(c => c.id === id)) return false;
    chDone.add(id); store.set(chKey(), JSON.stringify([...chDone]));
    event('level_complete', { level: 'c:' + id, value: chDone.size });
    if (chBox) { chBox.classList.remove('ql-pop'); void chBox.offsetWidth; chBox.classList.add('ql-pop'); }
    if (chDone.size === chList.length) {
      event('win', { level: 'challenges', value: chDone.size });
      if (typeof chOpts.done === 'function') chOpts.done(); else setTimeout(() => offerQuiz('🏆 All challenges done! A 1-minute debrief checks what stuck and helps us improve this page.'), 900);
    }
    return true;
  }
  function challengeDone(id) {
    const c = chNext();
    if (!c) return false;
    if (c.id === id) { const r = chMark(id); chDraw(); return r; }
    // Done out of order: remember it, and tick it when its turn comes.
    if (chList.some(x => x.id === id)) chEarly.add(id);
    return false;
  }
  function chReset() { chDone = new Set(); store.set(chKey(), '[]'); chEarly = new Set(); if (chBox) chBox.classList.remove('ql-all'); chDraw(); }

  // ── The frontier board (ErrorSweeper's code designs) ─────────────────────────────────────────────────────
  // submitDesign sends a design (blocked on test hosts, opt-out, DNT and offline like everything else);
  // frontier reads the shared board, which sends nothing, so it works on test hosts too.
  function submitDesign(rec) {
    const d = cleanDesign(rec, ctx());
    return d ? send({ v: V, design: d }, false) : false;
  }
  // a concept map the visitor chose to share (pressing Share sends it; nothing is sent while they build it)
  function submitMap(rec) {
    const V0 = root.QL_CONCEPTS; if (!V0) return false;
    const d = cleanMap(Object.assign({ depth: curDepth }, rec), ctx(), V0.CONCEPTS.map(x => x.id), V0.RELATIONS.map(x => x.id));
    return d ? send({ v: V, map: d }, false) : false;
  }
  // the combined picture from everyone's shared maps: { maps, edges: [{ a, b, n, rel }], concepts: [{ id, degree }] }
  function mapBoard() {
    if (!CFG.endpoint || offline || typeof fetch !== 'function') return Promise.resolve(null);
    const u = CFG.endpoint + (CFG.endpoint.includes('?') ? '&' : '?') + 'maps=1';
    return fetch(u, { redirect: 'follow', cache: 'no-store' }).then(r => (r.ok ? r.json() : null)).then(j => (j && Array.isArray(j.edges) ? j : null)).catch(() => null);
  }
  function frontier(mission) {
    if (!CFG.endpoint || offline || !MISSION.test(mission || '') || typeof fetch !== 'function') return Promise.resolve([]);
    const u = CFG.endpoint + (CFG.endpoint.includes('?') ? '&' : '?') + 'designs=1&mission=' + encodeURIComponent(mission);
    return fetch(u, { redirect: 'follow', cache: 'no-store' }).then(r => (r.ok ? r.json() : null))
      .then(j => (j && Array.isArray(j.designs) ? j.designs : [])).catch(() => []);
  }

  root.QL = { depth, setDepth, onDepth, atDepth, progress, DEPTHS, challenges, challengeDone, submitDesign, frontier, submitMap, mapBoard, explored, event, context, answer, answered: id => answered.has(id), addItems, labLink, toLab, learning, panel, showQuiz, offerQuiz, enabled, optOut, code,
              get game() { return game; }, get spec() { return spec; },
              _clean: { redact, cleanEvent, cleanQuiz } };

  // after the site bar has added its footer (it does so on DOMContentLoaded, or on a 0 ms timer)
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', privacyNote); else setTimeout(privacyNote, 0);
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', pageFeedback); else setTimeout(pageFeedback, 0);
  // Counted once per page load, after the page's own learning file has had a chance to set the id.
  // Arrivals from another activity (?from=beamline) are counted by source, so the dashboard can show which
  // activities send people on to the Virtual Qubit Lab.
  setTimeout(() => {
    const m = /[?&]from=([a-z0-9-]{1,32})/.exec(location.search);
    event('open', m ? { level: 'from:' + m[1] } : undefined);
  }, 0);
})(typeof window !== 'undefined' ? window : globalThis);
