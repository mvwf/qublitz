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
 *   QL.showQuiz()               open the debrief: any unanswered items, two ratings and two free-text boxes
 *   QL.offerQuiz(message)       a small prompt that opens the debrief (once per page load)
 *   QL.labLink(query)           URL of the Virtual Qubit Lab with ?from=<this page> (plus your query)
 *   QL.toLab(anchor, query)     the same, set on a link, counting the click as a to_lab event
 *   QL.enabled(), QL.optOut(on), QL.code()
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

  if (typeof root.document === 'undefined') {           // Node: export the pure parts for the tests
    if (typeof module !== 'undefined') module.exports = { redact, cleanEvent, cleanQuiz, cleanExtra };
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
  const enabled = () => !!CFG.endpoint && !offline && !optedOut() && !dnt();

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
    const e = cleanEvent(game, name, data, ctx());
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
.ql-panel h3{margin:0 0 4px;font-size:15px;color:#00693e}.ql-panel .ql-tag{color:#5a6b63;margin:0 0 8px}
.ql-panel .ql-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:6px 22px}
.ql-panel h4{margin:6px 0 2px;font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:#5a6b63}
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
.ql-privacy{font:12px/1.5 Helvetica,Arial,sans-serif;color:#5a6b63;text-align:center;margin:24px 12px 10px}
.ql-privacy button{font:inherit;color:#00693e;background:none;border:0;padding:0;text-decoration:underline;cursor:pointer}`;
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
  function panel(target, opts) {
    const host = typeof target === 'string' ? doc.querySelector(target) : target;
    if (!host || !spec) return null;
    const o = opts || {};
    const list = (items, ordered) => { const l = el(ordered ? 'ol' : 'ul'); (items || []).forEach(t => l.append(el('li', { html: t }))); return l; };
    const box = el('section', { class: 'ql-panel', 'aria-label': 'What you will learn and how to play' },
      el('h3', {}, o.title || spec.title || 'What you will learn'),
      spec.tagline ? el('p', { class: 'ql-tag', html: spec.tagline }) : null,
      el('div', { class: 'ql-cols' },
        el('div', {}, el('h4', {}, 'You will learn to'), list((spec.objectives || []).map(x => x.text))),
        spec.howTo ? el('div', {}, el('h4', {}, 'How to play'), list(spec.howTo, true)) : null,
        spec.goals ? el('div', {}, el('h4', {}, 'Goals'), list(spec.goals)) : null),
      spec.fun ? el('p', { class: 'ql-fun', html: '<b>Why it is fun:</b> ' + spec.fun.charAt(0).toUpperCase() + spec.fun.slice(1) }) : null,
      o.noDebrief ? null : el('div', { class: 'ql-row', style: 'margin-top:8px;justify-content:flex-start' },
        el('button', { class: 'ql-btn', type: 'button', onclick: () => showQuiz() }, 'Debrief: what did you learn? (1 min) ▸')));
    if (o.replace) host.replaceChildren(box); else host.append(box);
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
  let quizOpen = false;
  function showQuiz() {
    if (!spec || quizOpen) return;
    quizOpen = true;
    event('quiz_open');
    const items = (spec.quiz || []).filter(q => !answered.has(q.id));
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
    dlg.append(el('div', { class: 'ql-q' },
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
        msg.textContent = (items.length ? `You got ${score} of ${items.length}. ` : '') + (sent ? 'Thank you, your answers were sent anonymously.' : 'Thank you!');
        btnCheck.textContent = 'Done'; btnClose.remove();
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
    draw(); doc.body.append(p);
  }
  function optOut(on) { store.set('ql_optout', on ? '1' : '0'); if (on) queue = []; }

  root.QL = { explored, event, context, answer, answered: id => answered.has(id), addItems, labLink, toLab, learning, panel, showQuiz, offerQuiz, enabled, optOut, code,
              get game() { return game; }, get spec() { return spec; },
              _clean: { redact, cleanEvent, cleanQuiz } };

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', privacyNote); else privacyNote();
  // Counted once per page load, after the page's own learning file has had a chance to set the id.
  // Arrivals from another activity (?from=beamline) are counted by source, so the dashboard can show which
  // activities send people on to the Virtual Qubit Lab.
  setTimeout(() => {
    const m = /[?&]from=([a-z0-9-]{1,32})/.exec(location.search);
    event('open', m ? { level: 'from:' + m[1] } : undefined);
  }, 0);
})(typeof window !== 'undefined' ? window : globalThis);
