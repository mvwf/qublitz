/* Code.gs: the outreach collector, a Google Apps Script web app bound to a Google Sheet you own.
 *
 *   POST  (from web/shared/outreach.js)   { v: 1, events: [...] }  or  { v: 1, quiz: {...} }
 *         Rows go to three tabs: events, quiz, feedback (the free text, kept apart so the dashboard never
 *         shows it). Everything is validated and capped again here; the browser is not trusted.
 *   GET   ?summary=1[&days=90]   aggregate counts as JSON for web/outreach/dashboard.html (no free text).
 *
 * Setup: tools/outreach-collector/README.md. The pure functions at the bottom (cleanEventRow, summarize) are
 * also loaded by tools/outreach-collector/collector.test.js in Node.
 */
var TABS = {
  events:   ['received', 'day', 'game', 'event', 'level', 'value', 'stars', 'dur_s', 'code', 'device'],
  quiz:     ['received', 'day', 'game', 'code', 'device', 'score', 'n', 'fun', 'learned', 'answers', 'source'],
  feedback: ['received', 'day', 'game', 'code', 'learned_text', 'confusing_text'],
};
var MAX_BODY = 20000, MAX_EVENTS = 60, CACHE_S = 300;

function doPost(e) {
  try {
    var body = e && e.postData && e.postData.contents;
    if (!body || body.length > MAX_BODY) return text_('too big');
    var msg = JSON.parse(body);
    if (!msg || msg.v !== 1) return text_('bad version');
    var now = new Date(), lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try {
      if (Array.isArray(msg.events)) {
        var rows = [];
        msg.events.slice(0, MAX_EVENTS).forEach(function (ev) { var r = cleanEventRow(ev, now); if (r) rows.push(r); });
        if (rows.length) append_('events', rows);
      }
      if (msg.quiz) {
        var q = cleanQuizRows(msg.quiz, now);
        if (q) { append_('quiz', [q.quiz]); if (q.feedback) append_('feedback', [q.feedback]); }
      }
    } finally { lock.releaseLock(); }
    CacheService.getScriptCache().remove('summary');
    return text_('ok');
  } catch (err) { return text_('error'); }
}

function doGet(e) {
  var p = (e && e.parameter) || {};
  if (!p.summary) return text_('outreach collector: ok');
  var days = Math.max(1, Math.min(730, parseInt(p.days || '90', 10) || 90));
  var cache = CacheService.getScriptCache(), key = 'summary' + days, hit = cache.get(key);
  if (hit) return json_(hit);
  var out = JSON.stringify(summarize(read_('events'), read_('quiz'), read_('feedback'), days, new Date()));
  if (out.length < 90000) cache.put(key, out, CACHE_S);
  return json_(out);
}

function sheet_(name) {
  var ss = SpreadsheetApp.getActiveSpreadsheet(), sh = ss.getSheetByName(name);
  if (!sh) { sh = ss.insertSheet(name); sh.appendRow(TABS[name]); sh.setFrozenRows(1); }
  return sh;
}
function append_(name, rows) { var sh = sheet_(name); sh.getRange(sh.getLastRow() + 1, 1, rows.length, rows[0].length).setValues(rows); }
function read_(name) { var sh = sheet_(name), n = sh.getLastRow(); return n < 2 ? [] : sh.getRange(2, 1, n - 1, TABS[name].length).getValues(); }
function text_(s) { return ContentService.createTextOutput(s).setMimeType(ContentService.MimeType.TEXT); }
function json_(s) { return ContentService.createTextOutput(s).setMimeType(ContentService.MimeType.JSON); }

// ── Pure helpers (shared with the Node test) ──────────────────────────────────────────────────────────
var RE_ID = /^[a-z0-9-]{1,32}$/, RE_EV = /^[a-z_]{1,32}$/, RE_CODE = /^[A-Za-z0-9-]{1,24}$/, RE_DAY = /^\d{4}-\d{2}-\d{2}$/;
function s_(v, n) { return (typeof v === 'string') ? v.replace(/[^\w .:/|+-]/g, '').slice(0, n) : ''; }
function n_(v, lo, hi) { return (typeof v === 'number' && isFinite(v) && v >= lo && v <= hi) ? v : ''; }
function day_(v, now) { return (typeof v === 'string' && RE_DAY.test(v)) ? v : Utilities_formatDay_(now); }
function Utilities_formatDay_(d) { return d.getUTCFullYear() + '-' + ('0' + (d.getUTCMonth() + 1)).slice(-2) + '-' + ('0' + d.getUTCDate()).slice(-2); }
function redact_(t) {
  if (typeof t !== 'string') return '';
  return t.replace(/[\w.+-]+@[\w-]+(\.[\w-]+)+/g, '[removed]').replace(/(\+?\d[\d\s().-]{7,}\d)/g, '[removed]')
          .replace(/https?:\/\/\S+/g, '[link]').replace(/\s+/g, ' ').trim().slice(0, 600);
}
function cleanEventRow(ev, now) {
  if (!ev || !RE_ID.test(ev.game || '') || !RE_EV.test(ev.ev || '')) return null;
  return [now, day_(ev.day, now), ev.game, ev.ev, s_(ev.level, 40), n_(ev.value, -1e6, 1e6), n_(ev.stars, 0, 3),
          n_(ev.dur_s, 0, 86400), RE_CODE.test(ev.code || '') ? ev.code : '', ev.dev === 'touch' ? 'touch' : 'desktop'];
}
function cleanQuizRows(q, now) {
  if (!q || !RE_ID.test(q.game || '')) return null;
  var answers = (Array.isArray(q.answers) ? q.answers : []).slice(0, 20).map(function (a) {
    return { q: s_(String(a && a.q), 40), obj: s_(String(a && a.obj || ''), 40), pick: n_(a && a.pick, -1, 20), ok: !!(a && a.ok) };
  });
  var code = RE_CODE.test(q.code || '') ? q.code : '', day = day_(q.day, now);
  var score = answers.filter(function (a) { return a.ok; }).length;
  var quiz = [now, day, q.game, code, q.dev === 'touch' ? 'touch' : 'desktop', score, answers.length,
              n_(q.fun, 1, 5), n_(q.learned, 1, 5), JSON.stringify(answers), q.src === 'ingame' ? 'ingame' : 'debrief'];
  var l = redact_(q.txt_learned), c = redact_(q.txt_confusing);
  return { quiz: quiz, feedback: (l || c) ? [now, day, q.game, code, l, c] : null };
}

/* Aggregate the three tabs for the dashboard. Rows are arrays in TABS order. Returns only counts and averages. */
function summarize(events, quiz, feedback, days, now) {
  var cutoff = new Date(now.getTime() - days * 86400000);
  var since = Utilities_formatDay_(cutoff);
  var inRange = function (d) { return String(d) >= since; };
  var out = { generated: now.toISOString ? now.toISOString() : String(now), days: days, since: since,
              games: {}, byDay: {}, codes: {}, devices: {}, quiz: {}, feedbackCount: {}, labFrom: {} };
  var g = function (id) {
    return out.games[id] || (out.games[id] = { opens: 0, starts: 0, completes: 0, wins: 0, losses: 0, quizOpens: 0,
      leaves: 0, durSum: 0, levels: {}, events: {}, quits: {}, completedPerVisit: {} });
  };
  events.forEach(function (r) {
    var day = String(r[1]), game = r[2], ev = r[3], level = r[4], stars = r[6], dur = r[7], code = r[8], dev = r[9];
    if (!inRange(day)) return;
    var G = g(game);
    G.events[ev] = (G.events[ev] || 0) + 1;
    if (ev === 'open') {
      G.opens++;
      var D = out.byDay[day] || (out.byDay[day] = {}); D[game] = (D[game] || 0) + 1;
      if (code) out.codes[code] = (out.codes[code] || 0) + 1;
      if (String(level).indexOf('from:') === 0) { var src = String(level).slice(5); out.labFrom[src] = (out.labFrom[src] || 0) + 1; }
      out.devices[dev] = (out.devices[dev] || 0) + 1;
    }
    if (ev === 'start') G.starts++;
    if (ev === 'win') G.wins++;
    if (ev === 'lose') G.losses++;
    if (ev === 'quiz_open') G.quizOpens++;
    if (ev === 'leave') {
      // Where visits end: the last place the page reported, and how many levels the visit completed.
      var at = level || 'unknown', done = r[5] === '' ? 0 : Number(r[5]);
      G.quits[at] = (G.quits[at] || 0) + 1;
      G.completedPerVisit[done] = (G.completedPerVisit[done] || 0) + 1;
      if (dur !== '') { G.leaves++; G.durSum += Number(dur); }
    }
    if (level && (ev === 'level_start' || ev === 'level_complete')) {
      var L = G.levels[level] || (G.levels[level] = { starts: 0, completes: 0, starsSum: 0 });
      if (ev === 'level_start') L.starts++;
      else { L.completes++; G.completes++; if (stars !== '') L.starsSum += Number(stars); }
    }
  });
  Object.keys(out.games).forEach(function (id) {
    var G = out.games[id]; G.avgMinutes = G.leaves ? Math.round(G.durSum / G.leaves / 6) / 10 : null; delete G.durSum;
  });
  quiz.forEach(function (r) {
    var day = String(r[1]), game = r[2];
    if (!inRange(day)) return;
    var Q = out.quiz[game] || (out.quiz[game] = { n: 0, ingame: 0, scoreSum: 0, itemsSum: 0, fun: [0, 0], learned: [0, 0], items: {} });
    // In-game answers (one question each) feed the item accuracy but are not counted as debriefs.
    if (r[10] === 'ingame') Q.ingame++; else Q.n++;
    Q.scoreSum += Number(r[5]) || 0; Q.itemsSum += Number(r[6]) || 0;
    if (r[7] !== '') { Q.fun[0] += Number(r[7]); Q.fun[1]++; }
    if (r[8] !== '') { Q.learned[0] += Number(r[8]); Q.learned[1]++; }
    var answers = []; try { answers = JSON.parse(r[9] || '[]'); } catch (e) {}
    answers.forEach(function (a) {
      var I = Q.items[a.q] || (Q.items[a.q] = { objective: a.obj, n: 0, correct: 0, picks: {} });
      I.n++; if (a.ok) I.correct++; I.picks[a.pick] = (I.picks[a.pick] || 0) + 1;
    });
  });
  Object.keys(out.quiz).forEach(function (id) {
    var Q = out.quiz[id];
    Q.avgScore = Q.itemsSum ? Math.round(Q.scoreSum / Q.itemsSum * 1000) / 1000 : null;
    Q.avgFun = Q.fun[1] ? Math.round(Q.fun[0] / Q.fun[1] * 100) / 100 : null;
    Q.avgLearned = Q.learned[1] ? Math.round(Q.learned[0] / Q.learned[1] * 100) / 100 : null;
    delete Q.scoreSum; delete Q.itemsSum; delete Q.fun; delete Q.learned;
  });
  feedback.forEach(function (r) { if (inRange(String(r[1]))) out.feedbackCount[r[2]] = (out.feedbackCount[r[2]] || 0) + 1; });
  return out;
}

if (typeof module !== 'undefined') module.exports = { cleanEventRow: cleanEventRow, cleanQuizRows: cleanQuizRows, summarize: summarize, redact_: redact_ };
