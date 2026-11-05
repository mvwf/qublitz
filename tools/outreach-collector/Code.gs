/* Code.gs: the outreach collector, a Google Apps Script web app bound to a Google Sheet you own.
 *
 *   POST  (from web/shared/outreach.js)   { v: 1, events: [...] }  or  { v: 1, quiz: {...} }
 *         Rows go to two tabs: events and quiz. No free text is accepted or stored (a decision made so that no
 *         visitor, including a child, can hand over personal information); any text fields are dropped here. The
 *         old `feedback` tab (from earlier versions) is only ever emptied, by purgeOld (see RETAIN_DAYS).
 *         Everything is validated and capped again here; the browser is not trusted.
 *   GET   ?summary=1[&days=90]   aggregate counts as JSON for web/outreach/dashboard.html (no free text).
 *         Rows go to three tabs: events, quiz, feedback (the free text, kept apart so the dashboard never
 *         shows it). Everything is validated and capped again here; the browser is not trusted.
 *   GET   ?summary=1[&days=90][&from=YYYY-MM-DD&to=YYYY-MM-DD][&code=SCIFAIR26]   aggregate counts as JSON for
 *         web/outreach/dashboard.html (no free text). from/to replace days; code keeps only that outreach session.
 *   GET   ?review=1&key=K[&days=90]   the summary plus the free-text feedback rows, for the owner only: K must equal
 *         the Script Property REVIEW_KEY (Project Settings > Script properties). Otherwise the answer is "forbidden".
 *
 * Setup: tools/outreach-collector/README.md. The pure functions at the bottom (cleanEventRow, summarize) are
 * also loaded by web/shared/tests/outreach.test.js in Node.
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
        if (q) append_('quiz', [q.quiz]);
      }
    } finally { lock.releaseLock(); }
    // Every cached summary key carries this version, so bumping it makes all of them stale at once
    // (removing 'summary' never matched, because the keys include the period).
    CacheService.getScriptCache().put('ver', String(now.getTime()), 21600);
    return text_('ok');
  } catch (err) { return text_('error'); }
}

function doGet(e) {
  var p = (e && e.parameter) || {};
  var days = Math.max(1, Math.min(730, parseInt(p.days || '90', 10) || 90));
  if (!p.summary) return text_('outreach collector: ok');
  var cache = CacheService.getScriptCache(), key = 'summary' + days + ':' + (cache.get('ver') || '0'), hit = cache.get(key);
  if (hit) return json_(hit);
  var out = JSON.stringify(summarize(read_('events'), read_('quiz'), [], days, new Date()));
  var opts = { from: RE_DAY.test(p.from || '') ? p.from : '', to: RE_DAY.test(p.to || '') ? p.to : '', code: RE_CODE.test(p.code || '') ? p.code : '' };
  var cache = CacheService.getScriptCache(), key = ['summary', days, opts.from, opts.to, opts.code].join('|'), hit = cache.get(key);
  if (hit) return json_(hit);
  var out = JSON.stringify(summarize(read_('events'), read_('quiz'), read_('feedback'), days, new Date(), opts));
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
/* A sheet cell that looks like a date comes back from getValues() as a Date at midnight in the sheet's time zone,
 * so read the day back with local getters; plain strings pass through. */
function dayKey_(v) {
  if (v && typeof v.getFullYear === 'function') return v.getFullYear() + '-' + ('0' + (v.getMonth() + 1)).slice(-2) + '-' + ('0' + v.getDate()).slice(-2);
  return String(v);
}
function redact_(t) {
  if (typeof t !== 'string') return '';
  return t.replace(/[\w.+-]+@[\w-]+(\.[\w-]+)+/g, '[removed]').replace(/(\+?\d[\d\s().-]{7,}\d)/g, '[removed]')
          .replace(/https?:\/\/\S+/g, '[link]').replace(/\s+/g, ' ').trim().slice(0, 600);
}
function cleanEventRow(ev, now) {
  if (!ev || !RE_ID.test(ev.game || '') || !RE_EV.test(ev.ev || '')) return null;
  // `received` is the server's day, not its time: nothing finer than the day is ever stored.
  return [Utilities_formatDay_(now), day_(ev.day, now), ev.game, ev.ev, s_(ev.level, 40), n_(ev.value, -1e6, 1e6), n_(ev.stars, 0, 3),
          n_(ev.dur_s, 0, 86400), RE_CODE.test(ev.code || '') ? ev.code : '', ev.dev === 'touch' ? 'touch' : 'desktop'];
}
function cleanQuizRows(q, now) {
  if (!q || !RE_ID.test(q.game || '')) return null;
  // A blank (pick < 0) is not an answer: dropping it keeps unanswered questions out of the accuracy figures.
  var answers = (Array.isArray(q.answers) ? q.answers : []).filter(function (a) { return a && typeof a.pick === 'number' && a.pick >= 0; }).slice(0, 20).map(function (a) {
    return { q: s_(String(a && a.q), 40), obj: s_(String(a && a.obj || ''), 40), pick: n_(a && a.pick, -1, 20), ok: !!(a && a.ok) };
  });
  var code = RE_CODE.test(q.code || '') ? q.code : '', day = day_(q.day, now);
  var score = answers.filter(function (a) { return a.ok; }).length;
  var quiz = [Utilities_formatDay_(now), day, q.game, code, q.dev === 'touch' ? 'touch' : 'desktop', score, answers.length,
              n_(q.fun, 1, 5), n_(q.learned, 1, 5), JSON.stringify(answers), q.src === 'ingame' ? 'ingame' : 'debrief'];
  return { quiz: quiz, feedback: null };   // free text is never stored
}

/* Aggregate the three tabs for the dashboard. Rows are arrays in TABS order. Returns only counts and averages:
 *   games{id: {opens, starts, completes, wins, losses, quizOpens, leaves, avgMinutes, levels{name: {starts, completes,
 *     starsSum}}, events{name: n}, quits{place: n}, completedPerVisit{k: n}}}, byDay{day: {game: opens}},
 *   codes{code: opens}, devices{dev: opens}, labFrom{game: arrivals at the lab},
 *   codeGames{code: {game: opens}}, codeQuiz{code: {n, ingame, avgScore, avgFun, avgLearned}},
 *   quiz{game: {n (debriefs), ingame, avgScore, avgFun, avgLearned,
 *     items{q: {objective, n, correct, picks{choice: n}, wrong{choice: n}, ingame{n, correct}}}}},
 *   feedbackCount{game: n}, totalMinutes (active minutes over every reported visit end, also per game),
 *   byMonth{YYYY-MM: {game: {opens, minutes, completes, debriefs, ingame}}}, codeTotals{code: {minutes, completes}}.
 *   Free text never leaves the sheet.
 * opts (optional): { from, to: 'YYYY-MM-DD' (inclusive; from replaces days), code: keep only that outreach session }. */
function summarize(events, quiz, feedback, days, now, opts) {
  var o = opts || {};
  var since = RE_DAY.test(o.from || '') ? o.from : Utilities_formatDay_(new Date(now.getTime() - days * 86400000));
  var until = RE_DAY.test(o.to || '') ? o.to : '';
  var only = RE_CODE.test(o.code || '') ? o.code : '';
  var inRange = function (d) { return String(d) >= since && (!until || String(d) <= until); };
  var out = { generated: now.toISOString ? now.toISOString() : String(now), days: days, since: since, until: until || Utilities_formatDay_(now),
              code: only, games: {}, byDay: {}, byMonth: {}, codes: {}, devices: {}, codeGames: {}, codeQuiz: {}, codeTotals: {}, quiz: {},
              feedbackCount: {}, labFrom: {}, totalMinutes: 0 };
  var g = function (id) {
    return out.games[id] || (out.games[id] = { opens: 0, starts: 0, completes: 0, wins: 0, losses: 0, quizOpens: 0,
      leaves: 0, durSum: 0, levels: {}, events: {}, quits: {}, completedPerVisit: {} });
  };
  var month = function (day, game) {
    var M = out.byMonth[day.slice(0, 7)] || (out.byMonth[day.slice(0, 7)] = {});
    return M[game] || (M[game] = { opens: 0, minutes: 0, completes: 0, debriefs: 0, ingame: 0 });
  };
  var codeT = function (c) { return out.codeTotals[c] || (out.codeTotals[c] = { minutes: 0, completes: 0 }); };
  events.forEach(function (r) {
    var day = dayKey_(r[1]), game = r[2], ev = r[3], level = r[4], stars = r[6], dur = r[7], code = r[8], dev = r[9];
    if (!inRange(day) || (only && code !== only)) return;
    var G = g(game);
    G.events[ev] = (G.events[ev] || 0) + 1;
    if (ev === 'open') {
      G.opens++; month(day, game).opens++;
      var D = out.byDay[day] || (out.byDay[day] = {}); D[game] = (D[game] || 0) + 1;
      if (code) {
        out.codes[code] = (out.codes[code] || 0) + 1;
        var CG = out.codeGames[code] || (out.codeGames[code] = {}); CG[game] = (CG[game] || 0) + 1;
      }
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
      if (dur !== '') {
        G.leaves++; G.durSum += Number(dur); month(day, game).minutes += Number(dur) / 60;
        if (code) codeT(code).minutes += Number(dur) / 60;
      }
    }
    if (level && (ev === 'level_start' || ev === 'level_complete')) {
      var L = G.levels[level] || (G.levels[level] = { starts: 0, completes: 0, starsSum: 0 });
      if (ev === 'level_start') L.starts++;
      else {
        L.completes++; G.completes++; month(day, game).completes++; if (code) codeT(code).completes++;
        if (stars !== '') L.starsSum += Number(stars);
      }
    }
  });
  Object.keys(out.games).forEach(function (id) {
    var G = out.games[id]; G.avgMinutes = G.leaves ? Math.round(G.durSum / G.leaves / 6) / 10 : null;
    G.totalMinutes = Math.round(G.durSum / 6) / 10; out.totalMinutes += G.durSum / 60; delete G.durSum;
  });
  out.totalMinutes = Math.round(out.totalMinutes * 10) / 10;
  var r1 = function (x) { return Math.round(x * 10) / 10; };
  Object.keys(out.byMonth).forEach(function (m) { Object.keys(out.byMonth[m]).forEach(function (id) { out.byMonth[m][id].minutes = r1(out.byMonth[m][id].minutes); }); });
  Object.keys(out.codeTotals).forEach(function (c) { out.codeTotals[c].minutes = r1(out.codeTotals[c].minutes); });
  // In-game answers (one question each, r[10] === 'ingame') feed the score and the item accuracy but are not
  // counted as debriefs: n counts debriefs, ingame counts in-game answers.
  var tally = function (Q, r) {
    if (r[10] === 'ingame') Q.ingame++; else Q.n++;
    Q.scoreSum += Number(r[5]) || 0; Q.itemsSum += Number(r[6]) || 0;
    if (r[7] !== '') { Q.fun[0] += Number(r[7]); Q.fun[1]++; }
    if (r[8] !== '') { Q.learned[0] += Number(r[8]); Q.learned[1]++; }
  };
  var finish = function (Q) {
    Q.avgScore = Q.itemsSum ? Math.round(Q.scoreSum / Q.itemsSum * 1000) / 1000 : null;
    Q.avgFun = Q.fun[1] ? Math.round(Q.fun[0] / Q.fun[1] * 100) / 100 : null;
    Q.avgLearned = Q.learned[1] ? Math.round(Q.learned[0] / Q.learned[1] * 100) / 100 : null;
    delete Q.scoreSum; delete Q.itemsSum; delete Q.fun; delete Q.learned;
  };
  var blank = function () { return { n: 0, ingame: 0, scoreSum: 0, itemsSum: 0, fun: [0, 0], learned: [0, 0] }; };
  quiz.forEach(function (r) {
    var day = dayKey_(r[1]), game = r[2], code = r[3];
    if (!inRange(day) || (only && code !== only)) return;
    if (r[10] === 'ingame') month(day, game).ingame++; else month(day, game).debriefs++;
    var Q = out.quiz[game] || (out.quiz[game] = blank());
    if (!Q.items) Q.items = {};
    tally(Q, r);
    if (code) tally(out.codeQuiz[code] || (out.codeQuiz[code] = blank()), r);
    var answers = []; try { answers = JSON.parse(r[9] || '[]'); } catch (e) {}
    answers.forEach(function (a) {
      var I = Q.items[a.q] || (Q.items[a.q] = { objective: a.obj, n: 0, correct: 0, picks: {}, wrong: {}, ingame: { n: 0, correct: 0 } });
      I.n++; if (a.ok) I.correct++; I.picks[a.pick] = (I.picks[a.pick] || 0) + 1;
      if (!a.ok) I.wrong[a.pick] = (I.wrong[a.pick] || 0) + 1;
      if (r[10] === 'ingame') { I.ingame.n++; if (a.ok) I.ingame.correct++; }
    });
  });
  Object.keys(out.quiz).forEach(function (id) { finish(out.quiz[id]); });
  Object.keys(out.codeQuiz).forEach(function (c) { finish(out.codeQuiz[c]); });
  feedback.forEach(function (r) { if (inRange(dayKey_(r[1])) && (!only || r[3] === only)) out.feedbackCount[r[2]] = (out.feedbackCount[r[2]] || 0) + 1; });
  return out;
}

/* Retention. Nothing written is collected, but a deployment of an older version could have written free text
 * to the `feedback` tab: purgeOld deletes every such row older than RETAIN_DAYS, and anything at all in that tab
 * when purgeFeedbackNow is run. Run setupRetention once (Apps Script editor > Run) to install a daily trigger. */
var RETAIN_DAYS = 365;
function setupRetention() {
  ScriptApp.getProjectTriggers().forEach(function (t) { if (t.getHandlerFunction() === 'purgeOld') ScriptApp.deleteTrigger(t); });
  ScriptApp.newTrigger('purgeOld').timeBased().everyDays(1).atHour(3).create();
}
function purgeOld() { purgeRows_('feedback', RETAIN_DAYS); }
function purgeFeedbackNow() { purgeRows_('feedback', 0); }
function purgeRows_(name, keepDays) {
  var ss = SpreadsheetApp.getActiveSpreadsheet(), sh = ss.getSheetByName(name);
  if (!sh || sh.getLastRow() < 2) return 0;
  var rows = sh.getRange(2, 1, sh.getLastRow() - 1, 1).getValues(), cutoff = Date.now() - keepDays * 86400000, gone = 0;
  for (var i = rows.length - 1; i >= 0; i--) {
    var t = rows[i][0] instanceof Date ? rows[i][0].getTime() : Date.parse(rows[i][0]);
    if (!(t >= cutoff)) { sh.deleteRow(i + 2); gone++; }
  }
  return gone;
}

if (typeof module !== 'undefined') module.exports = { cleanEventRow: cleanEventRow, cleanQuizRows: cleanQuizRows, summarize: summarize, redact_: redact_ };
