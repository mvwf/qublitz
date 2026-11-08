// Tests for the outreach kit: the browser-side sanitising in web/shared/outreach.js and the collector's
// validation and aggregation in tools/outreach-collector/Code.gs (both load in Node without a DOM).
'use strict';
const path = require('path');
const root = path.join(__dirname, '..', '..', '..');
const kit = require(path.join(root, 'web/shared/outreach.js'));
const gs = require(path.join(root, 'tools/outreach-collector/Code.gs'));
global.window = {}; require(path.join(root, 'web/learning/manifest.js')); const GAMES = global.window.QL_GAMES;

let fails = 0;
// most checks below use rows without a page version, so they ask the summary to include them (dev: '1')
const summ = (e, q, f, d, n, o) => gs.summarize(e, q, f, d, n, Object.assign({ dev: '1' }, o || {}));
const check = (name, ok, detail) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  · ' + detail : ''}`); if (!ok) fails++; };

// depth tags: 'x' only, 'x+' and deeper, '-x' up to x; untagged or unknown tags show everywhere
{
  const m = kit.depthMatch;
  check('depth: untagged content shows at every depth', kit.DEPTHS.every(d => m(undefined, d) && m('', d) && m('banana', d)));
  check('depth: "engineer+" shows for engineer and researcher only', !m('engineer+', 'curious') && !m('engineer+', 'student') && m('engineer+', 'engineer') && m('engineer+', 'researcher'));
  check('depth: "-student" shows for curious and student only', m('-student', 'curious') && m('-student', 'student') && !m('-student', 'engineer'));
  check('depth: "curious researcher" lists two depths', m('curious researcher', 'curious') && m('curious researcher', 'researcher') && !m('curious researcher', 'student'));
  const ev = kit.cleanEvent('beamline', 'open', { depth: 'engineer' }, { day: '2026-11-08', dev: 'touch' });
  check('the depth travels as a one-word label in x', ev && ev.x && ev.x.depth === 'engineer');
}

// redaction: identifying text never survives, ordinary text does
const r = kit.redact('I am jo.smith+x@dartmouth.edu, call 603-555-0199 or see https://x.y/z. Rotations   about x!');
check('redact removes emails, phone numbers and links, keeps the rest', !/@|555|https/.test(r) && /Rotations about x!/.test(r), r);
check('redact caps length at 600', kit.redact('a'.repeat(5000)).length === 600);
check('collector redacts the same way', gs.redact_('mail me a@b.co') === 'mail me [removed]');

// events: only known shapes pass
const ctx = { day: '2026-11-05', dev: 'desktop', code: 'SCIFAIR26' };
const e = kit.cleanEvent('beamline', 'level_complete', { level: 'half', stars: 3, dur_s: 12.4, junk: 'x' }, ctx);
check('cleanEvent keeps game, event, level, stars, duration and code', e && e.level === 'half' && e.stars === 3 && e.dur_s === 12 && e.code === 'SCIFAIR26' && !('junk' in e));
check('cleanEvent rejects bad ids', kit.cleanEvent('Beam Line', 'open', {}, ctx) === null && kit.cleanEvent('beamline', 'drop table', {}, ctx) === null);
check('cleanEvent drops a bad outreach code', !('code' in kit.cleanEvent('rider', 'open', {}, { ...ctx, code: '<script>' })));

// quiz: score is recomputed from the answers, and no text survives (there is no free text by design)
const q = kit.cleanQuiz('copter', { answers: [{ q: 'q1', obj: 'o1', pick: 0, ok: true }, { q: 'q2', obj: 'o2', pick: 2, ok: false }],
  fun: 5, learned: 4, txt_learned: 'rotations! me@x.org', txt_confusing: '' }, ctx);
check('cleanQuiz recomputes score and drops any text fields', q.score === 1 && q.n === 2 && q.fun === 5 && !('txt_learned' in q) && !('txt_confusing' in q) && !JSON.stringify(q).includes('me@x.org'));

// collector rows and summary
const now = new Date('2026-11-05T12:00:00Z');
const rows = [
  gs.cleanEventRow({ game: 'beamline', ev: 'open', day: '2026-11-05', dev: 'desktop', code: 'SCIFAIR26' }, now),
  gs.cleanEventRow({ game: 'beamline', ev: 'level_start', level: 'half', day: '2026-11-05' }, now),
  gs.cleanEventRow({ game: 'beamline', ev: 'level_complete', level: 'half', stars: 3, day: '2026-11-05' }, now),
  gs.cleanEventRow({ game: 'beamline', ev: 'leave', dur_s: 120, day: '2026-11-05' }, now),
  gs.cleanEventRow({ game: 'rider', ev: 'open', day: '2025-01-01' }, now),            // outside 90 days
  gs.cleanEventRow({ game: 'BAD', ev: 'open' }, now),
].filter(Boolean);
check('collector rejects a bad event row', rows.length === 5);
const qz = gs.cleanQuizRows({ game: 'beamline', day: '2026-11-05', answers: [{ q: 'q1', obj: 'o1', pick: 1, ok: true }, { q: 'q2', obj: 'o2', pick: 0, ok: false }],
  fun: 4, learned: 5, txt_learned: 'phase sets the axis', txt_confusing: 'call 603 555 0199' }, now);
check('collector stores no free text, even if a client sends it', qz.quiz[5] === 1 && qz.feedback === null && !JSON.stringify(qz).includes('phase sets the axis') && !JSON.stringify(qz).includes('555'));
const S = summ(rows, [qz.quiz], [], 90, now);
const B = S.games.beamline;
check('summary counts opens, level funnel and average minutes', B.opens === 1 && B.levels.half.starts === 1 && B.levels.half.completes === 1 && B.avgMinutes === 2);
check('summary skips rows outside the window', !S.games.rider);
check('summary has quiz accuracy per item and ratings', S.quiz.beamline.items.q1.correct === 1 && S.quiz.beamline.avgScore === 0.5 && S.quiz.beamline.avgFun === 4);
check('summary contains no free text', !JSON.stringify(S).includes('phase sets the axis'));
check('summary splits by outreach code', S.codes.SCIFAIR26 === 1);
check('summary splits outreach codes by game', S.codeGames.SCIFAIR26.beamline === 1 && Object.keys(S.codeGames).length === 1);
check('summary records the wrong picks per item', S.quiz.beamline.items.q2.wrong[0] === 1 && Object.keys(S.quiz.beamline.items.q1.wrong).length === 0);

// quizzes per outreach code: n, score and ratings, and no code means no entry
const coded = [gs.cleanQuizRows({ game: 'rider', day: '2026-11-04', code: 'SCIFAIR26', answers: [{ q: 'q1', pick: 0, ok: true }, { q: 'q2', pick: 1, ok: true }], fun: 5, learned: 3 }, now).quiz,
  gs.cleanQuizRows({ game: 'copter', day: '2026-11-04', code: 'SCIFAIR26', answers: [{ q: 'q1', pick: 2, ok: false }, { q: 'q2', pick: 1, ok: true }], fun: 3 }, now).quiz,
  qz.quiz];
const SC = summ([], coded, [], 90, now);
const CQ = SC.codeQuiz.SCIFAIR26;
check('summary aggregates quizzes per outreach code', CQ && CQ.n === 2 && CQ.avgScore === 0.75 && CQ.avgFun === 4 && CQ.avgLearned === 3 && Object.keys(SC.codeQuiz).length === 1,
  JSON.stringify(CQ));

// a day cell that Sheets turned into a Date still lands on the right day
const dated = gs.cleanEventRow({ game: 'drum', ev: 'open', day: '2026-11-03' }, now); dated[1] = new Date(2026, 10, 3);
const S3 = summ([dated], [], [], 7, now);
check('summary reads day cells that Sheets stored as dates', S3.byDay['2026-11-03'] && S3.byDay['2026-11-03'].drum === 1, JSON.stringify(S3.byDay));

// the private review feed was removed with the free text: the collector exports no review helpers
check('no review feed is exported', !('reviewFeed' in gs) && !('reviewKeyOk' in gs));

// engagement totals, months, date ranges and the outreach-code filter (for the dashboard's report)
const ER = [
  gs.cleanEventRow({ game: 'copter', ev: 'open', day: '2026-10-30', code: 'FAIR' }, now),
  gs.cleanEventRow({ game: 'copter', ev: 'level_complete', level: 'hover', day: '2026-10-30', code: 'FAIR' }, now),
  gs.cleanEventRow({ game: 'copter', ev: 'leave', dur_s: 600, day: '2026-10-30', code: 'FAIR' }, now),
  gs.cleanEventRow({ game: 'copter', ev: 'open', day: '2026-11-02' }, now),
  gs.cleanEventRow({ game: 'copter', ev: 'leave', dur_s: 300, day: '2026-11-02' }, now),
  gs.cleanEventRow({ game: 'rider', ev: 'leave', dur_s: 120, day: '2026-11-03' }, now),
];
const EQ = [gs.cleanQuizRows({ game: 'copter', day: '2026-10-30', code: 'FAIR', answers: [{ q: 'q1', pick: 0, ok: true }], fun: 5 }, now).quiz,
  gs.cleanQuizRows({ game: 'copter', day: '2026-11-02', src: 'ingame', answers: [{ q: 'q1', pick: 1, ok: false }] }, now).quiz];
const E1 = summ(ER, EQ, [], 90, now);
check('summary totals active minutes per game and overall', E1.games.copter.totalMinutes === 15 && E1.games.rider.totalMinutes === 2 && E1.totalMinutes === 17, JSON.stringify([E1.games.copter.totalMinutes, E1.totalMinutes]));
check('summary groups visits, minutes, completions and answers by month', E1.byMonth['2026-10'].copter.opens === 1 && E1.byMonth['2026-10'].copter.minutes === 10 &&
  E1.byMonth['2026-10'].copter.completes === 1 && E1.byMonth['2026-10'].copter.debriefs === 1 && E1.byMonth['2026-11'].copter.ingame === 1 && E1.byMonth['2026-11'].copter.minutes === 5);
check('summary totals minutes and completions per outreach code', E1.codeTotals.FAIR.minutes === 10 && E1.codeTotals.FAIR.completes === 1);
const E2 = summ(ER, EQ, [], 90, now, { from: '2026-11-01', to: '2026-11-02' });
check('summary honours from/to (inclusive)', E2.since === '2026-11-01' && E2.until === '2026-11-02' && E2.games.copter.opens === 1 && !E2.games.rider && E2.quiz.copter.ingame === 1 && E2.quiz.copter.n === 0);
const E3 = summ(ER, EQ, [], 90, now, { code: 'FAIR' });
check('summary keeps only one outreach code when asked', E3.code === 'FAIR' && E3.games.copter.opens === 1 && E3.totalMinutes === 10 && E3.quiz.copter.n === 1 && !E3.quiz.copter.ingame);
check('summary ignores a malformed range or code', summ(ER, EQ, [], 90, now, { from: 'yesterday', code: '<x>' }).games.rider.leaves === 1);

// insights: each rule fires below its threshold once enough people were measured, and flags rank by n × gap
const IN = require(path.join(root, 'web/outreach/insights.js'));
const specs = { beamline: { id: 'beamline', title: 'Beamline', objectives: [{ id: 'axis', text: 'Relate <b>phase</b> to axis' }],
  quiz: [{ id: 'q3', objective: 'axis', q: 'Which axis?', choices: ['x', 'y', '&minus;x', 'z'], answer: 1 }] } };
const item = (n, correct, wrong) => ({ objective: 'axis', n, correct, picks: {}, wrong });
const sum = {
  games: {
    beamline: { opens: 80, quizOpens: 80, leaves: 60, avgMinutes: 4, levels: { easy: { starts: 50, completes: 45 }, hard: { starts: 30, completes: 6 }, rare: { starts: 4, completes: 0 } } },
    rider: { opens: 20, quizOpens: 4, leaves: 20, avgMinutes: 0.4, levels: {} },
    hub: { opens: 300, leaves: 200, avgMinutes: 0.2, levels: {} },
  },
  quiz: {
    beamline: { n: 30, avgFun: 4.2, avgLearned: 2.5, items: { q3: item(40, 12, { 0: 3, 2: 22, 3: 3 }), q4: item(40, 30, { 0: 10 }), q5: item(6, 0, { 1: 6 }) } },
    rider: { n: 3, avgFun: 1.5, avgLearned: 1.5, items: {} },
  },
};
const F = IN.insights(sum, specs, { beamline: 'Beamline', rider: 'Rabi Rider' });
const kinds = F.map(f => f.kind + ':' + f.game + ':' + f.ref);
check('insights flag the low-accuracy quiz item with its misconception', F.some(f => f.kind === 'quiz' && f.ref === 'q3' && /possible misconception: “−x”/.test(f.evidence) && /phase to axis/.test(f.fix)), kinds.join(' '));
check('insights skip items above 50% or with too few answers', !kinds.includes('quiz:beamline:q4') && !kinds.includes('quiz:beamline:q5'));
check('insights flag a level with a low completion ratio, not a rarely played one', kinds.includes('level:beamline:hard') && !kinds.includes('level:beamline:easy') && !kinds.includes('level:beamline:rare'));
check('insights flag low ratings only with enough quizzes', kinds.includes('rating:beamline:learned') && !kinds.includes('rating:beamline:fun') && !kinds.some(k => k.startsWith('rating:rider')));
check('insights flag debrief drop-off', kinds.includes('dropoff:beamline:debrief') && !kinds.some(k => k.startsWith('dropoff:rider')));
check('insights flag short visits but not on the hub', kinds.includes('time:rider:time') && !kinds.some(k => k.startsWith('time:hub')));
check('insights rank by n × relative gap', F.every((f, i) => i === 0 || F[i - 1].impact >= f.impact) && F[0].kind === 'dropoff' && F[0].impact === 20 && F[1].ref === 'q3',
  F.map(f => f.ref + '=' + f.impact).join(' '));
const stopSum = { games: { rider: { opens: 100, levels: { a: { starts: 90, completes: 80 }, b: { starts: 60, completes: 30 }, c: { starts: 30, completes: 28 }, d: { starts: 20, completes: 3 } },
  quits: { menu: 40, a: 8, 'b|boss': 33, c: 4, d: 6 } } } };
const endSum = { games: { copter: { opens: 50, levels: { a: { starts: 50, completes: 45 }, last: { starts: 40, completes: 36 } }, quits: { a: 10, last: 40 } } } };
check('insights do not flag the last level when people finish it and leave', !IN.insights(endSum, {}, {}).some(f => f.kind === 'stop'));
const SF = IN.insights(stopSum, {}, {}).filter(f => f.kind === 'stop');
check('insights flag a stop hotspot against the share that reach it', SF.length === 1 && SF[0].kind === 'stop' && SF[0].ref === 'b|boss' && /33 visits ended there of 60 that started b, 30 of them without finishing/.test(SF[0].evidence),
  SF.map(f => f.kind + ':' + f.ref + ' ' + f.evidence).join(' | '));
check('topWrong falls back to picks minus the right answer', IN.topWrong({ n: 10, correct: 4, picks: { 0: 1, 1: 4, 3: 5 } }, specs.beamline.quiz[0]).text === 'z');
check('insights tolerate an empty summary', IN.insights({}, {}, {}).length === 0 && IN.insights(null).length === 0);

// quit map, visit depth, in-game answers and lab arrivals
const rows2 = [
  gs.cleanEventRow({ game: 'rider', ev: 'leave', level: 'photon', value: 1, dur_s: 90, day: '2026-11-05' }, now),
  gs.cleanEventRow({ game: 'rider', ev: 'leave', level: 'photon', value: 1, dur_s: 60, day: '2026-11-05' }, now),
  gs.cleanEventRow({ game: 'rider', ev: 'leave', level: 'menu', value: 0, dur_s: 10, day: '2026-11-05' }, now),
  gs.cleanEventRow({ game: 'qubit-lab', ev: 'open', level: 'from:beamline', day: '2026-11-05' }, now),
];
const ing = gs.cleanQuizRows({ game: 'rider', day: '2026-11-05', src: 'ingame', answers: [{ q: 'q1', obj: 'o1', pick: 2, ok: true }] }, now);
const S2 = summ(rows2, [ing.quiz], [], 90, now);
check('summary maps where visits end and how far they got', S2.games.rider.quits.photon === 2 && S2.games.rider.quits.menu === 1 && S2.games.rider.completedPerVisit[1] === 2);
check('in-game answers count toward items but not debriefs', S2.quiz.rider.items.q1.correct === 1 && S2.quiz.rider.n === 0 && S2.quiz.rider.ingame === 1);
check('summary counts lab arrivals by source activity', S2.labFrom.beamline === 1);
check('items split in-game answers from debrief answers', S2.quiz.rider.items.q1.ingame.n === 1 && S2.quiz.rider.items.q1.ingame.correct === 1 && S.quiz.beamline.items.q1.ingame.n === 0);
const qi = kit.cleanQuiz('rider', { answers: [{ q: 'q1', obj: 'o1', pick: 2, ok: true }], src: 'ingame' }, ctx);
check('cleanQuiz keeps the in-game source', qi.src === 'ingame');

// nothing finer than the day: the collector's received column is a day too
check('collector stores the received day, not a time', /^\d{4}-\d{2}-\d{2}$/.test(rows[0][0]) && /^\d{4}-\d{2}-\d{2}$/.test(qz.quiz[0]));

// skipped questions are not wrong answers
const sk = kit.cleanQuiz('rider', { answers: [{ q: 'a', obj: 'o', pick: 1, ok: true }, { q: 'b', obj: 'o', pick: -1, ok: false }] }, ctx);
const skg = gs.cleanQuizRows({ game: 'rider', answers: [{ q: 'a', pick: 1, ok: true }, { q: 'b', pick: -1, ok: false }] }, now);
check('blank answers are dropped in the browser and the collector', sk.n === 1 && sk.score === 1 && skg.quiz[6] === 1);

// report, redesign brief, issue link and before/after (web/outreach/report.js)
const RP = require(path.join(root, 'web/outreach/report.js'));
const repS = Object.assign({}, sum, { since: '2026-08-07', until: '2026-11-05', byMonth: { '2026-10': { beamline: { opens: 50, minutes: 200, completes: 30, debriefs: 10, ingame: 5 }, hub: { opens: 99, minutes: 9, completes: 0, debriefs: 0, ingame: 0 } },
  '2026-11': { beamline: { opens: 30, minutes: 100, completes: 20, debriefs: 20, ingame: 2 } } }, codes: { FAIR: 12 }, codeTotals: { FAIR: { minutes: 90, completes: 4 } },
  codeQuiz: { FAIR: { n: 5, ingame: 1, avgScore: 0.6, avgFun: 4.4, avgLearned: 4 } }, labFrom: { beamline: 7 } });
repS.games = JSON.parse(JSON.stringify(sum.games)); repS.games.beamline.totalMinutes = 300; repS.games.beamline.completes = 51; repS.games.rider.totalMinutes = 60; repS.games.hub.totalMinutes = 600;
const NB = RP.numbers(repS, specs, { beamline: 'Beamline', rider: 'Rabi Rider' });
check('report numbers leave the hub out of activity totals', NB.visits === 100 && NB.hubVisits === 300 && NB.activities.length === 2 && NB.hours === 6 && NB.completes === 51, JSON.stringify([NB.visits, NB.hours]));
check('report numbers total answers, ratings, lab arrivals and months', NB.answers === 86 && NB.debriefs === 33 && NB.labArrivals === 7 && NB.months.length === 2 && NB.months[0].visits === 50 && NB.codes[0].minutes === 90);
const TX = RP.reportText(repS, specs, { beamline: 'Beamline' });
check('report text states the range and the caveats', /From Aug 7, 2026 to Nov 5, 2026/.test(TX) && /not unique people/.test(TX) && /lower bound/.test(TX) && !/DEMO/.test(TX));
check('report text labels demo data', /^\[DEMO DATA/.test(RP.reportText(Object.assign({}, repS, { demo: true }))));
const MC = RP.monthsCsv(repS, { beamline: 'Beamline' }).trim().split('\n');
check('CSV per game per month', MC[0] === 'month,game,title,visits,active_minutes,levels_completed,debriefs,ingame_answers' && MC.length === 4 && MC[1] === '2026-10,beamline,Beamline,50,200,30,10,5');
check('CSV per event code', RP.codesCsv(repS).trim().split('\n')[1] === 'FAIR,12,90,4,5,1,0.6,4.4,4');
const BR = RP.brief(repS, specs, { beamline: 'Beamline' }, 'beamline');
check('brief names the misconception, stops, ratings and experiments', /Redesign brief: Beamline/.test(BR.title) && /most-picked wrong answer “−x”/.test(BR.body) && /### Suggested next experiments\n1\. /.test(BR.body) && !/[Ww]ritten|free text/.test(BR.body), BR.body.slice(0, 200));
const big = RP.issueUrl(BR.title, BR.body + '\n' + 'x'.repeat(20000), 6000);
check('issue link stays under the limit and says it was truncated', big.truncated && big.url.length <= 6000 && /Truncated/.test(decodeURIComponent(big.url)) && /labels=feedback/.test(big.url));
check('short issue link is not truncated', RP.issueUrl('t', 'body').truncated === false);
const lv = (s, c) => ({ games: { beamline: { levels: { hard: { starts: s, completes: c } }, quizOpens: 40, leaves: 30, avgMinutes: 3 } }, quiz: { beamline: { n: 25, avgFun: 3.5, avgLearned: 3, items: { q3: item(40, 12, {}) } } } });
const CMP = RP.compare(lv(30, 6), lv(25, 15), { game: 'beamline', level: 'hard', objective: 'axis' }, specs);
const row = k => CMP.find(r => r.key === k);
check('before/after compares the level and the objective with n on each side', row('level').before.v === 0.2 && row('level').after.v === 0.6 && row('level').enough && row('correct').before.n === 40 && /phase/.test(row('correct').label));
check('before/after greys out rows with too few answers', !RP.compare(lv(30, 6), lv(5, 4), { game: 'beamline', level: 'hard' }, specs).find(r => r.key === 'level').enough);
const W = RP.windows('2026-10-20', '2026-11-05');
check('before/after windows are equal and adjacent', W.before.to === '2026-10-19' && W.after.from === '2026-10-20' && W.after.to === '2026-11-05' && W.before.from === '2026-10-03' && W.days === 17, JSON.stringify(W));

// flexible game fields: numbers, booleans and one-word labels only; never sentences or nested objects
const xe = kit.cleanEvent('beamline', 'fire', { level: 'half', theta_pi: 0.5, beam: 'x', hinted: true, note: 'my name is Jo',
  nested: { a: 1 }, BadKey: 1, list: [1, 2] }, { ...ctx, gv: '2026-11-05+abc1234' });
check('extra fields keep numbers, booleans and single-word labels, and drop text, objects and bad keys',
  xe.x && xe.x.theta_pi === 0.5 && xe.x.beam === 'x' && xe.x.hinted === true && !('note' in xe.x) && !('nested' in xe.x) &&
  !('BadKey' in xe.x) && !('list' in xe.x) && !('level' in xe.x) && xe.level === 'half', JSON.stringify(xe.x));
const many = {}; for (let i = 0; i < 20; i++) many['k' + i] = i;
check('at most 12 extra fields', Object.keys(kit.cleanExtra(many)).length === 12);
check('events and questions carry the page version', xe.gv === '2026-11-05+abc1234' &&
  kit.cleanQuiz('rider', { answers: [{ q: 'a', obj: 'o', pick: 0, ok: true }] }, { ...ctx, gv: '2026-11-05+abc1234' }).gv === '2026-11-05+abc1234' &&
  !('gv' in kit.cleanEvent('rider', 'open', {}, { ...ctx, gv: 'has space' })));
const xr = gs.cleanEventRow({ game: 'beamline', ev: 'fire', day: '2026-11-05', gv: '2026-11-05+abc1234', x: { theta_pi: 0.5, beam: 'x', note: 'my name is Jo' } }, now);
check('collector stores the version and the cleaned extra fields as JSON (no text)', xr[10] === '2026-11-05+abc1234' &&
  JSON.parse(xr[11]).theta_pi === 0.5 && !xr[11].includes('Jo'));
check('collector rows keep their old column positions', xr.length === 12 && xr[3] === 'fire' && xr[9] === 'desktop');
const recs = gs.rowsToRecords_('events', ['received', 'day', 'game', 'event', 'level', 'value', 'stars', 'dur_s', 'code', 'device', 'gv', 'extra'], [xr]);
check('export turns rows into JSON records with the extra fields parsed', recs[0].table === 'events' && recs[0].extra.beam === 'x' &&
  recs[0].gv === '2026-11-05+abc1234' && !('received' in recs[0]) && !('code' in recs[0]));

// ── Anonymity checks ─────────────────────────────────────────────────────────────────────────────────
// These pin exactly what is sent and stored, so nothing that could identify a person can slip in. If one fails,
// a change is collecting something new: keep every record anonymous and update web/privacy.html to match.
const fs = require('fs');
const EVENT_KEYS = ['game', 'ev', 'day', 'dev', 'level', 'value', 'stars', 'dur_s', 'x', 'code', 'gv'];
const QUIZ_KEYS = ['kind', 'game', 'day', 'dev', 'answers', 'score', 'n', 'fun', 'learned', 'src', 'code', 'gv'];
const ANSWER_KEYS = ['q', 'obj', 'pick', 'ok'];
const kitchen = { level: 'a', value: 1, stars: 3, dur_s: 9, theta: 1, name: 'Jo Smith', email: 'a@b.co', ip: '1.2.3.4', id: 'u-123' };
const pe = kit.cleanEvent('beamline', 'fire', kitchen, { day: '2026-11-05', dev: 'desktop', code: 'X1', gv: 'v1', ip: '1.2.3.4', uid: 'u1' });
check('anonymous: an event carries only the documented fields', Object.keys(pe).every(k => EVENT_KEYS.includes(k)), Object.keys(pe).join(','));
check('anonymous: no name, email, IP or id survives in an event', !/Jo Smith|a@b\.co|1\.2\.3\.4/.test(JSON.stringify(pe)));
const pq = kit.cleanQuiz('rider', { answers: [{ q: 'a', obj: 'o', pick: 0, ok: true, name: 'Jo' }], fun: 5, learned: 4, txt_learned: 'Jo Smith', email: 'a@b.co' },
  { day: '2026-11-05', dev: 'touch', code: 'X1', gv: 'v1' });
check('anonymous: a question record carries only the documented fields', Object.keys(pq).every(k => QUIZ_KEYS.includes(k)) &&
  pq.answers.every(a => Object.keys(a).every(k => ANSWER_KEYS.includes(k))), Object.keys(pq).join(','));
check('anonymous: the collector stores exactly the documented columns', JSON.stringify(gs.TABS.events) ===
  JSON.stringify(['received', 'day', 'game', 'event', 'level', 'value', 'stars', 'dur_s', 'code', 'device', 'gv', 'extra']) &&
  JSON.stringify(gs.TABS.quiz) === JSON.stringify(['received', 'day', 'game', 'code', 'device', 'score', 'n', 'fun', 'learned', 'answers', 'source', 'gv']));
check('anonymous: the collector writes no free text, whatever a client sends', gs.cleanQuizRows({ game: 'rider', answers: [{ q: 'a', pick: 0, ok: true }],
  txt_learned: 'Jo Smith', txt_confusing: 'a@b.co' }, now).feedback === null);
const kitSrc = fs.readFileSync(path.join(root, 'web/shared/outreach.js'), 'utf8');
check('anonymous: the browser kit has no text boxes, cookies or device ids', !/el\('textarea'|type:\s*'text'|document\.cookie|navigator\.userAgent|deviceMemory|getBattery/.test(kitSrc));
const xr2 = gs.cleanEventRow({ game: 'beamline', ev: 'fire', x: { user_id: 'u42', ip: 'abc', theta: 1, phone: 'x', tag: '192.168.0.1', ref: '6035550199' } }, now);
check('anonymous: identifier-like fields and values are refused by the collector too', xr2[11] === JSON.stringify({ theta: 1 }), xr2[11]);

// manifest ids are valid collector ids and unique
check('manifest ids are valid and unique', GAMES.every(g => /^[a-z0-9-]{1,32}$/.test(g.id)) && new Set(GAMES.map(g => g.id)).size === GAMES.length);

// test traffic and page versions
check('localhost, 127.x, ::1, .local and automated browsers are test hosts', ['localhost', '127.0.0.1', '[::1]', 'mac.local', ''].every(h => kit.isTestHost(h, false)) && kit.isTestHost('mvwf.github.io', true));
check('the published site is not a test host', !kit.isTestHost('mvwf.github.io', false) && !kit.isTestHost('example.org', undefined));
{
  const now2 = new Date('2026-11-07T12:00:00Z');
  const ev = [gs.cleanEventRow({ game: 'rider', ev: 'open', day: '2026-11-07', gv: '2026-11-05+f175c6a8' }, now2),
    gs.cleanEventRow({ game: 'rider', ev: 'leave', dur_s: 90, day: '2026-11-07', gv: '2026-11-05+f175c6a8' }, now2),
    gs.cleanEventRow({ game: 'rider', ev: 'open', day: '2026-11-07', gv: '2026-11-07+abc1234' }, now2),
    gs.cleanEventRow({ game: 'rider', ev: 'open', day: '2026-11-07' }, now2)];               // no version: a local copy
  const qv = [gs.cleanQuizRows({ game: 'rider', day: '2026-11-07', gv: '2026-11-07+abc1234', answers: [{ q: 'a', obj: 'o', pick: 0, ok: true }, { q: 'b', obj: 'o', pick: 1, ok: false }] }, now2).quiz];
  const SV = gs.summarize(ev, qv, [], 30, now2), R = SV.games.rider;
  check('summary leaves out rows with no page version, and counts them', R.opens === 2 && SV.unversioned.opens === 1 && !SV.unversioned.included);
  check('summary splits each game by page version', R.versions['2026-11-05+f175c6a8'].opens === 1 && R.versions['2026-11-05+f175c6a8'].avgMinutes === 1.5 &&
    R.versions['2026-11-07+abc1234'].opens === 1 && R.versions['2026-11-07+abc1234'].answered === 2 && R.versions['2026-11-07+abc1234'].correct === 1 && !R.versions.unversioned);
  const SD = gs.summarize(ev, qv, [], 30, now2, { dev: '1' });
  check('dev: 1 includes unversioned rows under their own key', SD.games.rider.opens === 3 && SD.games.rider.versions.unversioned.opens === 1);
}
// the debrief picks a few questions, spread over objectives
{
  const items = [['a', 'x'], ['b', 'x'], ['c', 'y'], ['d', 'z'], ['e', 'z'], ['f', 'w']].map(([id, objective]) => ({ id, objective }));
  let seed = 7; const R = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const p3 = kit.pickQuiz(items, 3, new Set(), R);
  check('debrief asks 3 questions from 3 different objectives', p3.length === 3 && new Set(p3.map(q => q.objective)).size === 3);
  check('debrief skips questions already answered', kit.pickQuiz(items, 6, new Set(['a', 'b']), R).every(q => q.id !== 'a' && q.id !== 'b') && kit.pickQuiz(items, 10, new Set(), R).length === 6);
}

// ---- code designs for the frontier board ----
{
  const dctx = { day: '2026-11-07', dev: 'desktop', gv: '2026-11-07+abc1234' };
  const good = { mission: 'grid-5x5', family: 'lattice', n: 25, k: 1, d: 5, dExact: true, score: 1.000049, spec: '1.1;1.2|Z0,1,2,3', name: 'Jo', note: 'hello world' };
  const cd = kit.cleanDesign(good, dctx);
  check('cleanDesign keeps a good record, rounds the score and drops other fields', cd && cd.n === 25 && cd.d === 5 && cd.dExact === true && cd.score === 1 && cd.gv === dctx.gv && !('name' in cd) && !('note' in cd));
  const bad = [{ mission: 'Grid 5' }, { mission: 'x'.repeat(33) }, { family: 'color' }, { n: 0 }, { n: 2.5 }, { n: 2001 }, { k: 26 }, { k: -1 }, { d: 0 }, { d: 201 }, { d: undefined },
    { score: -1 }, { score: Infinity }, { score: 2e6 }, { spec: 'has space' }, { spec: '<script>' }, { spec: '' }, { spec: 'x'.repeat(2001) }, { spec: 'a@b.co' }];
  const rejected = bad.filter(b => kit.cleanDesign(Object.assign({}, good, b), dctx) === null).length;
  check(`cleanDesign rejects bad mission, family, ranges and spec characters (${rejected}/${bad.length})`, rejected === bad.length);
  check('cleanDesign allows no d when k = 0', kit.cleanDesign(Object.assign({}, good, { k: 0, d: undefined }), dctx) !== null && !('d' in kit.cleanDesign(Object.assign({}, good, { k: 0 }), dctx)));
  const now3 = new Date('2026-11-07T12:00:00Z');
  check('collector design rows reject what the page rejects', bad.every(b => gs.cleanDesignRow_(Object.assign({}, good, b), now3) === null) && gs.cleanDesignRow_(good, now3) !== null);
  check('collector design row stores no free text', !JSON.stringify(gs.cleanDesignRow_(good, now3)).includes('hello') && gs.TABS.designs.length === gs.cleanDesignRow_(good, now3).length);
  // dedup, ordering, top 25, unversioned exclusion
  const row = (o, day) => gs.cleanDesignRow_(Object.assign({}, good, { day: day || '2026-11-07', gv: dctx.gv }, o), now3);
  const rows = [row({ spec: 'A', score: 1 }, '2026-11-06'), row({ spec: 'A', score: 1 }, '2026-11-05'), row({ spec: 'A', score: 1 }),
    row({ spec: 'B', score: 1.29, d: 3 }), row({ spec: 'C', score: 1.29, d: 5 }), row({ spec: 'D', score: 1.29, d: 5 }, '2026-11-04'),
    row({ spec: 'E', score: 9, gv: '' }), row({ spec: 'F', score: 9, mission: 'other' })];
  for (let i = 0; i < 40; i++) rows.push(row({ spec: 'Z' + i, score: 0.1 + i / 1000 }));
  const B = gs.frontier_(rows, 'grid-5x5', {});
  check('board: best score first, then larger d, then the earliest day', B[0].spec === 'D' && B[1].spec === 'C' && B[2].spec === 'B' && B[3].spec === 'A', B.slice(0, 4).map(x => x.spec).join(','));
  check('board: duplicates merged with a count and the earliest day', B[3].count === 3 && B[3].day === '2026-11-05');
  check('board: top 25 only, one mission only', B.length === 25 && B.every(x => x.spec !== 'F'));
  check('board: rows with no page version are left out (dev includes them)', B.every(x => x.spec !== 'E') && gs.frontier_(rows, 'grid-5x5', { dev: true })[0].spec === 'E');
  const SD = gs.summarize([], [], [], 30, now3, {}, rows);
  check('summary counts design submissions per mission (versioned only)', SD.designs['grid-5x5'] === 46 && SD.designs.other === 1, JSON.stringify(SD.designs));
}

console.log(fails ? `\n${fails} check(s) failed` : '\nall checks passed');
process.exit(fails ? 1 : 0);
