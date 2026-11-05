// Tests for the outreach kit: the browser-side sanitising in web/shared/outreach.js and the collector's
// validation and aggregation in tools/outreach-collector/Code.gs (both load in Node without a DOM).
'use strict';
const path = require('path');
const root = path.join(__dirname, '..', '..', '..');
const kit = require(path.join(root, 'web/shared/outreach.js'));
const gs = require(path.join(root, 'tools/outreach-collector/Code.gs'));
global.window = {}; require(path.join(root, 'web/learning/manifest.js')); const GAMES = global.window.QL_GAMES;

let fails = 0;
const check = (name, ok, detail) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  · ' + detail : ''}`); if (!ok) fails++; };

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
const S = gs.summarize(rows, [qz.quiz], [], 90, now);
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
const SC = gs.summarize([], coded, [], 90, now);
const CQ = SC.codeQuiz.SCIFAIR26;
check('summary aggregates quizzes per outreach code', CQ && CQ.n === 2 && CQ.avgScore === 0.75 && CQ.avgFun === 4 && CQ.avgLearned === 3 && Object.keys(SC.codeQuiz).length === 1,
  JSON.stringify(CQ));

// a day cell that Sheets turned into a Date still lands on the right day
const dated = gs.cleanEventRow({ game: 'drum', ev: 'open', day: '2026-11-03' }, now); dated[1] = new Date(2026, 10, 3);
const S3 = gs.summarize([dated], [], [], 7, now);
check('summary reads day cells that Sheets stored as dates', S3.byDay['2026-11-03'] && S3.byDay['2026-11-03'].drum === 1, JSON.stringify(S3.byDay));

// the private review feed was removed with the free text: the collector exports no review helpers
check('no review feed is exported', !('reviewFeed' in gs) && !('reviewKeyOk' in gs));

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
const S2 = gs.summarize(rows2, [ing.quiz], [], 90, now);
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

// manifest ids are valid collector ids and unique
check('manifest ids are valid and unique', GAMES.every(g => /^[a-z0-9-]{1,32}$/.test(g.id)) && new Set(GAMES.map(g => g.id)).size === GAMES.length);

console.log(fails ? `\n${fails} check(s) failed` : '\nall checks passed');
process.exit(fails ? 1 : 0);
