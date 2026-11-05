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

// quiz: score is recomputed from the answers, free text is scrubbed
const q = kit.cleanQuiz('copter', { answers: [{ q: 'q1', obj: 'o1', pick: 0, ok: true }, { q: 'q2', obj: 'o2', pick: 2, ok: false }],
  fun: 5, learned: 4, txt_learned: 'rotations! me@x.org', txt_confusing: '' }, ctx);
check('cleanQuiz recomputes score and scrubs text', q.score === 1 && q.n === 2 && q.fun === 5 && !/@/.test(q.txt_learned));

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
check('collector splits quiz and free text, scrubbing the text', qz.quiz[5] === 1 && qz.feedback && /\[removed\]/.test(qz.feedback[5]));
const S = gs.summarize(rows, [qz.quiz], [qz.feedback], 90, now);
const B = S.games.beamline;
check('summary counts opens, level funnel and average minutes', B.opens === 1 && B.levels.half.starts === 1 && B.levels.half.completes === 1 && B.avgMinutes === 2);
check('summary skips rows outside the window', !S.games.rider);
check('summary has quiz accuracy per item and ratings', S.quiz.beamline.items.q1.correct === 1 && S.quiz.beamline.avgScore === 0.5 && S.quiz.beamline.avgFun === 4);
check('summary counts free text but never includes it', S.feedbackCount.beamline === 1 && !JSON.stringify(S).includes('phase sets the axis'));
check('summary splits by outreach code', S.codes.SCIFAIR26 === 1);

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
const qi = kit.cleanQuiz('rider', { answers: [{ q: 'q1', obj: 'o1', pick: 2, ok: true }], src: 'ingame' }, ctx);
check('cleanQuiz keeps the in-game source', qi.src === 'ingame');

// manifest ids are valid collector ids and unique
check('manifest ids are valid and unique', GAMES.every(g => /^[a-z0-9-]{1,32}$/.test(g.id)) && new Set(GAMES.map(g => g.id)).size === GAMES.length);

console.log(fails ? `\n${fails} check(s) failed` : '\nall checks passed');
process.exit(fails ? 1 : 0);
