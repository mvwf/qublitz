// Checks the ranking of scripts/feedback_digest.js on made-up counts and issues. No network.
//   node web/shared/tests/feedback-digest.test.js
'use strict';
const path = require('path');
const D = require(path.join(__dirname, '../../../scripts/feedback_digest.js'));
let fails = 0;
const check = (name, ok) => { console.log((ok ? 'ok   ' : 'FAIL ') + name); if (!ok) fails++; };

const specs = D.loadSpecs();
check('learning files load through the stub', Object.keys(specs.SPECS).length > 5 && specs.TITLES.fridge === 'Fridge Noise Explorer');

const S = { games: {
  fridge: { opens: 40, leaves: 30, events: { helpful_yes: 2, helpful_no: 5, js_error: 3 }, levels: {}, quits: {} },
  resonator: { opens: 20, leaves: 10, events: { helpful_yes: 8, helpful_no: 1 }, levels: {}, quits: {} },
  drum: { opens: 5, leaves: 5, events: { helpful_no: 3 }, levels: {}, quits: {} } }, quiz: {} };
const f = D.flagsFromSummary(S, specs);
const by = (g, k) => f.find(x => x.game === g && x.kind === k);
check('script errors are flagged', by('fridge', 'error') && by('fridge', 'error').n === 3);
check('5 of 7 No is flagged with impact 7 x (0.714 - 0.4) / 0.4 = 5.5', by('fridge', 'page') && Math.abs(by('fridge', 'page').impact - 5.5) < 0.01);
check('a mostly useful page is not flagged', !by('resonator', 'page'));
check('three votes are too few', !by('drum', 'page'));
check('no summary gives no flags', D.flagsFromSummary(null, specs).length === 0);

const I = [{ number: 3, title: 'Broken slider', url: 'u3', labels: ['bug'], age_days: 21, comments: 0 },
  { number: 4, title: 'Idea', url: 'u4', labels: ['enhancement'], age_days: 2, comments: 1 }];
const g = D.rank(D.flagsFromIssues(I).concat(f));
const pos = ref => g.findIndex(x => x.ref === ref);
check('three script errors (impact 6) rank first', g[0].kind === 'error' && g[0].impact === 6);
check('an unanswered three-week-old bug climbs to 2.5 + 3 = 5.5', g[pos('#3')].impact === 5.5);
check('the bug ranks above an answered idea', pos('#3') < pos('#4'));
const md = D.markdown(7, S, [], g, [], specs);
check('the digest links the issue', md.includes('[#3 Broken slider](u3)'));
check('the page table shows votes and errors', md.includes('| Fridge Noise Explorer | 40 | 2 / 5 | 3 |'));

console.log(fails ? `${fails} FAILED` : 'all feedback digest checks pass');
process.exit(fails ? 1 : 0);
