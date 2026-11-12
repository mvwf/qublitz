#!/usr/bin/env node
/* Weekly feedback digest for Qublitz: what people told us and what the anonymous counts show, ranked by what to fix
 * first. The same loop as Q-Vault's scripts/feedback_digest.py (docs/FEEDBACK_LOOP.md).
 *
 * Reads, all public or in the repository:
 *   1. the outreach collector's summary (web/shared/outreach-config.js endpoint, GET ?summary=1&days=N),
 *   2. the open GitHub issues of mvwf/qublitz,
 *   3. web/outreach/improvements.json, the log of changes made because of feedback,
 * and ranks them. Counts are flagged by the dashboard's own rules (web/outreach/insights.js: hard questions, levels
 * people do not finish, stop hotspots, low ratings, short visits), plus two signals from the page feedback strip in
 * web/shared/outreach.js: script errors (js_error) and pages marked not useful (helpful_no). Issues rank above counts,
 * because someone took the time to write them.
 *
 *   node scripts/feedback_digest.js                      # last 7 days, markdown to stdout
 *   node scripts/feedback_digest.js --days 30 --out digest.md --json flags.json
 *   node scripts/feedback_digest.js --summary file.json   # offline, from a saved summary
 * Uses GITHUB_TOKEN (or GH_TOKEN) for issues, else the gh command when installed. It never writes to GitHub.
 */
'use strict';
const fs = require('fs'), path = require('path'), cp = require('child_process');
const ROOT = path.resolve(__dirname, '..');
const REPO = process.env.GITHUB_REPOSITORY || 'mvwf/qublitz';

const RULES = { UNHELPFUL: 0.4, UNHELPFUL_MIN_VOTES: 5 };
const SKIP_LABELS = new Set(['feedback-digest', 'dependencies']);
const ISSUE_KINDS = [  // [label, kind, base priority, what to do]
  ['bug', 'bug', 2.5, 'Reproduce with the page link in the issue, fix, and reply with the pull request.'],
  ['feedback', 'feedback', 2.0, "Reply within a few days. If it is actionable, open a pull request that says 'Feedback #N'."],
  ['enhancement', 'idea', 1.0, 'Decide yes, later or no, and say so in the issue.'],
];

// Games and their learning files, loaded through a stub QL exactly as the dashboard does.
function loadSpecs() {
  global.window = global.window || {};
  require(path.join(ROOT, 'web/learning/manifest.js'));
  const GAMES = global.window.QL_GAMES || [], SPECS = {}, TITLES = {};
  let last = null;
  global.window.QL = global.QL = { learning(s) { if (s && typeof s.id === 'string') { SPECS[s.id] = s; last = s; } },
    addItems(list) { if (!last) return; last.quiz = last.quiz || []; (list || []).forEach(it => { if (it && it.id && !last.quiz.some(q => q.id === it.id)) last.quiz.push(it); }); } };
  GAMES.forEach(g => {
    TITLES[g.id] = g.title;
    const f = path.join(ROOT, 'web/learning', g.id + '.js');
    if (fs.existsSync(f)) { try { require(f); } catch (e) { /* a learning file that needs the DOM is skipped */ } }
  });
  return { GAMES, SPECS, TITLES };
}

function endpoint() {
  const m = /endpoint:\s*'([^']*)'/.exec(fs.readFileSync(path.join(ROOT, 'web/shared/outreach-config.js'), 'utf8'));
  return m ? m[1] : '';
}
async function getJSON(url, headers) {
  const r = await fetch(url, { headers: { 'User-Agent': 'qublitz-feedback-digest', ...(headers || {}) }, redirect: 'follow' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
async function summary(days, file) {
  if (file) return [JSON.parse(fs.readFileSync(file, 'utf8')), null];
  const url = endpoint();
  if (!url) return [null, 'Usage counting is switched off (web/shared/outreach-config.js has no endpoint).'];
  try { return [await getJSON(`${url}?summary=1&days=${days}`), null]; } catch (e) { return [null, `The outreach collector did not answer (${e.message}).`]; }
}
async function issues() {
  const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN, p = `repos/${REPO}/issues?state=open&per_page=100`;
  let rows;
  try {
    if (token) rows = await getJSON(`https://api.github.com/${p}`, { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json' });
    else rows = JSON.parse(cp.execFileSync('gh', ['api', p], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }));
  } catch (e) { return [[], 'Issues could not be read (no GITHUB_TOKEN, and gh failed or is not installed).']; }
  const now = Date.now();
  return [rows.filter(r => !r.pull_request).map(r => ({ number: r.number, title: r.title, url: r.html_url,
    labels: (r.labels || []).map(l => l.name).sort(), age_days: Math.floor((now - Date.parse(r.created_at)) / 864e5), comments: r.comments || 0 }))
    .filter(r => !r.labels.some(l => SKIP_LABELS.has(l))), null];
}

const gapAbove = (v, t) => Math.max(0, Math.min(1, (v - t) / t));

// Flags from the counts: the dashboard's insights plus the page feedback strip's errors and votes.
function flagsFromSummary(S, specs) {
  if (!S || !S.games) return [];
  const IN = require(path.join(ROOT, 'web/outreach/insights.js'));
  const flags = IN.insights(S, specs.SPECS, specs.TITLES).map(f => ({ ...f, kind: f.kind }));
  const name = id => specs.TITLES[id] || id;
  Object.keys(S.games).forEach(g => {
    const ev = S.games[g].events || {};
    if (ev.js_error) flags.push({ kind: 'error', game: g, ref: 'js_error', n: ev.js_error, impact: ev.js_error * 2,
      title: `${name(g)}: ${ev.js_error} page load${ev.js_error === 1 ? '' : 's'} hit a script error`,
      evidence: `${ev.js_error} js_error events (the message is never sent).`,
      fix: 'Open the page with the browser console, reproduce, fix, and add a check to the page\'s engine test.' });
    const yes = ev.helpful_yes || 0, no = ev.helpful_no || 0, n = yes + no;
    if (n >= RULES.UNHELPFUL_MIN_VOTES && no / n >= RULES.UNHELPFUL) flags.push({ kind: 'page', game: g, ref: 'helpful', n,
      impact: Math.round(n * gapAbove(no / n, RULES.UNHELPFUL) * 100) / 100,
      title: `${name(g)}: ${Math.round(no / n * 100)} % of ${n} votes say not useful`,
      evidence: `${no} No and ${yes} Yes on 'Was this page useful?'.`,
      fix: 'Read the feedback issues for this page, then watch one new player. Usually the first screen does not say what to do or why.' });
  });
  return flags;
}
function flagsFromIssues(rows) {
  return rows.map(r => {
    const k = ISSUE_KINDS.find(x => r.labels.includes(x[0]));
    const [, kind, base, todo] = k || ['', 'other', 1.5, 'Read it, label it, and reply.'];
    const impact = base + (r.comments === 0 ? Math.min(4, r.age_days / 7) : 0);
    return { kind: 'issue:' + kind, ref: '#' + r.number, n: 1, impact: Math.round(impact * 100) / 100, title: `#${r.number} ${r.title}`, url: r.url,
      evidence: `open ${r.age_days} days, ${r.comments} comments, labels: ${r.labels.join(', ') || 'none'}.`, fix: todo };
  });
}
function improvements(since) {
  try {
    const j = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/outreach/improvements.json'), 'utf8'));
    return (Array.isArray(j) ? j : []).filter(r => r && !r.demo && String(r.date || '') >= since);
  } catch (e) { return []; }
}
const rank = fs_ => fs_.slice().sort((a, b) => b.impact - a.impact || String(a.title).localeCompare(String(b.title)));

function markdown(days, S, notes, flags, imps, specs) {
  const today = new Date().toISOString().slice(0, 10), since = new Date(Date.now() - days * 864e5).toISOString().slice(0, 10);
  const L = [`# Qublitz feedback digest, ${since} to ${today}`, '',
    'What players told us and what the anonymous counts show, ranked by what to fix first. Written by `scripts/feedback_digest.js`; ' +
    'the same numbers, with charts, are on the [outreach dashboard](https://mvwf.github.io/qublitz/outreach/dashboard.html). To act on it with Claude Code, run `/improve qublitz`.', ''];
  if (S && S.games) {
    const opens = Object.values(S.games).reduce((a, g) => a + (g.opens || 0), 0);
    L.push(`**${opens} page opens, ${Math.round(S.totalMinutes || 0)} active minutes** in the last ${days} days.`, '');
  }
  notes.filter(Boolean).forEach(n => L.push(`> ${n}`, ''));
  L.push('## Act on these first', '');
  if (flags.length) {
    flags.slice(0, 12).forEach((f, i) => L.push(`${i + 1}. **${f.url ? `[${f.title}](${f.url})` : f.title}** (${f.kind}, impact ${f.impact})`, `   ${f.evidence} *Next:* ${f.fix}`));
    if (flags.length > 12) L.push('', `...and ${flags.length - 12} smaller items.`);
  } else L.push('Nothing crosses a threshold. Flags only fire once enough people were measured.');
  if (S && S.games) {
    L.push('', '## Pages', '', '| Page | Opens | Useful: yes / no | Script errors |', '|---|---:|---:|---:|');
    Object.entries(S.games).sort((a, b) => (b[1].opens || 0) - (a[1].opens || 0)).slice(0, 12).forEach(([g, G]) => {
      const ev = G.events || {};
      L.push(`| ${specs.TITLES[g] || g} | ${G.opens || 0} | ${ev.helpful_yes || 0} / ${ev.helpful_no || 0} | ${ev.js_error || 0} |`);
    });
  }
  L.push('', '## Changes made because of feedback', '');
  if (imps.length) imps.forEach(r => L.push(`- ${r.date}: **${specs.TITLES[r.game] || r.game || ''}** ${r.change || ''} Because: ${r.because || ''}${r.pr ? ` (${r.pr})` : ''}`));
  else L.push('None logged in this period. Add each change to `web/outreach/improvements.json` when its pull request merges; the dashboard then compares before and after.');
  L.push('', '## Rules', '', '- Questions, levels, stop hotspots, ratings, debrief drop-off and short visits: the dashboard\'s rules in `web/outreach/insights.js`.',
    `- Page: at least ${RULES.UNHELPFUL * 100} % No on 'Was this page useful?', with ${RULES.UNHELPFUL_MIN_VOTES} or more votes. Script error: any.`,
    '- Issues: bugs first, then feedback, then ideas, each climbing one point per week without a reply.');
  return L.join('\n') + '\n';
}

async function main() {
  const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
  const days = +(arg('--days') || 7), specs = loadSpecs();
  const [S, noteS] = await summary(days, arg('--summary'));
  const [iss, noteI] = await issues();
  const flags = rank(flagsFromIssues(iss).concat(flagsFromSummary(S, specs)));
  const since = new Date(Date.now() - days * 864e5).toISOString().slice(0, 10);
  const md = markdown(days, S, [noteS, noteI], flags, improvements(since), specs);
  if (arg('--out')) fs.writeFileSync(arg('--out'), md); else process.stdout.write(md);
  if (arg('--json')) fs.writeFileSync(arg('--json'), JSON.stringify({ site: 'qublitz', repo: REPO, days, flags }, null, 1));
}

module.exports = { flagsFromSummary, flagsFromIssues, markdown, loadSpecs, rank, RULES };
if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });
