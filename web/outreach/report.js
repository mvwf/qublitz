/* report.js: the dashboard's numbers for proposals, redesign briefs and before/after comparisons, computed from the
 * collector's summary (Code.gs summarize()). Pure: no DOM. The dashboard loads it as a script (window.QL_REPORT);
 * web/shared/tests/outreach.test.js requires it. Needs insights.js (window.QL_INSIGHTS, or require in Node).
 *
 * Everything here is aggregate. Visits are page visits, not people: there are no identifiers, so one person who
 * comes back three times counts three times. Active minutes come from the visit-end ("leave") reports, which some
 * browsers never send, so hours are a lower bound.
 */
(function (root) {
  'use strict';
  const IN = (typeof module !== 'undefined' && module.exports) ? require('./insights.js') : root.QL_INSIGHTS;
  const plain = IN.plain;

  const COMPARE = {
    // Below these counts on either side a before/after row is greyed out as "too few answers to compare".
    MIN_N: 20,          // answers, level starts, visits, debrief openings
    MIN_RATINGS: 10,    // debriefs with a rating
  };
  const SKIP = ['hub'];   // not an activity: left out of activity totals (its visits are reported separately)

  const sum = a => a.reduce((x, y) => x + y, 0);
  const pct = x => x == null || !isFinite(x) ? '–' : Math.round(x * 100) + '%';
  const fix1 = x => x == null || !isFinite(x) ? '–' : (Math.round(x * 10) / 10).toFixed(1);
  const fmt = n => n == null ? '–' : Math.round(n).toLocaleString('en-US');
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const dLong = d => d ? `${MON[+d.slice(5, 7) - 1]} ${+d.slice(8, 10)}, ${d.slice(0, 4)}` : '';
  const mLong = m => `${MON[+m.slice(5, 7) - 1]} ${m.slice(0, 4)}`;
  const wavg = (list, key, w) => { let t = 0, n = 0; list.forEach(q => { if (q && q[key] != null && q[w]) { t += q[key] * q[w]; n += q[w]; } }); return n ? t / n : null; };

  /* Per learning objective: answers, share right, and the item with the lowest share (with its top wrong answer). */
  function objectiveStats(S, specs, titles) {
    const out = [], sp = specs || {}, tt = titles || {};
    Object.keys((S && S.quiz) || {}).forEach(game => {
      const spec = sp[game] || {}, items = S.quiz[game].items || {}, groups = {};
      Object.keys(items).forEach(qid => {
        const q = (spec.quiz || []).find(x => x.id === qid), ob = (q && q.objective) || items[qid].objective || '';
        (groups[ob] = groups[ob] || []).push(qid);
      });
      Object.keys(groups).forEach(ob => {
        const ids = groups[ob], n = sum(ids.map(i => items[i].n)), correct = sum(ids.map(i => items[i].correct));
        if (!n) return;
        const o = (spec.objectives || []).find(x => x.id === ob);
        let worst = null;
        // the weakest question among those with enough answers to mean something (else among all)
        const solid = ids.filter(i => items[i].n >= IN.RULES.QUIZ_MIN_ANSWERS), pool = solid.length ? solid : ids;
        pool.forEach(i => { const it = items[i], a = it.n ? it.correct / it.n : 1; if (!worst || a < worst.acc) worst = { id: i, acc: a, n: it.n }; });
        const q = (spec.quiz || []).find(x => x.id === worst.id);
        worst.question = q ? plain(q.q) : '';
        worst.wrong = IN.topWrong(items[worst.id], q);
        out.push({ game, gameTitle: tt[game] || spec.title || game, objective: ob, text: o ? plain(o.text) : (ob || 'No objective given'),
          n, correct, acc: correct / n, worst });
      });
    });
    return out.sort((a, b) => a.acc - b.acc);
  }

  /* The headline numbers for a proposal. */
  function numbers(S, specs, titles) {
    const games = (S && S.games) || {}, quiz = (S && S.quiz) || {};
    const acts = Object.keys(games).filter(g => SKIP.indexOf(g) < 0 && games[g].opens > 0);
    const Q = Object.keys(quiz).filter(g => SKIP.indexOf(g) < 0).map(g => quiz[g]);
    const items = Q.flatMap(q => Object.values(q.items || {}));
    const months = Object.keys((S && S.byMonth) || {}).sort().map(m => {
      const M = S.byMonth[m], ids = Object.keys(M).filter(g => SKIP.indexOf(g) < 0);
      return { month: m, visits: sum(ids.map(g => M[g].opens)), minutes: sum(ids.map(g => M[g].minutes)),
        completes: sum(ids.map(g => M[g].completes)), debriefs: sum(ids.map(g => M[g].debriefs)), ingame: sum(ids.map(g => M[g].ingame)) };
    });
    const codes = Object.keys((S && S.codes) || {}).sort((a, b) => S.codes[b] - S.codes[a]).map(c => {
      const T = (S.codeTotals || {})[c] || {}, CQ = (S.codeQuiz || {})[c] || {};
      return { code: c, visits: S.codes[c], minutes: T.minutes || 0, completes: T.completes || 0, debriefs: CQ.n || 0, ingame: CQ.ingame || 0,
        correct: CQ.avgScore == null ? null : CQ.avgScore, fun: CQ.avgFun == null ? null : CQ.avgFun, learned: CQ.avgLearned == null ? null : CQ.avgLearned };
    });
    return {
      since: S && S.since, until: S && (S.until || (S.generated || '').slice(0, 10)), code: (S && S.code) || '', demo: !!(S && S.demo),
      visits: sum(acts.map(g => games[g].opens)), hubVisits: (games.hub && games.hub.opens) || 0,
      activities: acts.sort((a, b) => games[b].opens - games[a].opens).map(g => ({ id: g, title: (titles || {})[g] || g, visits: games[g].opens, minutes: games[g].totalMinutes || 0 })),
      hours: sum(acts.map(g => games[g].totalMinutes || 0)) / 60,
      completes: sum(acts.map(g => games[g].completes || 0)),
      ingame: sum(Q.map(q => q.ingame || 0)), debriefs: sum(Q.map(q => q.n || 0)),
      answers: sum(items.map(i => i.n)), correct: items.length ? sum(items.map(i => i.correct)) / Math.max(1, sum(items.map(i => i.n))) : null,
      fun: wavg(Q, 'avgFun', 'n'), learned: wavg(Q, 'avgLearned', 'n'),
      labArrivals: sum(Object.values((S && S.labFrom) || {})),
      objectives: objectiveStats(S, specs, titles), months, codes,
    };
  }

  const CAVEATS = [
    'Visits are page visits, not unique people: nothing identifies a visitor, so return visits count again.',
    'Active minutes are counted while the page is visible and reported when a visit ends; some browsers never send that report, so hours are a lower bound.',
    'Visitors with Do Not Track or Global Privacy Control on, or who opted out, are not counted. Counts are indicative, not audited.',
    'Question results mix in-game answers and debrief answers; ratings come from the debrief only.',
  ];

  /* A ready-to-paste paragraph plus bullets, with the date range and the caveats. */
  function reportText(S, specs, titles) {
    const N = numbers(S, specs, titles);
    const scope = N.code ? ` at the outreach event “${N.code}”` : '';
    const lines = [];
    if (N.demo) lines.push('[DEMO DATA: synthetic numbers, not real usage]', '');
    lines.push(`From ${dLong(N.since)} to ${dLong(N.until)}, the Qublitz interactive demos (https://mvwf.github.io/qublitz/) recorded ${fmt(N.visits)} visits${scope} to ` +
      `${N.activities.length} learning activities, about ${fmt(N.hours)} hours of active use. Visitors completed ${fmt(N.completes)} levels or missions, ` +
      `answered ${fmt(N.answers)} questions on the learning objectives (${pct(N.correct)} right), and rated the activities ` +
      `${fix1(N.fun)}/5 for fun and ${fix1(N.learned)}/5 for how much they learned.`);
    lines.push('');
    lines.push(`- Visits: ${fmt(N.visits)} to activities (plus ${fmt(N.hubVisits)} to the demo hub); these are page visits, not unique people`);
    lines.push(`- Activities used: ${N.activities.length}${N.activities.length ? ' (' + N.activities.slice(0, 6).map(a => a.title).join(', ') + (N.activities.length > 6 ? ', …' : '') + ')' : ''}`);
    lines.push(`- Active engagement: ${fmt(N.hours)} hours`);
    lines.push(`- Levels or missions completed: ${fmt(N.completes)}`);
    lines.push(`- Questions answered: ${fmt(N.ingame)} in-game and ${fmt(N.debriefs)} end-of-activity debriefs; ${pct(N.correct)} of ${fmt(N.answers)} answers right`);
    lines.push(`- Ratings: fun ${fix1(N.fun)}/5, learned ${fix1(N.learned)}/5 (from ${fmt(N.debriefs)} debriefs)`);
    lines.push(`- Visits to the Virtual Qubit Lab from another activity: ${fmt(N.labArrivals)}`);
    const best = N.objectives.filter(o => o.n >= COMPARE.MIN_N).sort((a, b) => b.acc - a.acc).slice(0, 3);
    if (best.length) lines.push(`- Strongest learning objectives: ${best.map(o => `${o.text} (${pct(o.acc)} of ${fmt(o.n)})`).join('; ')}`);
    if (N.months.length > 1) lines.push(`- Visits by month: ${N.months.map(m => `${mLong(m.month)} ${fmt(m.visits)}`).join(', ')}`);
    if (N.codes.length) lines.push(`- Outreach events: ${N.codes.slice(0, 6).map(c => `${c.code}: ${fmt(c.visits)} visits, ${fmt(c.minutes)} min` +
      (c.correct != null ? `, ${pct(c.correct)} right` : '') + (c.fun != null ? `, fun ${fix1(c.fun)}/5` : '')).join('; ')}`);
    lines.push('');
    lines.push('Notes: ' + CAVEATS.join(' '));
    return lines.join('\n');
  }

  const csvCell = v => { const s = v == null ? '' : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const csv = (head, rows, demo) => (demo ? '# DEMO DATA: synthetic, not real usage\n' : '') + [head].concat(rows).map(r => r.map(csvCell).join(',')).join('\n') + '\n';
  function monthsCsv(S, titles) {
    const rows = [];
    Object.keys((S && S.byMonth) || {}).sort().forEach(m => Object.keys(S.byMonth[m]).sort().forEach(g => {
      const x = S.byMonth[m][g]; rows.push([m, g, (titles || {})[g] || g, x.opens, x.minutes, x.completes, x.debriefs, x.ingame]);
    }));
    return csv(['month', 'game', 'title', 'visits', 'active_minutes', 'levels_completed', 'debriefs', 'ingame_answers'], rows, S && S.demo);
  }
  function codesCsv(S) {
    const N = numbers(S);
    return csv(['event_code', 'visits', 'active_minutes', 'levels_completed', 'debriefs', 'ingame_answers', 'share_right', 'avg_fun', 'avg_learned'],
      N.codes.map(c => [c.code, c.visits, c.minutes, c.completes, c.debriefs, c.ingame, c.correct == null ? '' : c.correct, c.fun == null ? '' : c.fun, c.learned == null ? '' : c.learned]), S && S.demo);
  }

  /* Suggested next experiment for one insight flag, with a success criterion. */
  function experiment(f, R) {
    switch (f.kind) {
      case 'quiz': return `Question ${f.ref}: add an in-game moment where the misconception${f.misconception ? ` “${f.misconception}”` : ''} visibly fails before the debrief asks. Success: at least ${pct(R.QUIZ_ACCURACY + 0.1)} right over ${COMPARE.MIN_N}+ answers.`;
      case 'level': return `Level “${f.ref}”: add a hint after the first failed try, or an easier step before it. Success: at least ${pct(R.LEVEL_COMPLETION + 0.1)} of ${COMPARE.MIN_N}+ starts completed.`;
      case 'stop': return `“${f.ref}”: add a checkpoint or a smaller step before it, so a failed try is not the end of the visit. Success: under ${pct(R.STOP_RATE)} of the visits that reach it stop there.`;
      case 'rating': return f.ref === 'fun' ? `Fun: give a first win within a minute and a visible goal or score. Success: fun at least 3.5/5 over ${COMPARE.MIN_RATINGS}+ debriefs.`
        : `Learning: name the idea in the level brief and in the feedback after each try. Success: learned at least 3.5/5 over ${COMPARE.MIN_RATINGS}+ debriefs.`;
      case 'dropoff': return `Debrief: move one or two questions into the game (QL.answer) and cut the rest. Success: at least ${pct(R.QUIZ_SUBMIT + 0.1)} of openings sent.`;
      case 'time': return `Onboarding: one clear first action in the first ten seconds. Success: average visit at least ${R.MINUTES * 2} min over ${COMPARE.MIN_N}+ visits.`;
      default: return '';
    }
  }

  /* A markdown redesign brief for one game, built from its flags and numbers (aggregate counts only). */
  function brief(S, specs, titles, game, opts) {
    const o = opts || {}, tt = titles || {}, sp = specs || {}, R = IN.RULES;
    const name = tt[game] || (sp[game] && sp[game].title) || game;
    const G = (S.games || {})[game] || {}, Q = (S.quiz || {})[game] || {};
    const flags = IN.insights(S, sp, tt).filter(f => f.game === game);
    flags.forEach(f => { if (f.kind === 'quiz') { const m = /possible misconception: “([^”]*)”/.exec(f.evidence); if (m) f.misconception = m[1]; } });
    const L = [];
    const title = `Redesign brief: ${name}` + (o.demo || S.demo ? ' [DEMO DATA]' : '');
    if (o.demo || S.demo) L.push('> **DEMO DATA**: synthetic numbers from the dashboard demo, not real usage.', '');
    L.push(`Data: ${dLong(S.since)} – ${dLong(S.until || (S.generated || '').slice(0, 10))}${S.code ? `, outreach event ${S.code}` : ''}, from the outreach usage dashboard. ` +
      'Aggregate counts only; visits are page visits, not people.', '');
    L.push('### Numbers');
    L.push(`- Visits: ${fmt(G.opens || 0)}; started: ${G.opens ? pct((G.starts || 0) / G.opens) : '–'}; levels completed: ${fmt(G.completes || 0)}`);
    L.push(`- Average visit: ${G.avgMinutes == null ? '–' : fix1(G.avgMinutes) + ' min'}; total active time: ${fmt((G.totalMinutes || 0) / 60)} h`);
    L.push(`- Debriefs: ${fmt(Q.n || 0)} sent of ${fmt(G.quizOpens || 0)} opened; in-game answers: ${fmt(Q.ingame || 0)}; ${pct(Q.avgScore)} right overall`);
    L.push(`- Ratings: fun ${fix1(Q.avgFun)}/5, learned ${fix1(Q.avgLearned)}/5 (from ${fmt(Q.n || 0)} debriefs)`);
    const quits = G.quits || {}, qt = sum(Object.values(quits));
    if (qt) {
      L.push('', '### Where visits end');
      Object.keys(quits).sort((a, b) => quits[b] - quits[a]).slice(0, 5).forEach(p => {
        const lv = (G.levels || {})[p.split('|')[0]];
        L.push(`- ${p}: ${pct(quits[p] / qt)} of visit ends (${fmt(quits[p])})` + (lv ? `; ${fmt(lv.completes)} of ${fmt(lv.starts)} who started it finished` : ''));
      });
    }
    const obj = objectiveStats({ quiz: { [game]: Q.items ? Q : { items: {} } } }, sp, tt).slice(0, 3);
    if (obj.length) {
      L.push('', '### Weakest learning objectives');
      obj.forEach(x => L.push(`- ${x.text}: ${pct(x.acc)} right of ${fmt(x.n)} answers. Weakest question ${x.worst.id}` +
        (x.worst.question ? ` (“${x.worst.question}”)` : '') + `: ${pct(x.worst.acc)}` +
        (x.worst.wrong ? `; most-picked wrong answer “${x.worst.wrong.text}” (${pct(x.worst.wrong.share)})` : '')));
    }
    L.push('', '### Flags');
    if (flags.length) flags.slice(0, 6).forEach(f => L.push(`- ${f.title.replace(name + ': ', '')}. ${f.evidence}. Fix type: ${f.fix}`));
    else L.push('- No rule fires for this game in this period (rules wait for enough people; see the dashboard for the thresholds).');
    L.push('', '### Suggested next experiments');
    const ex = flags.slice(0, 4).map(f => experiment(f, R)).filter(Boolean);
    if (!ex.length) ex.push(obj.length ? `Add an in-game question for “${obj[0].text}” so it is practised before the debrief.` : 'Add in-game questions (QL.answer) so the learning objectives are measured.',
      'Try one harder optional level or a time challenge, and watch fun and visit depth.');
    ex.forEach((e, i) => L.push(`${i + 1}. ${e}`));
    L.push('', '### After the change',
      '- Log it in `web/outreach/improvements.json` with the date it went live, the game, and the level or objective it targets.',
      '- Compare before and after on the dashboard (Before / after), waiting until both sides have enough answers.');
    return { title, body: L.join('\n') };
  }

  /* A new-issue link with the title, body and `feedback` label prefilled, kept under `max` characters. */
  function issueUrl(title, body, max) {
    const lim = max || 7000, base = 'https://github.com/mvwf/qublitz/issues/new?labels=feedback&title=' + encodeURIComponent(title) + '&body=';
    const note = '\n\n_(Truncated to fit in a link: copy the full brief from the dashboard.)_';
    if ((base + encodeURIComponent(body)).length <= lim) return { url: base + encodeURIComponent(body), truncated: false };
    const lines = body.split('\n');
    while (lines.length && (base + encodeURIComponent(lines.join('\n') + note)).length > lim) lines.pop();
    let b = lines.join('\n');
    while (b.length && (base + encodeURIComponent(b + note)).length > lim) b = b.slice(0, -50);
    return { url: base + encodeURIComponent(b + note), truncated: true };
  }

  /* Before/after rows for one change. target: { game, level?, objective? }. Each row: { key, label, unit, before: {v, n},
   * after: {v, n}, enough }. A row is "enough" only when both sides reach the minimum count. */
  function compare(before, after, target, specs) {
    const t = target || {}, sp = specs || {}, rows = [];
    const side = (S, fn) => { try { return fn(S || {}); } catch (e) { return { v: null, n: 0 }; } };
    const G = S => (S.games || {})[t.game] || {}, Q = S => (S.quiz || {})[t.game] || {};
    const add = (key, label, unit, fn, min) => {
      const b = side(before, fn), a = side(after, fn);
      rows.push({ key, label, unit, before: b, after: a, enough: b.n >= min && a.n >= min && b.v != null && a.v != null });
    };
    if (t.level) add('level', `Completion of level “${t.level}”`, 'pct', S => { const L = (G(S).levels || {})[t.level] || { starts: 0, completes: 0 }; return { v: L.starts ? L.completes / L.starts : null, n: L.starts }; }, COMPARE.MIN_N);
    else add('levels', 'Level completion (all levels)', 'pct', S => { const L = Object.values(G(S).levels || {}); const n = sum(L.map(x => x.starts)); return { v: n ? sum(L.map(x => x.completes)) / n : null, n }; }, COMPARE.MIN_N);
    const spec = sp[t.game] || {};
    const objOf = (qid, it) => { const q = (spec.quiz || []).find(x => x.id === qid); return (q && q.objective) || it.objective || ''; };
    const o = t.objective && (spec.objectives || []).find(x => x.id === t.objective);
    add('correct', t.objective ? `Right on “${o ? plain(o.text) : t.objective}”` : 'Questions right (all)', 'pct', S => {
      const items = Q(S).items || {}, ids = Object.keys(items).filter(q => !t.objective || objOf(q, items[q]) === t.objective);
      const n = sum(ids.map(q => items[q].n)); return { v: n ? sum(ids.map(q => items[q].correct)) / n : null, n };
    }, COMPARE.MIN_N);
    add('fun', 'Fun rating', 'rating', S => ({ v: Q(S).avgFun == null ? null : Q(S).avgFun, n: Q(S).n || 0 }), COMPARE.MIN_RATINGS);
    add('learned', 'Learned rating', 'rating', S => ({ v: Q(S).avgLearned == null ? null : Q(S).avgLearned, n: Q(S).n || 0 }), COMPARE.MIN_RATINGS);
    add('minutes', 'Average visit', 'min', S => ({ v: G(S).avgMinutes == null ? null : G(S).avgMinutes, n: G(S).leaves || 0 }), COMPARE.MIN_N);
    add('debrief', 'Debriefs sent of opened', 'pct', S => ({ v: G(S).quizOpens ? Math.min(1, (Q(S).n || 0) / G(S).quizOpens) : null, n: G(S).quizOpens || 0 }), COMPARE.MIN_N);
    return rows;
  }

  /* Equal windows either side of a change date, ending today: before = the same number of days just before it. */
  function windows(date, today, maxDays) {
    const day = 86400000, d = Date.parse(date + 'T00:00:00Z'), t = Date.parse(today + 'T00:00:00Z');
    const len = Math.max(1, Math.min(maxDays || 180, Math.round((t - d) / day) + 1));
    const iso = x => new Date(x).toISOString().slice(0, 10);
    return { before: { from: iso(d - len * day), to: iso(d - day) }, after: { from: date, to: iso(Math.min(t, d + (len - 1) * day)) }, days: len };
  }

  const api = { COMPARE, CAVEATS, numbers, objectiveStats, reportText, monthsCsv, codesCsv, brief, issueUrl, compare, windows, experiment };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.QL_REPORT = api;
})(typeof window !== 'undefined' ? window : globalThis);
