/* insights.js: the dashboard's "what to fix next" flags, computed from the collector's summary (Code.gs summarize()).
 * Pure: no DOM. The dashboard loads it as a script (window.QL_INSIGHTS); web/shared/tests/outreach.test.js requires it.
 *
 * Each rule compares one measured value with a threshold, and only fires once enough people have been measured
 * (the minimum counts below), so one bad afternoon does not raise a flag. Flags are ranked by impact:
 *     impact = n × gap,   gap = (threshold − value) / threshold   (how far below the bar, as a fraction of it)
 * where n is the number of people the rule measured (answers, level attempts, debriefs, debrief openings, visits).
 * Dividing by the threshold puts every rule on the same 0–1 scale, so "40 answers at 30% correct" and
 * "40 debrief openings with 30% sent" rank alike. Rules that fire ABOVE a threshold (stop hotspots) use
 * gap = (value − threshold) / threshold, capped at 1.
 */
(function (root) {
  'use strict';

  const RULES = {
    // A question is a possible misconception when fewer than half of its answers are right (chance on a
    // four-choice item is 25%, so below 50% means many people hold a definite wrong idea), given enough answers.
    QUIZ_ACCURACY: 0.5, QUIZ_MIN_ANSWERS: 10,
    // A level is too hard (or unclear) when under 40% of the people who start it finish it.
    LEVEL_COMPLETION: 0.4, LEVEL_MIN_STARTS: 10,
    // Ratings are 1–5; 3 is "neutral", so an average below 3 means most raters did not enjoy it / did not learn.
    RATING: 3, RATING_MIN_QUIZZES: 5,
    // People open the debrief but do not send it: fewer than half of the openings end in a submission.
    QUIZ_SUBMIT: 0.5, QUIZ_MIN_OPENS: 10,
    // Average active time on the page below one minute: people leave before they get going.
    MINUTES: 1, MINUTES_MIN_VISITS: 10,
    // A stop hotspot: of the visits that reach a level (level_start), at least half end there without finishing it,
    // and that is at least 1.5× the game's typical (median) rate per level, given at least 10 visits ending there.
    // A visit "ends there" when the page's last QL.context place is in that level. Visits that finished the level
    // and then left (the natural end of the last level) are not stops, so a place's stops are capped at the
    // level's starts − completions. "menu" and "unknown" are left to the short-visit rule.
    STOP_RATE: 0.5, STOP_VS_TYPICAL: 1.5, STOP_MIN_QUITS: 10, STOP_SKIP: ['menu', 'unknown'],
    // Pages that are not activities (time on the hub is short by design).
    SKIP_TIME: ['hub'],
  };

  const pct = x => Math.round(x * 100) + '%';
  const gap = (value, threshold) => Math.max(0, (threshold - value) / threshold);

  // Plain text of a learning-file string (they may carry <b>, <i>, <sub>): tags dropped, common entities decoded.
  const ENT = { amp: '&', lt: '<', gt: '>', quot: '"', '#39': "'", nbsp: ' ', minus: '−', times: '×', pi: 'π', theta: 'θ', phi: 'φ' };
  const plain = s => String(s == null ? '' : s).replace(/<[^>]*>/g, '').replace(/&(#?\w+);/g, (m, k) => ENT[k] != null ? ENT[k] : m)
    .replace(/\s+/g, ' ').trim();

  /* The most-picked wrong choice of a summary question. Uses the collector's `wrong` tally when present,
   * otherwise `picks` minus the right answer from the learning file. Returns { pick, count, share, text } or null. */
  function topWrong(item, specQ) {
    if (!item) return null;
    let tally = item.wrong;
    if (!tally) {
      if (!specQ || typeof specQ.answer !== 'number') return null;
      tally = {};
      Object.keys(item.picks || {}).forEach(k => { if (Number(k) !== specQ.answer && k !== '-1' && k !== '') tally[k] = item.picks[k]; });
    }
    let best = null;
    Object.keys(tally).forEach(k => {
      if (k === '-1' || k === '') return;                        // unanswered
      if (!best || tally[k] > best.count) best = { pick: Number(k), count: tally[k] };
    });
    if (!best) return null;
    best.share = item.n ? best.count / item.n : 0;
    const choice = specQ && specQ.choices && specQ.choices[best.pick];
    best.text = choice != null ? plain(choice) : 'choice ' + (best.pick + 1);
    return best;
  }

  /* summary: Code.gs summarize() output. specs: { gameId: QL.learning spec }. titles: { gameId: title }.
   * Returns flags, highest impact first: { kind, game, ref, title, evidence, fix, n, value, threshold, impact }. */
  function insights(summary, specs, titles, rules) {
    const R = Object.assign({}, RULES, rules || {});
    const S = summary || {}, sp = specs || {}, tt = titles || {};
    const name = id => tt[id] || (sp[id] && sp[id].title) || id;
    const flags = [];
    const add = f => { f.impact = Math.round(f.n * gap(f.value, f.threshold) * 100) / 100; flags.push(f); };

    // 1. questions most people get wrong
    Object.keys(S.quiz || {}).forEach(game => {
      const spec = sp[game] || {}, items = (S.quiz[game] && S.quiz[game].items) || {};
      Object.keys(items).forEach(qid => {
        const it = items[qid];
        if (!it || it.n < R.QUIZ_MIN_ANSWERS) return;
        const acc = it.correct / it.n;
        if (acc >= R.QUIZ_ACCURACY) return;
        const q = (spec.quiz || []).find(x => x.id === qid);
        const objId = (q && q.objective) || it.objective || '';
        const obj = (spec.objectives || []).find(o => o.id === objId);
        const w = topWrong(it, q);
        add({ kind: 'quiz', game, ref: qid, objective: objId,
          title: `${name(game)}: question ${qid} — ${pct(acc)} correct`,
          question: q ? plain(q.q) : '',
          evidence: `question ${qid}: ${pct(acc)} correct of ${it.n} answers` +
            (w ? `; possible misconception: “${w.text}” (${pct(w.share)} picked it)` : ''),
          fix: `Revise the lesson or level that teaches this objective${obj ? ` (“${plain(obj.text)}”)` : objId ? ` (${objId})` : ''}, ` +
            'or the question itself if the wrong choice is a fair reading.',
          n: it.n, value: acc, threshold: R.QUIZ_ACCURACY });
      });
    });

    // 2. levels people start but rarely finish
    Object.keys(S.games || {}).forEach(game => {
      const levels = S.games[game].levels || {};
      Object.keys(levels).forEach(lv => {
        const L = levels[lv];
        if (!L || L.starts < R.LEVEL_MIN_STARTS) return;
        const ratio = L.completes / L.starts;
        if (ratio >= R.LEVEL_COMPLETION) return;
        add({ kind: 'level', game, ref: lv,
          title: `${name(game)}: level “${lv}” — ${pct(ratio)} finish`,
          evidence: `level ${lv}: ${L.completes} completions of ${L.starts} starts (${pct(ratio)})`,
          fix: 'Add a hint, loosen the par, or move the level later / lower its difficulty.',
          n: L.starts, value: ratio, threshold: R.LEVEL_COMPLETION });
      });
    });

    // 3. low fun or learned ratings
    Object.keys(S.quiz || {}).forEach(game => {
      const Q = S.quiz[game];
      if (!Q || Q.n < R.RATING_MIN_QUIZZES) return;
      [['avgFun', 'fun', 'Engagement: shorten the start, add a goal or feedback moment, raise the stakes.'],
       ['avgLearned', 'learned', 'Clarity: make the idea explicit in the brief, the feedback and the explanation after each level.']]
        .forEach(([key, label, fix]) => {
          const v = Q[key];
          if (v == null || v >= R.RATING) return;
          add({ kind: 'rating', game, ref: label,
            title: `${name(game)}: “${label}” rated ${v.toFixed(1)} / 5`,
            evidence: `${label} rating: average ${v.toFixed(2)} from ${Q.n} debriefs`,
            fix, n: Q.n, value: v, threshold: R.RATING });
        });
    });

    // 4. debrief opened but not sent
    Object.keys(S.games || {}).forEach(game => {
      const G = S.games[game], opened = G.quizOpens || 0;
      if (opened < R.QUIZ_MIN_OPENS) return;
      const done = (S.quiz && S.quiz[game] && S.quiz[game].n) || 0;
      const ratio = Math.min(1, done / opened);
      if (ratio >= R.QUIZ_SUBMIT) return;
      add({ kind: 'dropoff', game, ref: 'debrief',
        title: `${name(game)}: ${pct(ratio)} of debrief openings sent`,
        evidence: `debrief drop-off: ${done} sent of ${opened} opened`,
        fix: 'Shorten the debrief (fewer or shorter questions, or ask more of them in-game), and make the ratings step quick.',
        n: opened, value: ratio, threshold: R.QUIZ_SUBMIT });
    });

    // 5. very short visits
    Object.keys(S.games || {}).forEach(game => {
      if (R.SKIP_TIME.indexOf(game) >= 0) return;
      const G = S.games[game];
      if (G.avgMinutes == null || (G.leaves || 0) < R.MINUTES_MIN_VISITS || G.avgMinutes >= R.MINUTES) return;
      add({ kind: 'time', game, ref: 'time',
        title: `${name(game)}: average visit ${G.avgMinutes.toFixed(1)} min`,
        evidence: `time on page: ${G.avgMinutes.toFixed(1)} min average over ${G.leaves} visits`,
        fix: 'Onboarding: a first action within seconds, a one-line goal, and a quick early win.',
        n: G.leaves, value: G.avgMinutes, threshold: R.MINUTES });
    });

    // 6. places where an unusual share of the visits that reach them end
    Object.keys(S.games || {}).forEach(game => {
      const G = S.games[game], quits = G.quits || {}, levels = G.levels || {};
      const reachOf = place => { const L = levels[String(place).split('|')[0]]; return L ? L.starts : 0; };
      const unfinished = (n, L) => Math.max(0, Math.min(n, L.starts - L.completes));
      const rates = Object.keys(levels).filter(l => levels[l].starts > 0)
        .map(l => unfinished(sumQuits(quits, l), levels[l]) / levels[l].starts).sort((a, b) => a - b);
      if (!rates.length) return;
      const typical = rates[Math.floor((rates.length - 1) / 2)];      // median stop rate per level
      Object.keys(quits).forEach(place => {
        if (R.STOP_SKIP.indexOf(place) >= 0 || quits[place] < R.STOP_MIN_QUITS) return;
        const lv = String(place).split('|')[0], reach = reachOf(place);
        if (!reach) return;
        const L = levels[lv], stops = unfinished(quits[place], L), rate = Math.min(1, stops / reach);
        if (rate < R.STOP_RATE || rate < R.STOP_VS_TYPICAL * typical) return;
        const f = { kind: 'stop', game, ref: place,
          title: `${name(game)}: ${pct(rate)} of visits that reach “${place}” stop there`,
          evidence: `stop hotspot ${place}: ${quits[place]} visits ended there of ${reach} that started ${lv}, ` +
            `${stops} of them without finishing it (typical for this game: ${pct(typical)}); ${L.completes} of ${L.starts} finished ${lv}`,
          fix: 'A hint, an easier step before it, or a checkpoint so people can come back to it.',
          n: stops, value: rate, threshold: R.STOP_RATE };
        f.impact = Math.round(f.n * Math.min(1, (rate - R.STOP_RATE) / R.STOP_RATE) * 100) / 100;
        flags.push(f);
      });
    });

    return flags.sort((a, b) => b.impact - a.impact || a.title.localeCompare(b.title));
  }

  // visits ending anywhere in a level: the level itself plus its "level|step" places
  function sumQuits(quits, level) {
    let n = 0; Object.keys(quits).forEach(p => { if (p === level || p.indexOf(level + '|') === 0) n += quits[p]; }); return n;
  }

  const api = { RULES, insights, topWrong, plain };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.QL_INSIGHTS = api;
})(typeof window !== 'undefined' ? window : globalThis);
