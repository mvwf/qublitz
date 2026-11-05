/* dashboard.js: the outreach usage dashboard (web/outreach/dashboard.html). No libraries; charts are hand-drawn SVG.
 *
 * Data: GET <endpoint>?summary=1&days=N, where endpoint comes from ../shared/outreach-config.js or ?endpoint=.
 * The response is Code.gs summarize() (tools/outreach-collector/Code.gs). Question and objective text come from
 * the pages' learning files (../learning/<id>.js), loaded through a QL stub that only records each spec.
 * URL options: ?demo=1 (synthetic data, clearly labelled), ?days=7|30|90|365, ?since=YYYY-MM-DD, ?endpoint=URL.
 * All text from the data or the learning files goes into the page with textContent, never innerHTML.
 */
(function () {
  'use strict';
  const P = new URLSearchParams(location.search);
  const DEMO = P.get('demo') === '1';
  const ENDPOINT = (P.get('endpoint') || (window.QL_CONFIG && window.QL_CONFIG.endpoint) || '').trim();
  const DAY_CHOICES = [7, 30, 90, 365];
  const GAMES = window.QL_GAMES || [];
  const TITLES = {}; GAMES.forEach(g => { TITLES[g.id] = g.title; });
  const IN = window.QL_INSIGHTS;
  const SPECS = {};
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';

  let days = DAY_CHOICES.includes(+P.get('days')) ? +P.get('days') : 30;
  let since = /^\d{4}-\d{2}-\d{2}$/.test(P.get('since') || '') ? P.get('since') : '';
  let summary = null, improvements = [];

  // ── tiny DOM helpers ───────────────────────────────────────────────────────────────────────────────────
  function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const k in attrs || {}) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'text') n.textContent = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(String(c)));
    return n;
  }
  function s(tag, attrs) { const n = document.createElementNS(NS, tag); for (const k in attrs || {}) n.setAttribute(k, attrs[k]); return n; }
  const title = id => TITLES[id] || (SPECS[id] && SPECS[id].title) || id;
  const plain = IN.plain;
  const fmt = n => n == null ? '–' : n >= 10000 ? (Math.round(n / 100) / 10) + 'K' : Math.round(n).toLocaleString('en-US');
  const pct = x => x == null || !isFinite(x) ? '–' : Math.round(x * 100) + '%';
  const fix1 = x => x == null ? '–' : x.toFixed(1);
  const sum = a => a.reduce((x, y) => x + y, 0);
  const byOpens = (a, b) => (summary.games[b] ? summary.games[b].opens : 0) - (summary.games[a] ? summary.games[a].opens : 0);
  function table(head, rows, numericFrom) {
    const nf = numericFrom == null ? 1 : numericFrom;
    return h('table', {}, h('thead', {}, h('tr', {}, head.map((t, i) => h('th', { scope: 'col', class: i >= nf ? 'n' : null, text: t })))),
      h('tbody', {}, rows.map(r => h('tr', {}, r.map((c, i) => i === 0 ? h('th', { scope: 'row', text: c }) : h('td', { class: i >= nf ? 'n' : null, text: c }))))));
  }
  function niceMax(v) {
    if (v <= 4) return 4;
    const p = Math.pow(10, Math.floor(Math.log10(v))), m = v / p;
    return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
  }
  const dayList = (from, to) => {
    const out = [], d = new Date(from + 'T00:00:00Z'), end = new Date(to + 'T00:00:00Z');
    while (d <= end && out.length < 800) { out.push(d.toISOString().slice(0, 10)); d.setUTCDate(d.getUTCDate() + 1); }
    return out;
  };
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const dShort = d => MON[+d.slice(5, 7) - 1] + ' ' + (+d.slice(8, 10));
  const dLong = d => dShort(d) + ', ' + d.slice(0, 4);
  const today = () => new Date().toISOString().slice(0, 10);
  const daysSince = d => Math.max(1, Math.min(730, Math.ceil((Date.now() - Date.parse(d + 'T00:00:00Z')) / 86400000)));

  // ── tooltip (enhances only: every value is also in a label or a table) ──────────────────────────────────
  const tip = $('tip');
  function showTip(x, y, head, rows) {
    tip.replaceChildren(h('div', { class: 'h', text: head }),
      ...rows.map(r => h('div', { class: 'r' }, h('i', { style: r.c ? { '--c': r.c } : null }), h('b', { text: r.v }), h('span', { text: r.l || '' }))));
    tip.style.display = 'block';
    const w = tip.offsetWidth, hh = tip.offsetHeight;
    let left = x + 14, top = y - hh - 10;
    if (left + w > innerWidth - 8) left = x - w - 14;
    if (left < 8) left = 8;
    if (top < 8) top = y + 16;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  const hideTip = () => { tip.style.display = 'none'; };
  function hoverable(el, fn) {   // fn() -> [head, rows]
    const at = (x, y) => { const [hd, rows] = fn(); showTip(x, y, hd, rows); };
    el.addEventListener('pointermove', e => at(e.clientX, e.clientY));
    el.addEventListener('pointerleave', hideTip);
    el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); at(r.left + r.width / 2, r.top); });
    el.addEventListener('blur', hideTip);
  }

  // ── area chart with crosshair (visits per day, and the small multiples) ─────────────────────────────────
  function areaChart(host, xs, ys, opt) {
    const W = Math.max(140, host.clientWidth || 600), small = !!opt.small;
    const H = small ? 54 : (W < 500 ? 170 : 210);
    const pad = small ? { l: 2, r: 4, t: 4, b: 4 } : { l: 38, r: 10, t: 10, b: 24 };
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
    const max = opt.max || niceMax(Math.max(1, ...ys));
    const X = i => pad.l + (xs.length < 2 ? iw / 2 : i / (xs.length - 1) * iw);
    const Y = v => pad.t + ih - v / max * ih;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', tabindex: '0', 'aria-label': opt.label });
    if (!small) {
      for (let k = 0; k <= 4; k++) {
        const v = max * k / 4, y = Y(v);
        svg.append(s('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: k ? 'var(--grid)' : 'var(--axis)', 'stroke-width': 1 }));
        const t = s('text', { x: pad.l - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)', style: 'font-variant-numeric:tabular-nums' });
        t.textContent = fmt(v); svg.append(t);
      }
      const nT = Math.min(xs.length, W < 500 ? 3 : 6);
      for (let k = 0; k < nT; k++) {
        const i = nT < 2 ? 0 : Math.round(k * (xs.length - 1) / (nT - 1));
        const t = s('text', { x: X(i), y: H - 6, 'text-anchor': k === 0 ? 'start' : k === nT - 1 ? 'end' : 'middle', 'font-size': 11, fill: 'var(--muted)' });
        t.textContent = dShort(xs[i]); svg.append(t);
      }
    } else {
      svg.append(s('line', { x1: pad.l, x2: W - pad.r, y1: Y(0), y2: Y(0), stroke: 'var(--axis)', 'stroke-width': 1 }));
    }
    const pts = ys.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
    svg.append(s('path', { d: `M${X(0)},${Y(0)}L${pts.join('L')}L${X(ys.length - 1)},${Y(0)}Z`, fill: 'var(--s1-wash)' }));
    svg.append(s('path', { d: 'M' + pts.join('L'), fill: 'none', stroke: 'var(--s1)', 'stroke-width': small ? 1.5 : 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    // end dot + label on the big chart
    if (!small && ys.length) {
      const li = ys.length - 1;
      svg.append(s('circle', { cx: X(li), cy: Y(ys[li]), r: 4, fill: 'var(--s1)', stroke: 'var(--surface)', 'stroke-width': 2 }));
    }
    const cross = s('line', { y1: pad.t, y2: pad.t + ih, stroke: 'var(--ink-2)', 'stroke-width': 1, visibility: 'hidden' });
    const dot = s('circle', { r: 4, fill: 'var(--s1)', stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' });
    svg.append(cross, dot);
    const hit = s('rect', { x: pad.l, y: 0, width: iw, height: H, fill: 'transparent' });
    svg.append(hit);
    let cur = -1;
    const show = (i, cx, cy) => {
      cur = Math.max(0, Math.min(xs.length - 1, i));
      cross.setAttribute('x1', X(cur)); cross.setAttribute('x2', X(cur)); cross.setAttribute('visibility', 'visible');
      dot.setAttribute('cx', X(cur)); dot.setAttribute('cy', Y(ys[cur])); dot.setAttribute('visibility', 'visible');
      const [hd, rows] = opt.tip(cur);
      if (cx == null) { const r = svg.getBoundingClientRect(); cx = r.left + X(cur) / W * r.width; cy = r.top + Y(ys[cur]) / H * r.height; }
      showTip(cx, cy, hd, rows);
    };
    const hide = () => { cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); hideTip(); };
    svg.addEventListener('pointermove', e => {
      const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * W;
      show(Math.round((x - pad.l) / iw * (xs.length - 1)), e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('blur', hide);
    svg.addEventListener('focus', () => show(cur < 0 ? xs.length - 1 : cur));
    svg.addEventListener('keydown', e => {
      const step = { ArrowLeft: -1, ArrowRight: 1, Home: -1e9, End: 1e9 }[e.key];
      if (step == null) return;
      e.preventDefault(); show((cur < 0 ? xs.length - 1 : cur) + step);
    });
    host.replaceChildren(svg);
  }

  // ── sections ───────────────────────────────────────────────────────────────────────────────────────────
  function renderKpis() {
    const S = summary, G = Object.values(S.games), Q = Object.values(S.quiz);
    const visits = sum(G.map(g => g.opens)), quizzes = sum(Q.map(q => q.n)), ingame = sum(Q.map(q => q.ingame || 0));
    const lab = sum(Object.values(S.labFrom || {}));
    const wavg = key => { let w = 0, t = 0; Q.forEach(q => { if (q[key] != null) { w += q.n; t += q[key] * q.n; } }); return w ? t / w : null; };
    const fb = sum(Object.values(S.feedbackCount || {}));
    const tile = (k, v, d, hero) => h('div', { class: 'tile' + (hero ? ' hero' : '') }, h('div', { class: 'k', text: k }), h('div', { class: 'v' }, v), d ? h('div', { class: 'd', text: d }) : null);
    const score = wavg('avgScore');
    $('kpis').replaceChildren(
      tile('Visits', fmt(visits), `${Object.keys(S.games).filter(g => S.games[g].opens).length} pages, since ${dLong(S.since)}`, true),
      tile('Debriefs sent', fmt(quizzes), quizzes && visits ? `${pct(quizzes / visits)} of visits` : ''),
      tile('Questions right', score == null ? '–' : pct(score), `share of answers, debrief and in-game (${fmt(ingame)} in-game)`),
      tile('Fun', [fix1(wavg('avgFun')), h('small', { text: ' / 5' })], 'average rating'),
      tile('Learned', [fix1(wavg('avgLearned')), h('small', { text: ' / 5' })], 'average rating'),
      tile('Lab arrivals', fmt(lab), 'visits to the Virtual Qubit Lab from an activity'),
      tile('Written responses', fmt(fb), 'count only; the text stays in the Sheet'));
  }

  function renderInsights() {
    const flags = IN.insights(summary, SPECS, TITLES);
    const R = IN.RULES;
    $('flags').replaceChildren(...(flags.length ? flags.slice(0, 12).map(f => h('li', { class: 'flag' },
      h('span', { class: 'imp', text: `impact ${f.impact}` }),
      h('div', { class: 't' }, h('span', { class: 'ic', 'aria-hidden': 'true', text: '▲' }), f.title),
      f.question ? h('div', { class: 'q', text: '“' + f.question + '”' }) : null,
      h('div', { class: 'ev' }, h('b', { text: 'Evidence: ' }), f.evidence),
      h('div', { class: 'fx' }, h('b', { text: 'Try: ' }), f.fix))) :
      [h('li', { class: 'empty', text: 'Nothing crosses a threshold in this period. (Rules only fire once enough people were measured.)' })]));
    if (flags.length > 12) $('flags').append(h('li', { class: 'empty', text: `…and ${flags.length - 12} smaller flags.` }));
    $('rules').replaceChildren(h('ul', {},
      h('li', { text: `Question: under ${pct(R.QUIZ_ACCURACY)} of answers right (in-game and debrief together), with at least ${R.QUIZ_MIN_ANSWERS} answers. Shows the most-picked wrong choice.` }),
      h('li', { text: `Level: under ${pct(R.LEVEL_COMPLETION)} of starts end in a completion, with at least ${R.LEVEL_MIN_STARTS} starts.` }),
      h('li', { text: `Stop hotspot: at least ${pct(R.STOP_RATE)} of the visits that start a level end there without finishing it, and at least ${R.STOP_VS_TYPICAL}× the game's typical (median) rate per level, with at least ${R.STOP_MIN_QUITS} visits ending there. Visits that finish a level and then leave do not count as stops.` }),
      h('li', { text: `Rating: average fun or learned below ${R.RATING} of 5, with at least ${R.RATING_MIN_QUIZZES} debriefs.` }),
      h('li', { text: `Debrief drop-off: under ${pct(R.QUIZ_SUBMIT)} of debrief openings sent, with at least ${R.QUIZ_MIN_OPENS} openings.` }),
      h('li', { text: `Short visits: average active time under ${R.MINUTES} min, with at least ${R.MINUTES_MIN_VISITS} visits (the hub is skipped).` }),
      h('li', { text: 'Impact = people measured × (threshold − value) ÷ threshold (value − threshold for stop hotspots, capped at 1). Thresholds live in web/outreach/insights.js.' })));
  }

  function renderLog() {
    const list = improvements.slice().sort((a, b) => String(b.date).localeCompare(String(a.date)));
    $('log').replaceChildren(...(list.length ? list.map(it => h('li', {},
      h('span', { class: 'dt', text: it.date || '' }),
      h('div', {},
        h('div', { class: 'ch' }, it.demo ? h('span', { class: 'demo-chip', text: 'DEMO' }) : null, title(it.game || '') + ': ' + (it.change || '')),
        h('div', { class: 'bc', text: 'Because: ' + (it.because || '') }),
        /^\d{4}-\d{2}-\d{2}$/.test(it.date || '') ? h('button', { type: 'button', text: 'Show the numbers since ' + dLong(it.date), onclick: () => { setSince(it.date); } }) : null))) :
      [h('li', { class: 'empty', text: 'No changes logged yet. When feedback leads to a change, it is listed here with the evidence behind it.' })]));
  }

  function renderVisits() {
    const S = summary, xs = dayList(S.since, S.generated.slice(0, 10));
    const ids = Object.keys(S.games).filter(g => S.games[g].opens).sort(byOpens);
    const per = {}; ids.forEach(g => { per[g] = xs.map(d => (S.byDay[d] && S.byDay[d][g]) || 0); });
    const tot = xs.map((d, i) => sum(ids.map(g => per[g][i])));
    const peak = tot.indexOf(Math.max(...tot));
    areaChart($('visits'), xs, tot, {
      label: `Visits per day from ${dLong(xs[0])} to ${dLong(xs[xs.length - 1])}: ${fmt(sum(tot))} in total, peak ${fmt(tot[peak] || 0)} on ${dLong(xs[peak] || xs[0])}.`,
      tip: i => {
        const top = ids.map(g => [g, per[g][i]]).filter(x => x[1]).sort((a, b) => b[1] - a[1]);
        const rows = [{ v: fmt(tot[i]), l: 'all pages' }].concat(top.slice(0, 5).map(([g, v]) => ({ v: fmt(v), l: title(g), c: 'transparent' })));
        if (top.length > 5) rows.push({ v: fmt(sum(top.slice(5).map(x => x[1]))), l: `${top.length - 5} more`, c: 'transparent' });
        return [dLong(xs[i]), rows];
      } });
    const mmax = niceMax(Math.max(1, ...ids.map(g => Math.max(...per[g]))));
    const mult = $('multiples'); mult.replaceChildren();
    ids.forEach(g => {
      const box = h('div', { class: 'm' }, h('div', { class: 'mt' }, h('span', { text: title(g) }), h('span', { text: fmt(S.games[g].opens) })), h('div', { class: 'chart' }));
      mult.append(box);
      areaChart(box.lastChild, xs, per[g], { small: true, max: mmax, label: `${title(g)}: ${fmt(S.games[g].opens)} visits`,
        tip: i => [dLong(xs[i]), [{ v: fmt(per[g][i]), l: title(g) }]] });
    });
    if (!ids.length) mult.append(h('div', { class: 'empty', text: 'No visits in this period.' }));
    const rows = xs.slice().reverse().map((d, j) => { const i = xs.length - 1 - j; return [d, fmt(tot[i])].concat(ids.map(g => fmt(per[g][i]))); });
    $('visits-table').replaceChildren(table(['Day', 'All'].concat(ids.map(title)), rows));
  }

  function renderFunnel() {
    const S = summary;
    const ids = Object.keys(S.games).filter(g => S.games[g].opens || S.games[g].starts).sort(byOpens);
    if (!ids.length) { $('funnel').replaceChildren(h('div', { class: 'empty', text: 'No visits in this period.' })); return; }
    const cell = (v, share, sub) => h('td', { class: 'cell' },
      h('div', { class: 'num', text: fmt(v) }), sub ? h('div', { class: 'sub', text: sub }) : null,
      share != null ? h('div', { class: 'barbox', 'aria-hidden': 'true' }, h('div', { class: 'bar', style: { width: Math.min(100, share * 100).toFixed(1) + '%' } })) : null);
    $('funnel').replaceChildren(h('table', { class: 'funnel' },
      h('thead', {}, h('tr', {}, ['Game', 'Visits', 'Started', 'Levels solved', 'Debrief opened', 'Debrief sent', 'Avg time'].map((t, i) => h('th', { scope: 'col', class: i ? 'n' : null, text: t })))),
      h('tbody', {}, ids.map(g => {
        const G = S.games[g], o = G.opens || 0, qn = (S.quiz[g] && S.quiz[g].n) || 0;
        const sh = v => o ? v / o : null;
        return h('tr', {}, h('th', { scope: 'row', class: 'g', text: title(g) }),
          cell(o, null), cell(G.starts, sh(G.starts), o ? pct(sh(G.starts)) : ''),
          cell(G.completes, null, o && G.completes ? (G.completes / o).toFixed(1) + ' per visit' : ''),
          cell(G.quizOpens, sh(G.quizOpens), o ? pct(sh(G.quizOpens)) : ''), cell(qn, sh(qn), o ? pct(sh(qn)) : ''),
          h('td', { class: 'n', text: G.avgMinutes == null ? '–' : G.avgMinutes.toFixed(1) + ' min' }));
      }))));
  }

  function barRows(items, opt) {   // items: {label, frac, val, tag, tip:[head, rows]}
    return h('div', { class: 'rows', role: 'list' }, items.map(it => {
      const r = h('div', { class: 'row', role: 'listitem', tabindex: '0', 'aria-label': it.aria },
        h('span', { class: 'lab', text: it.label, title: it.label }),
        h('div', { class: 'track', 'aria-hidden': 'true' },
          h('div', { class: 'b', style: { width: Math.max(0, Math.min(1, it.frac)) * 100 + '%' } }),
          opt && opt.ref != null ? h('div', { class: 'ref', style: { left: opt.ref * 100 + '%' } }) : null),
        h('span', { class: 'val' }, it.val, it.tag ? [' ', h('span', { class: 'tag', text: it.tag })] : null));
      if (it.tip) hoverable(r, () => it.tip);
      return r;
    }));
  }

  function renderLevels() {
    const S = summary, R = IN.RULES;
    $('lvl-th').textContent = pct(R.LEVEL_COMPLETION);
    const ids = Object.keys(S.games).filter(g => Object.keys(S.games[g].levels || {}).length).sort(byOpens);
    const box = $('levels'); box.replaceChildren();
    if (!ids.length) { box.append(h('div', { class: 'empty', text: 'No level events in this period.' })); return; }
    const trows = [];
    ids.forEach(g => {
      const L = S.games[g].levels, names = Object.keys(L).sort((a, b) => L[b].starts - L[a].starts);
      box.append(h('h3', { text: title(g) }));
      box.append(barRows(names.map(n => {
        const x = L[n], fr = x.starts ? x.completes / x.starts : 0, stars = x.completes ? x.starsSum / x.completes : null;
        const low = x.starts >= R.LEVEL_MIN_STARTS && fr < R.LEVEL_COMPLETION;
        trows.push([title(g) + ' · ' + n, fmt(x.starts), fmt(x.completes), x.starts ? pct(fr) : '–', stars == null ? '–' : stars.toFixed(1)]);
        return { label: n, frac: fr, val: `${x.starts ? pct(fr) : '–'} of ${fmt(x.starts)}`, tag: low ? '▲ low' : '',
          aria: `${n}: ${x.completes} of ${x.starts} starts completed${stars != null ? ', average ' + stars.toFixed(1) + ' stars' : ''}${low ? ', below the flag line' : ''}`,
          tip: [`${title(g)} · ${n}`, [{ v: pct(fr), l: 'completed' }, { v: `${fmt(x.completes)} / ${fmt(x.starts)}`, l: 'completions / starts', c: 'transparent' }]
            .concat(stars != null ? [{ v: stars.toFixed(1), l: 'average stars', c: 'transparent' }] : [])] };
      }), { ref: R.LEVEL_COMPLETION }));
    });
    box.append(h('details', { class: 'tv' }, h('summary', { text: 'Table view' }), h('div', { class: 'tscroll' }, table(['Level', 'Starts', 'Completions', 'Rate', 'Avg stars'], trows))));
  }

  function renderRatings() {
    const S = summary;
    const ids = Object.keys(S.quiz).filter(g => S.quiz[g].avgFun != null || S.quiz[g].avgLearned != null).sort((a, b) => S.quiz[b].n - S.quiz[a].n);
    const host = $('ratings');
    if (!ids.length) { host.replaceChildren(h('div', { class: 'empty', text: 'No ratings in this period.' })); $('ratings-table').replaceChildren(); return; }
    const W = Math.max(260, host.clientWidth || 500), labW = Math.min(150, W * 0.34), valW = 70, rowH = 30;
    const pad = { t: 6, b: 22 }, x0 = labW + 8, x1 = W - valW, H = pad.t + ids.length * rowH + pad.b;
    const X = v => x0 + (v - 1) / 4 * (x1 - x0);
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
      'aria-label': 'Average fun and learned ratings per game: ' + ids.map(g => `${title(g)} fun ${fix1(S.quiz[g].avgFun)}, learned ${fix1(S.quiz[g].avgLearned)}`).join('; ') });
    for (let v = 1; v <= 5; v++) {
      svg.append(s('line', { x1: X(v), x2: X(v), y1: pad.t, y2: H - pad.b, stroke: v === 3 ? 'var(--axis)' : 'var(--grid)', 'stroke-width': 1 }));
      const t = s('text', { x: X(v), y: H - 6, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }); t.textContent = v; svg.append(t);
    }
    ids.forEach((g, i) => {
      const Q = S.quiz[g], cy = pad.t + i * rowH + rowH / 2;
      const grp = s('g', { tabindex: '0', role: 'img', 'aria-label': `${title(g)}: fun ${fix1(Q.avgFun)}, learned ${fix1(Q.avgLearned)}, ${Q.n} debriefs` });
      grp.append(s('rect', { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: 'transparent' }));
      const lt = s('text', { x: 0, y: cy + 4, 'font-size': 12, fill: 'var(--ink-2)' });
      let name = title(g); const maxCh = Math.floor(labW / 6.6); if (name.length > maxCh) name = name.slice(0, maxCh - 1) + '…';
      lt.textContent = name; grp.append(lt);
      if (Q.avgFun != null && Q.avgLearned != null)
        grp.append(s('line', { x1: X(Q.avgFun), x2: X(Q.avgLearned), y1: cy, y2: cy, stroke: 'var(--axis)', 'stroke-width': 2 }));
      if (Q.avgFun != null) grp.append(s('circle', { cx: X(Q.avgFun), cy, r: 5, fill: 'var(--s1)', stroke: 'var(--surface)', 'stroke-width': 2 }));
      if (Q.avgLearned != null) grp.append(s('rect', { x: X(Q.avgLearned) - 5, y: cy - 5, width: 10, height: 10, rx: 1.5, fill: 'var(--s2)', stroke: 'var(--surface)', 'stroke-width': 2 }));
      const vt = s('text', { x: W, y: cy + 4, 'text-anchor': 'end', 'font-size': 11.5, fill: 'var(--ink-2)', style: 'font-variant-numeric:tabular-nums' });
      vt.textContent = `${fix1(Q.avgFun)} · ${fix1(Q.avgLearned)}`; grp.append(vt);
      hoverable(grp, () => [title(g), [{ v: fix1(Q.avgFun), l: 'fun', c: 'var(--s1)' }, { v: fix1(Q.avgLearned), l: 'learned', c: 'var(--s2)' }, { v: fmt(Q.n), l: 'debriefs', c: 'transparent' }]]);
      svg.append(grp);
    });
    host.replaceChildren(svg);
    $('ratings-table').replaceChildren(table(['Game', 'Fun', 'Learned', 'Debriefs'], ids.map(g => [title(g), fix1(S.quiz[g].avgFun), fix1(S.quiz[g].avgLearned), fmt(S.quiz[g].n)])));
  }

  function renderQuiz() {
    const S = summary, box = $('quiz'); box.replaceChildren();
    const ids = Object.keys(S.quiz).filter(g => Object.keys(S.quiz[g].items || {}).length).sort((a, b) => S.quiz[b].n - S.quiz[a].n);
    if (!ids.length) { box.append(h('div', { class: 'empty', text: 'No answers in this period.' })); return; }
    const trows = [];
    ids.forEach((g, gi) => {
      const Q = S.quiz[g], spec = SPECS[g] || {}, items = Q.items;
      const flagged = Object.values(items).some(it => it.n >= IN.RULES.QUIZ_MIN_ANSWERS && it.correct / it.n < IN.RULES.QUIZ_ACCURACY);
      const sec = h('details', { class: 'gq', open: gi === 0 || flagged });
      box.append(sec);
      sec.append(h('summary', { class: 'gamehead' }, h('h3', { text: title(g) }),
        h('span', { text: `${fmt(Q.n)} debriefs · ${fmt(Q.ingame || 0)} in-game answers · ${pct(Q.avgScore)} right overall` + (spec.quiz ? '' : ' · question text not found (no learning file)') })));
      // group by objective, in the learning file's order; unknown objectives last
      const order = (spec.objectives || []).map(o => o.id), groups = {};
      Object.keys(items).forEach(qid => {
        const q = (spec.quiz || []).find(x => x.id === qid);
        const ob = (q && q.objective) || items[qid].objective || '';
        (groups[ob] = groups[ob] || []).push(qid);
      });
      const qOrder = qid => { const i = (spec.quiz || []).findIndex(x => x.id === qid); return i < 0 ? 1e3 : i; };
      Object.keys(groups).sort((a, b) => ((order.indexOf(a) + 1) || 1e3) - ((order.indexOf(b) + 1) || 1e3)).forEach(ob => {
        const o = (spec.objectives || []).find(x => x.id === ob);
        sec.append(h('h4', { text: o ? 'Objective: ' + plain(o.text) : ob ? 'Objective: ' + ob : 'No objective given' }));
        const list = h('div', { class: 'items' });
        groups[ob].sort((a, b) => qOrder(a) - qOrder(b)).forEach(qid => {
          const it = items[qid], q = (spec.quiz || []).find(x => x.id === qid), acc = it.n ? it.correct / it.n : 0;
          const w = IN.topWrong(it, q), low = it.n >= IN.RULES.QUIZ_MIN_ANSWERS && acc < IN.RULES.QUIZ_ACCURACY;
          const right = q && q.choices && typeof q.answer === 'number' ? plain(q.choices[q.answer]) : '';
          const ig = it.ingame || { n: 0, correct: 0 }, dn = it.n - ig.n, dc = it.correct - ig.correct;
          const split = ig.n ? `In-game: ${pct(ig.correct / ig.n)} of ${fmt(ig.n)} · debrief: ${dn ? pct(dc / dn) + ' of ' + fmt(dn) : 'none'}` : '';
          trows.push([`${title(g)} · ${qid}`, q ? plain(q.q) : '', fmt(it.n), pct(acc), ig.n ? pct(ig.correct / ig.n) : '–', dn ? pct(dc / dn) : '–', w ? `${w.text} (${pct(w.share)})` : '–']);
          const row = h('div', { class: 'acc', tabindex: '0', 'aria-label': `${qid}: ${pct(acc)} right of ${it.n} answers` + (w ? `; most-picked wrong answer ${w.text}, ${pct(w.share)}` : '') },
            h('div', { class: 'track', 'aria-hidden': 'true' }, h('div', { class: 'b', style: { width: acc * 100 + '%' } }), h('div', { class: 'ref', style: { left: '50%' } })),
            h('span', { class: 'val' }, `${pct(acc)} of ${fmt(it.n)}`, low ? [' ', h('span', { class: 'tag', text: '▲ below 50%' })] : null));
          hoverable(row, () => [`${title(g)} · ${qid}`, [{ v: pct(acc), l: 'right' }, { v: fmt(it.n), l: 'answers', c: 'transparent' }]
            .concat(w ? [{ v: pct(w.share), l: 'picked “' + w.text + '”', c: 'transparent' }] : [])]);
          list.append(h('div', { class: 'item' },
            h('div', { class: 'qt' }, h('span', { class: 'id', text: qid }), q ? plain(q.q) : 'Question text not in a learning file'),
            row,
            split ? h('div', { class: 'split', text: split }) : null,
            w ? h('div', { class: 'mis' }, 'Most-picked wrong answer: ', h('b', { text: '“' + w.text + '”' }), ` (${pct(w.share)} of answers)` + (right ? ` · right answer: “${right}”` : '')) : null));
        });
        sec.append(list);
      });
    });
    box.append(h('details', { class: 'tv' }, h('summary', { text: 'Table view' }), h('div', { class: 'tscroll' }, table(['Item', 'Question', 'Answers', 'Right', 'In-game right', 'Debrief right', 'Top wrong answer'], trows, 2))));
  }

  function renderCodes() {
    const S = summary, codes = Object.keys(S.codes || {}).sort((a, b) => S.codes[b] - S.codes[a]);
    const box = $('codes');
    if (!codes.length) { box.replaceChildren(h('div', { class: 'empty', text: 'No visits with an event code in this period.' })); return; }
    const max = Math.max(...codes.map(c => S.codes[c])), CQ = S.codeQuiz || {};
    box.replaceChildren(barRows(codes.map(c => ({ label: c, frac: S.codes[c] / max, val: fmt(S.codes[c]),
      aria: `${c}: ${S.codes[c]} visits`, tip: [c, [{ v: fmt(S.codes[c]), l: 'visits' }].concat(CQ[c] ? [{ v: fmt(CQ[c].n), l: 'debriefs', c: 'transparent' }] : [])] }))),
      h('details', { class: 'tv' }, h('summary', { text: 'Table view' }), h('div', { class: 'tscroll' }, table(['Code', 'Visits'], codes.map(c => [c, fmt(S.codes[c])])))));
  }

  function renderCodeGames() {
    const S = summary, CG = S.codeGames, CQ = S.codeQuiz || {}, card = $('codegames-card');
    const codes = Object.keys(S.codes || {}).sort((a, b) => S.codes[b] - S.codes[a]);
    if (!CG || !codes.length) { card.hidden = true; return; }
    card.hidden = false;
    const ids = Array.from(new Set(codes.flatMap(c => Object.keys(CG[c] || {})))).sort(byOpens);
    const max = Math.max(1, ...codes.flatMap(c => ids.map(g => (CG[c] || {})[g] || 0)));
    const steps = ['--q1', '--q2', '--q3', '--q4', '--q5', '--q6'];
    const heat = v => { if (!v) return {}; const k = Math.min(5, Math.floor(v / max * 5.999)); return { background: `var(${steps[k]})`, color: k >= 3 ? '#fff' : '#0b0b0b' }; };
    $('codegames').replaceChildren(h('table', { class: 'heat' },
      h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: 'Code' }), ids.map(g => h('th', { scope: 'col', class: 'n', text: title(g) })),
        ['Debriefs', 'Right', 'Fun', 'Learned'].map(t => h('th', { scope: 'col', class: 'n', text: t })))),
      h('tbody', {}, codes.map(c => {
        const q = CQ[c];
        return h('tr', {}, h('th', { scope: 'row', text: c }),
          ids.map(g => { const v = (CG[c] || {})[g] || 0; return h('td', { class: 'h', style: heat(v), text: v ? fmt(v) : '·', 'aria-label': v ? null : 'none' }); }),
          h('td', { class: 'n', text: q ? fmt(q.n) : '–' }), h('td', { class: 'n', text: q ? pct(q.avgScore) : '–' }),
          h('td', { class: 'n', text: q ? fix1(q.avgFun) : '–' }), h('td', { class: 'n', text: q ? fix1(q.avgLearned) : '–' }));
      }))));
  }

  function renderDevices() {
    const D = summary.devices || {}, parts = [['desktop', 'Desktop', 'var(--s1)'], ['touch', 'Touch screen', 'var(--s2)']].filter(p => D[p[0]]);
    const tot = sum(parts.map(p => D[p[0]])), box = $('devices');
    if (!tot) { box.replaceChildren(h('div', { class: 'empty', text: 'No visits in this period.' })); return; }
    const bar = h('div', { class: 'segbar', role: 'img', 'aria-label': parts.map(p => `${p[1]} ${pct(D[p[0]] / tot)} (${D[p[0]]})`).join(', ') });
    parts.forEach(p => {
      const sh = D[p[0]] / tot, seg = h('div', { style: { flex: `${sh} 1 0`, background: p[2] }, tabindex: '0', 'aria-label': `${p[1]}: ${pct(sh)}` }, sh > 0.12 ? pct(sh) : '');
      hoverable(seg, () => [p[1], [{ v: pct(sh), l: 'of visits', c: p[2] }, { v: fmt(D[p[0]]), l: 'visits', c: 'transparent' }]]);
      bar.append(seg);
    });
    box.replaceChildren(bar,
      h('div', { class: 'legend' }, parts.map(p => h('span', {}, h('i', { class: 'sq', style: { '--c': p[2] } }), `${p[1]} · ${fmt(D[p[0]])}`))));
  }

  function renderStops() {
    const S = summary, box = $('stops');
    const ids = Object.keys(S.games).filter(g => g !== 'hub' && S.games[g].quits && Object.keys(S.games[g].quits).length).sort(byOpens);
    box.replaceChildren();
    if (!ids.length) { box.append(h('div', { class: 'empty', text: 'No visit ends reported in this period (pages report them with QL.context).' })); return; }
    const hot = new Set(IN.insights(S, SPECS, TITLES).filter(f => f.kind === 'stop').map(f => f.game + '\u0000' + f.ref));
    const trows = [];
    ids.forEach((g, gi) => {
      const G = S.games[g], Qt = G.quits, levels = G.levels || {}, total = sum(Object.values(Qt));
      const places = Object.keys(Qt).sort((a, b) => Qt[b] - Qt[a]);
      const shown = places.slice(0, 8), rest = places.slice(8);
      const items = shown.map(p => {
        const L = levels[p.split('|')[0]], share = Qt[p] / total;
        const stops = L ? Math.max(0, Math.min(Qt[p], L.starts - L.completes)) : null;
        trows.push([`${title(g)} · ${p}`, fmt(Qt[p]), pct(share), L ? fmt(L.starts) : '–', L ? fmt(stops) : '–']);
        const flag = hot.has(g + '\u0000' + p);
        return { label: p, frac: Qt[p] / Qt[places[0]], val: pct(share), tag: flag ? '▲ hotspot' : '',
          aria: `${p}: ${Qt[p]} visits ended here, ${pct(share)} of visit ends` + (L ? `; ${L.starts} started this level, ${stops} of the visits ending here had not finished it` : '') + (flag ? '; flagged as a stop hotspot' : ''),
          tip: [`${title(g)} · ${p}`, [{ v: fmt(Qt[p]), l: 'visits ended here' }, { v: pct(share), l: 'of visit ends', c: 'transparent' }]
            .concat(L ? [{ v: fmt(L.starts), l: 'started this level', c: 'transparent' }, { v: fmt(stops), l: 'ended here without finishing', c: 'transparent' }] : [])] };
      });
      if (rest.length) { const r = sum(rest.map(p => Qt[p])); items.push({ label: `${rest.length} other places`, frac: r / Qt[places[0]], val: pct(r / total), aria: `${rest.length} other places: ${r} visits` }); }
      const C = G.completedPerVisit || {}, ks = Object.keys(C).map(Number).sort((a, b) => a - b), ctot = sum(ks.map(k => C[k]));
      const depth = ks.some(k => k > 0) ? barRows(ks.map(k => ({ label: k === 1 ? '1 level' : `${k} levels`, frac: C[k] / Math.max(...ks.map(j => C[j])), val: pct(C[k] / ctot),
        aria: `${k} levels completed: ${C[k]} visits, ${pct(C[k] / ctot)}`, tip: [`${title(g)} · ${k} levels in one visit`, [{ v: fmt(C[k]), l: 'visits' }, { v: pct(C[k] / ctot), l: 'of visits', c: 'transparent' }]] }))) :
        h('div', { class: 'empty', text: 'No levels completed (or this page has no levels).' });
      box.append(h('details', { class: 'gq', open: gi < 3 }, h('summary', { class: 'gamehead' }, h('h3', { text: title(g) }), h('span', { text: `${fmt(total)} visit ends` })),
        h('div', { class: 'stopgame' },
          h('div', {}, h('h4', { text: 'Where visits ended' }), barRows(items)),
          h('div', {}, h('h4', { text: 'Levels completed per visit' }), depth))));
    });
    box.append(h('details', { class: 'tv' }, h('summary', { text: 'Table view' }), h('div', { class: 'tscroll' }, table(['Place', 'Visits ended', 'Share of ends', 'Started the level', 'Ended without finishing'], trows))));
  }

  function renderLab() {
    const S = summary, from = S.labFrom || {}, box = $('lab');
    const ids = Array.from(new Set(Object.keys(from).concat(Object.keys(S.games).filter(g => S.games[g].events && S.games[g].events.to_lab)))).filter(g => g !== 'qubit-lab');
    if (!ids.length) { box.replaceChildren(h('div', { class: 'empty', text: 'No links followed to the Virtual Qubit Lab in this period.' })); return; }
    const clicks = g => (S.games[g] && S.games[g].events && S.games[g].events.to_lab) || 0, opens = g => (S.games[g] && S.games[g].opens) || 0;
    ids.sort((a, b) => (from[b] || 0) - (from[a] || 0) || clicks(b) - clicks(a));
    const max = Math.max(1, ...ids.map(g => Math.max(from[g] || 0, clicks(g))));
    box.replaceChildren(barRows(ids.map(g => ({ label: title(g), frac: (from[g] || 0) / max,
      val: `${fmt(from[g] || 0)} arrived · ${fmt(clicks(g))} clicks`,
      aria: `${title(g)}: ${from[g] || 0} arrivals at the lab, ${clicks(g)} link clicks, ${opens(g)} visits`,
      tip: [title(g), [{ v: fmt(from[g] || 0), l: 'arrived at the lab' }, { v: fmt(clicks(g)), l: 'lab link clicks', c: 'transparent' },
        { v: opens(g) ? pct((from[g] || 0) / opens(g)) : '–', l: 'of its visits', c: 'transparent' }]] }))),
      h('details', { class: 'tv' }, h('summary', { text: 'Table view' }), h('div', { class: 'tscroll' },
        table(['From', 'Visits', 'Lab link clicks', 'Arrived at the lab'], ids.map(g => [title(g), fmt(opens(g)), fmt(clicks(g)), fmt(from[g] || 0)])))));
  }

  function renderAll() {
    if (!summary) return;
    hideTip();
    renderKpis(); renderInsights(); renderLog(); renderVisits(); renderFunnel(); renderLevels(); renderRatings(); renderQuiz();
    renderStops(); renderLab(); renderCodes(); renderDevices(); renderCodeGames();
  }

  // ── states ─────────────────────────────────────────────────────────────────────────────────────────────
  function setBanner(...nodes) { $('banner').replaceChildren(...nodes); }
  function offState() {
    $('main').hidden = true;
    $('days').closest('.filters').hidden = true;
    setBanner(h('section', { class: 'banner', 'aria-labelledby': 'off-h' },
      h('h2', { id: 'off-h', text: 'Collection is off' }),
      h('p', {}, 'No collector endpoint is set in ', h('code', { text: 'web/shared/outreach-config.js' }),
        ', so the pages send nothing and there is nothing to show. To switch it on (about 10 minutes):'),
      h('ol', {},
        h('li', { text: 'Create a Google Sheet (a Dartmouth account is fine), e.g. “Qublitz outreach”.' }),
        h('li', {}, 'Extensions → Apps Script. Replace the contents of Code.gs with ', h('code', { text: 'tools/outreach-collector/Code.gs' }), '. Save.'),
        h('li', { text: 'Deploy → New deployment → Web app. Execute as: Me. Who has access: Anyone. Deploy, authorise, and copy the Web app URL (ends in /exec).' }),
        h('li', {}, 'Paste that URL into ', h('code', { text: 'web/shared/outreach-config.js' }), ' as endpoint, commit, and let Pages deploy.'),
        h('li', { text: 'Reload this page.' })),
      h('p', {}, 'Full instructions: ', h('a', { href: 'https://github.com/mvwf/qublitz/blob/main/tools/outreach-collector/README.md', text: 'tools/outreach-collector/README.md' }),
        '. To see what the dashboard will look like, open the ', h('a', { href: '?demo=1', text: 'demo with synthetic data' }), '.')));
    $('status').textContent = '';
  }
  function errorState(msg) {
    setBanner(h('div', { class: 'banner err', role: 'alert' }, h('h2', { text: 'Could not load the summary' }),
      h('p', {}, msg + ' ', h('button', { type: 'button', text: 'Try again', onclick: load }))));
  }

  function setSince(d) {
    since = d; syncUrl(); load();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function syncUrl() {
    const q = new URLSearchParams(location.search);
    q.set('days', days); if (since) q.set('since', since); else q.delete('since');
    history.replaceState(null, '', '?' + q.toString());
    document.querySelectorAll('#days input').forEach(i => { i.checked = !since && +i.value === days; });
    const pill = $('since-pill');
    if (since) {
      pill.hidden = false;
      pill.replaceChildren('Since ' + dLong(since), h('button', { type: 'button', 'aria-label': 'Clear the since date', text: '✕', onclick: () => { since = ''; syncUrl(); load(); } }));
    } else pill.hidden = true;
  }

  async function load() {
    const n = since ? daysSince(since) : days;
    const main = $('main');
    $('status').textContent = 'Loading…';
    main.classList.add('loading');
    try {
      if (DEMO) summary = demoSummary(n);
      else {
        const u = new URL(ENDPOINT, location.href);
        u.searchParams.set('summary', '1'); u.searchParams.set('days', String(n));
        const r = await fetch(u.toString(), { redirect: 'follow', cache: 'no-store' });
        if (!r.ok) throw new Error('The collector answered ' + r.status + '.');
        const j = await r.json();
        if (!j || typeof j !== 'object' || !j.games) throw new Error('The collector did not return a summary.');
        summary = Object.assign({ byDay: {}, codes: {}, devices: {}, quiz: {}, feedbackCount: {}, labFrom: {} }, j);
        if (!summary.generated) summary.generated = new Date().toISOString();
      }
      if (since) summary.since = since > summary.since ? since : summary.since;
      main.hidden = false;
      if (!DEMO) setBanner();
      renderAll();
      $('status').textContent = `${dLong(summary.since)} – ${dLong(summary.generated.slice(0, 10))}` + (DEMO ? ' · synthetic' : '');
    } catch (e) {
      $('status').textContent = '';
      errorState(String(e && e.message || e));
    } finally { main.classList.remove('loading'); }
  }

  // ── learning files, through a stub QL ──────────────────────────────────────────────────────────────────
  function loadSpecs() {
    let last = null;   // QL.addItems adds to the spec registered last, as in outreach.js
    window.QL = { learning(spec) { if (spec && typeof spec.id === 'string') { SPECS[spec.id] = spec; last = spec; } },
      addItems(list) { if (!last) return; last.quiz = last.quiz || []; (list || []).forEach(it => { if (it && it.id && !last.quiz.some(q => q.id === it.id)) last.quiz.push(it); }); },
      event() {}, context() {}, answer() { return null; }, labLink: () => '', toLab() {}, panel() {}, showQuiz() {}, offerQuiz() {},
      enabled: () => false, optOut() {}, code: () => '' };
    return Promise.all(GAMES.map(g => new Promise(res => {
      const sc = document.createElement('script');
      const t = setTimeout(res, 5000);
      sc.src = '../learning/' + encodeURIComponent(g.id) + '.js';
      sc.onload = sc.onerror = () => { clearTimeout(t); res(); };
      document.head.append(sc);
    })));
  }
  async function loadImprovements() {
    if (DEMO) { improvements = demoImprovements(); return; }
    try {
      const r = await fetch('improvements.json', { cache: 'no-store' });
      const j = r.ok ? await r.json() : [];
      improvements = Array.isArray(j) ? j.filter(x => x && typeof x === 'object') : [];
    } catch (e) { improvements = []; }
  }

  // ── demo data: synthetic, deterministic, clearly labelled ──────────────────────────────────────────────
  function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function demoSummary(n) {
    const rand = rng(53), now = new Date(), end = now.toISOString().slice(0, 10);
    const start = new Date(now.getTime() - n * 86400000).toISOString().slice(0, 10);
    const pois = m => { let L = Math.exp(-m), k = 0, p = 1; do { k++; p *= rand(); } while (p > L && k < 500); return k - 1; };
    const W = { hub: 2.0, 'qubit-lab': 1.3, copter: 1.2, beamline: 1.1, rider: 1.0, arena: 0.8, sonify: 0.6, trio: 0.45, string: 0.45, spectrogram: 0.4, drum: 0.35, fridge: 0.3, 'ep-tpd': 0.25 };
    const LV = { beamline: ['first-light', 'half-turn', 'phase', 'order', 'crosstalk', 'finale'], copter: ['hover', 'tilt', 'xyz', 'precise'], rider: ['1-1', '1-2', '1-3', '2-1', '2-2'] };
    const HARD = { 'beamline:crosstalk': 0.3, 'rider:2-2': 0.35 };   // completion rate of the hard levels
    const EVENTS = [{ code: 'SCIFAIR26', ago: 12, size: 70 }, { code: 'ENGS53-W26', ago: 40, size: 45 }, { code: 'OPENHOUSE', ago: 75, size: 55 }, { code: 'HSVISIT', ago: 200, size: 35 }];
    const out = { generated: now.toISOString(), days: n, since: start, demo: true, games: {}, byDay: {}, codes: {}, devices: {}, codeGames: {}, codeQuiz: {}, quiz: {}, feedbackCount: {}, labFrom: {} };
    const G = id => out.games[id] || (out.games[id] = { opens: 0, starts: 0, completes: 0, wins: 0, losses: 0, quizOpens: 0, leaves: 0, durSum: 0, levels: {}, events: {}, quits: {}, completedPerVisit: {} });
    const STEPS = { 'qubit-lab': ['rabi', 'ramsey', 't1', 'echo'], arena: ['training', 'duel', 'missions'], sonify: ['upload', 'play'], trio: ['heat', 'wave', 'schrodinger'] };
    const STOP_AT = { 'beamline:crosstalk': '|third-mirror', 'rider:2-2': '|boss' };
    const quizzable = id => id !== 'hub' && (LV[id] || (SPECS[id] && SPECS[id].quiz && SPECS[id].quiz.length) || ['qubit-lab', 'arena', 'trio'].includes(id));
    // question items only where a learning file has them (the rest still get ratings), so no question is made up
    const items = id => (SPECS[id] && SPECS[id].quiz && SPECS[id].quiz.length) ? SPECS[id].quiz.map(q => ({ id: q.id, obj: q.objective, k: (q.choices || []).length || 4, ans: q.answer })) : [];
    const diff = {};   // per item: chance of a right answer and the favourite wrong pick
    const rate = { copter: [4.4, 3.6], beamline: [4.1, 4.0], rider: [4.3, 3.3], 'qubit-lab': [3.4, 4.2], arena: [4.5, 2.7], trio: [3.0, 3.9] };
    for (const d of dayList(start, end)) {
      const ago = Math.round((Date.parse(end) - Date.parse(d)) / 86400000), wd = new Date(d + 'T00:00:00Z').getUTCDay();
      const base = (wd === 0 || wd === 6 ? 4 : 9) * (1 + 0.4 * Math.sin(ago / 9));
      const ev = EVENTS.find(e => e.ago === ago);
      for (const id in W) {
        const k = pois(base * W[id] / 2) + (ev ? pois(ev.size * W[id] / 6) : 0);
        for (let v = 0; v < k; v++) {
          const g = G(id), code = ev && rand() < 0.85 ? ev.code : '';
          g.opens++; g.events.open = (g.events.open || 0) + 1;
          (out.byDay[d] = out.byDay[d] || {})[id] = (out.byDay[d][id] || 0) + 1;
          if (code) { out.codes[code] = (out.codes[code] || 0) + 1; const cg = out.codeGames[code] = out.codeGames[code] || {}; cg[id] = (cg[id] || 0) + 1; }
          const dev = rand() < (code ? 0.62 : 0.38) ? 'touch' : 'desktop'; out.devices[dev] = (out.devices[dev] || 0) + 1;
          let minutes = id === 'hub' ? 0.3 + rand() * 0.6 : id === 'sonify' ? 0.3 + rand() * 1.1 : 1 + rand() * 9;
          let place = 'menu', done = 0, ingameAsked = false;
          if (id !== 'hub' && rand() < 0.82) {
            g.starts++;
            if (STEPS[id]) place = STEPS[id][Math.min(STEPS[id].length - 1, Math.floor(rand() * rand() * STEPS[id].length * 1.6))];
            for (const lv of LV[id] || []) {
              const L = g.levels[lv] = g.levels[lv] || { starts: 0, completes: 0, starsSum: 0 };
              L.starts++; place = lv;
              if (rand() > (HARD[id + ':' + lv] != null ? HARD[id + ':' + lv] : 0.85)) { if (STOP_AT[id + ':' + lv] && rand() < 0.7) place += STOP_AT[id + ':' + lv]; break; }
              L.completes++; g.completes++; done++; L.starsSum += 1 + Math.floor(rand() * 3);
              if (lv === 'half-turn' || lv === 'tilt' || lv === '1-2') ingameAsked = true;
              if (rand() < 0.25) break;
            }
          }
          if (ingameAsked && items(id).length) {   // one question asked inside the game
            const q = items(id)[0], Q = out.quiz[id] = out.quiz[id] || { n: 0, ingame: 0, sc: 0, it: 0, f: [0, 0], l: [0, 0], items: {} };
            const I = Q.items[q.id] = Q.items[q.id] || { objective: q.obj, n: 0, correct: 0, picks: {}, wrong: {}, ingame: { n: 0, correct: 0 } };
            const ok = rand() < 0.74, pick = ok ? q.ans : (q.ans + 1) % q.k;
            Q.ingame++; Q.sc += ok ? 1 : 0; Q.it++; I.n++; I.ingame.n++; I.picks[pick] = (I.picks[pick] || 0) + 1;
            if (ok) { I.correct++; I.ingame.correct++; } else I.wrong[pick] = (I.wrong[pick] || 0) + 1;
          }
          if (id !== 'qubit-lab' && id !== 'hub' && rand() < (id === 'copter' || id === 'rider' ? 0.16 : 0.07)) {
            g.events.to_lab = (g.events.to_lab || 0) + 1;
            if (rand() < 0.72) out.labFrom[id] = (out.labFrom[id] || 0) + 1;
          }
          if (quizzable(id) && rand() < (id === 'arena' ? 0.3 : 0.24)) {
            g.quizOpens++;
            if (rand() < (id === 'qubit-lab' ? 0.4 : 0.78)) {
              const Q = out.quiz[id] = out.quiz[id] || { n: 0, ingame: 0, sc: 0, it: 0, f: [0, 0], l: [0, 0], items: {} };
              const CQ = code ? (out.codeQuiz[code] = out.codeQuiz[code] || { n: 0, ingame: 0, sc: 0, it: 0, f: [0, 0], l: [0, 0] }) : null;
              let right = 0; const its = items(id).slice(ingameAsked ? 1 : 0);
              its.forEach(q => {
                const key = id + ':' + q.id;
                if (!diff[key]) diff[key] = { p: (items(id).findIndex(x => x.id === q.id) === 1 && LV[id]) || (id === 'arena' && q === items(id)[0]) ? 0.34 : 0.55 + rand() * 0.35, fav: (q.ans + 1 + Math.floor(rand() * (q.k - 1))) % q.k };
                const I = Q.items[q.id] = Q.items[q.id] || { objective: q.obj, n: 0, correct: 0, picks: {}, wrong: {}, ingame: { n: 0, correct: 0 } };
                const ok = rand() < diff[key].p;
                let pick = q.ans;
                if (!ok) { pick = rand() < 0.65 ? diff[key].fav : (q.ans + 1 + Math.floor(rand() * (q.k - 1))) % q.k; if (pick === q.ans) pick = diff[key].fav; I.wrong[pick] = (I.wrong[pick] || 0) + 1; }
                I.n++; if (ok) { I.correct++; right++; } I.picks[pick] = (I.picks[pick] || 0) + 1;
              });
              const rr = rate[id] || [3.8, 3.7];
              const fun = Math.max(1, Math.min(5, Math.round(rr[0] + (rand() - 0.5) * 2.4))), lea = Math.max(1, Math.min(5, Math.round(rr[1] + (rand() - 0.5) * 2.4)));
              for (const A of [Q, CQ]) if (A) { A.n++; A.sc += right; A.it += its.length; A.f[0] += fun; A.f[1]++; A.l[0] += lea; A.l[1]++; }
              if (rand() < 0.35) out.feedbackCount[id] = (out.feedbackCount[id] || 0) + 1;
              minutes += 1.2;
            }
          }
          g.quits[place] = (g.quits[place] || 0) + 1; g.completedPerVisit[done] = (g.completedPerVisit[done] || 0) + 1;
          if (rand() < 0.9) { g.leaves++; g.durSum += minutes * 60; }
        }
      }
    }
    for (const id in out.games) { const g = out.games[id]; g.avgMinutes = g.leaves ? Math.round(g.durSum / g.leaves / 6) / 10 : null; delete g.durSum; }
    const fin = A => { A.avgScore = A.it ? Math.round(A.sc / A.it * 1000) / 1000 : null; A.avgFun = A.f[1] ? Math.round(A.f[0] / A.f[1] * 100) / 100 : null;
      A.avgLearned = A.l[1] ? Math.round(A.l[0] / A.l[1] * 100) / 100 : null; delete A.sc; delete A.it; delete A.f; delete A.l; };
    Object.values(out.quiz).forEach(fin); Object.values(out.codeQuiz).forEach(fin);
    return out;
  }
  function demoImprovements() {
    const d = k => new Date(Date.now() - k * 86400000).toISOString().slice(0, 10);
    return [
      { date: d(20), game: 'beamline', change: 'Added a hint to the crosstalk level and moved it after “order”.', because: 'level crosstalk: 31% of starts completed', demo: true },
      { date: d(55), game: 'arena', change: 'Shortened the debrief from six questions to four.', because: 'quiz drop-off: 38% of openings submitted', demo: true },
    ];
  }

  // ── start ──────────────────────────────────────────────────────────────────────────────────────────────
  if (DEMO) {
    document.body.classList.add('is-demo');
    document.title = 'DEMO DATA · ' + document.title;
    setBanner(h('div', { class: 'banner demo', role: 'note' }, h('strong', { class: 'tag', text: 'DEMO DATA' }),
      'Everything below is synthetic, generated in your browser to show what the dashboard looks like. It is not real usage. ',
      h('a', { href: location.pathname, text: 'Leave the demo' })));
  }
  document.querySelectorAll('#days input').forEach(i => i.addEventListener('change', () => { days = +i.value; since = ''; syncUrl(); load(); }));
  syncUrl();
  if (!DEMO && !ENDPOINT) { offState(); return; }
  Promise.all([loadSpecs(), loadImprovements()]).then(load);
  let rt = 0, lastW = 0;
  new ResizeObserver(ents => {
    const w = Math.round(ents[0].contentRect.width);
    if (w === lastW) return; lastW = w;
    clearTimeout(rt); rt = setTimeout(renderAll, 120);
  }).observe($('main'));
})();
