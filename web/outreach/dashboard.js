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
    const tile = (k, v, d, hero) => h('div', { class: 'tile' + (hero ? ' hero' : '') }, h('div', { class: 'k', text: k }), h('div', { class: 'v' }, v), d ? h('div', { class: 'd', text: d }) : null);
    const score = wavg('avgScore');
    $('kpis').replaceChildren(
      tile('Visits', fmt(visits), `${Object.keys(S.games).filter(g => S.games[g].opens).length} pages, since ${dLong(S.since)}`, true),
      tile('Debriefs sent', fmt(quizzes), quizzes && visits ? `${pct(quizzes / visits)} of visits` : ''),
      tile('Questions right', score == null ? '–' : pct(score), `share of answers, debrief and in-game (${fmt(ingame)} in-game)`),
      tile('Fun', [fix1(wavg('avgFun')), h('small', { text: ' / 5' })], 'average rating'),
      tile('Learned', [fix1(wavg('avgLearned')), h('small', { text: ' / 5' })], 'average rating'),
      tile('Lab arrivals', fmt(lab), 'visits to the Virtual Qubit Lab from an activity'));
  }

  function renderInsights() {
    const flags = IN.insights(summary, SPECS, TITLES);
    const R = IN.RULES;
    $('flags').replaceChildren(...(flags.length ? flags.slice(0, 12).map(f => h('li', { class: 'flag' },
      h('span', { class: 'imp', text: `impact ${f.impact}` }),
      h('div', { class: 't' }, h('span', { class: 'ic', 'aria-hidden': 'true', text: '▲' }), f.title),
      f.question ? h('div', { class: 'q', text: '“' + f.question + '”' }) : null,
      h('div', { class: 'ev' }, h('b', { text: 'Evidence: ' }), f.evidence),
      h('div', { class: 'fx' }, h('b', { text: 'Try: ' }), f.fix),
      h('button', { type: 'button', class: 'brief-link', text: `Redesign brief for ${title(f.game)}`, onclick: () => openBrief(f.game) }))) :
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
        /^\d{4}-\d{2}-\d{2}$/.test(it.date || '') ? h('div', {}, h('button', { type: 'button', text: 'Show the numbers since ' + dLong(it.date), onclick: () => { setSince(it.date); } }),
          ' · ', h('button', { type: 'button', text: 'Compare before / after', onclick: () => openCompare(improvements.indexOf(it)) })) : null))) :
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
    return h('div', { class: 'rows' + (opt && opt.wrap ? ' wrap' : ''), role: 'list' }, items.map(it => {
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
    const heat = v => { if (!v) return {}; const k = Math.min(5, Math.floor(v / max * 5.999)); const dk = document.documentElement.classList.contains('retro');   // the retro ramp gets lighter with value
      return { background: `var(${steps[k]})`, color: dk ? (k >= 2 ? '#061016' : '#eaf0ff') : (k >= 3 ? '#fff' : '#0b0b0b') }; };
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

  function renderVersions() {
    const S = summary, box = $('versions'), U = S.unversioned;
    const devOn = /[?&]dev=1/.test(location.search), tog = new URL(location.href);
    if (devOn) tog.searchParams.delete('dev'); else tog.searchParams.set('dev', '1');
    $('unver').replaceChildren(U ? `${fmt(U.opens)} such visit${U.opens === 1 ? '' : 's'} (${fmt(U.events)} events, ${fmt(U.quiz)} quiz records) in this period${U.included ? ', included in every card here' : ''}. ` : 'this collector does not report them yet. ',
      U && (U.opens || U.events) ? h('a', { href: tog.toString(), text: devOn ? 'Hide them' : 'Include them' }) : '');
    const rows = [];
    Object.keys(S.games).sort(byOpens).forEach(g => {
      const V = S.games[g].versions || {};
      Object.keys(V).sort().reverse().forEach((k, i) => rows.push([g, k, V[k], i === 0]));
    });
    if (!rows.length) { box.replaceChildren(h('div', { class: 'empty', text: 'No version data yet (it needs the updated collector, Code.gs).' })); return; }
    const link = k => { const m = /\+([0-9a-f]{7,12})$/.exec(k); return m ? h('a', { href: `https://github.com/mvwf/qublitz/commit/${m[1]}`, target: '_blank', rel: 'noopener', text: k }) : document.createTextNode(k); };
    box.replaceChildren(h('table', {},
      h('thead', {}, h('tr', {}, ['Game', 'Version', 'Visits', 'Avg minutes', 'Levels done', 'Debriefs', 'In-game answers', 'Answers right'].map(t => h('th', { scope: 'col', text: t })))),
      h('tbody', {}, rows.map(([g, k, v, latest]) => h('tr', {},
        h('th', { scope: 'row', text: TITLES[g] || g }), h('td', {}, link(k), latest ? ' · latest' : ''),
        h('td', { class: 'n', text: fmt(v.opens) }), h('td', { class: 'n', text: fix1(v.avgMinutes) }), h('td', { class: 'n', text: fmt(v.completes) }),
        h('td', { class: 'n', text: fmt(v.debriefs) }), h('td', { class: 'n', text: fmt(v.ingame) }),
        h('td', { class: 'n', text: v.answered ? `${pct(v.correct / v.answered)} of ${fmt(v.answered)}` : '–' }))))));
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
    renderStops(); renderLab(); renderVersions(); renderCodes(); renderDevices(); renderCodeGames();
  }

  // ── fetching a summary (dashboard period, report range, before/after windows) ─────────────────────────────
  async function fetchSummary(o) {   // o: { days, from?, to?, code? }
    if (DEMO) return demoSummary(o);
    const u = new URL(ENDPOINT, location.href);
    u.searchParams.set('summary', '1'); u.searchParams.set('days', String(o.days || 90));
    if (o.from) u.searchParams.set('from', o.from);
    if (o.to) u.searchParams.set('to', o.to);
    if (o.code) u.searchParams.set('code', o.code);
    if (/[?&]dev=1/.test(location.search)) u.searchParams.set('dev', '1');   // include rows with no page version
    const r = await fetch(u.toString(), { redirect: 'follow', cache: 'no-store' });
    if (!r.ok) throw new Error('The collector answered ' + r.status + '.');
    const j = await r.json();
    if (!j || typeof j !== 'object' || !j.games) throw new Error('The collector did not return a summary.');
    if ((o.from && j.since !== o.from) || (o.code && j.code !== o.code))
      throw new Error('This collector deployment is older than the dashboard (it ignores date ranges or event codes). Paste the current tools/outreach-collector/Code.gs into Apps Script and deploy a new version.');
    const S = Object.assign({ byDay: {}, byMonth: {}, codes: {}, devices: {}, quiz: {}, labFrom: {}, codeTotals: {} }, j);
    if (!S.generated) S.generated = new Date().toISOString();
    if (!S.until) S.until = S.generated.slice(0, 10);
    return S;
  }

  // ── tabs ─────────────────────────────────────────────────────────────────────────────────────────────────
  const TABS = { main: 'tab-main', report: 'tab-report', redesign: 'tab-redesign' };
  let tab = TABS[P.get('tab')] ? P.get('tab') : 'main';
  function showTab(name, focus) {
    tab = name;
    Object.keys(TABS).forEach(k => {
      const b = $(TABS[k]), on = k === name;
      b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1;
      $(k).hidden = !on || !summary;
      if (on && focus) b.focus();
    });
    $('filters').hidden = name === 'report';
    const q = new URLSearchParams(location.search); if (name === 'main') q.delete('tab'); else q.set('tab', name);
    history.replaceState(null, '', '?' + q.toString());
    hideTip();
    if (name === 'report' && summary && !report) runReport();
    if (name === 'redesign' && summary) fillRedesign();
  }
  function initTabs() {
    const keys = Object.keys(TABS);
    keys.forEach(k => $(TABS[k]).addEventListener('click', () => showTab(k)));
    $('tabs').addEventListener('keydown', e => {
      const i = keys.indexOf(tab), d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (d) { e.preventDefault(); showTab(keys[(i + d + keys.length) % keys.length], true); }
    });
  }

  // ── small output helpers ─────────────────────────────────────────────────────────────────────────────────
  async function copyText(text, statusEl) {
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch (e) {
      const t = h('textarea', { style: { position: 'fixed', left: '-9999px' } }); t.value = text; document.body.append(t); t.select();
      try { ok = document.execCommand('copy'); } catch (e2) { ok = false; } t.remove();
    }
    if (statusEl) statusEl.textContent = ok ? 'Copied.' : 'Could not copy; select the text and copy it by hand.';
    return ok;
  }
  function download(name, text, type) {
    const a = h('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name });
    document.body.append(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }
  const RP = window.QL_REPORT;

  // ── report (for proposals) ──────────────────────────────────────────────────────────────────────────────
  let report = null;
  function initReport() {
    const today = new Date().toISOString().slice(0, 10);
    $('rep-from').value = P.get('rfrom') || new Date(Date.now() - 364 * 86400000).toISOString().slice(0, 10);
    $('rep-to').value = P.get('rto') || today;
    $('rep-go').addEventListener('click', runReport);
    $('rep-copy').addEventListener('click', () => report && copyText(RP.reportText(report, SPECS, TITLES), $('rep-status')));
    const base = () => `qublitz-outreach-${report.since}-to-${report.until}${report.code ? '-' + report.code : ''}${report.demo ? '-DEMO' : ''}`;
    $('rep-csv-m').addEventListener('click', () => report && download(base() + '-games-by-month.csv', RP.monthsCsv(report, TITLES), 'text/csv'));
    $('rep-csv-e').addEventListener('click', () => report && download(base() + '-events.csv', RP.codesCsv(report), 'text/csv'));
    $('rep-json').addEventListener('click', () => report && download(base() + '.json', JSON.stringify(report, null, 2), 'application/json'));
    $('rep-print').addEventListener('click', () => window.print());
  }
  async function runReport() {
    const from = $('rep-from').value, to = $('rep-to').value, code = $('rep-code').value;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(from) || !/^\d{4}-\d{2}-\d{2}$/.test(to) || from > to) { $('rep-status').textContent = 'Pick a start date on or before the end date.'; return; }
    $('rep-status').textContent = 'Loading…'; $('rep-body').classList.add('loading');
    try {
      report = await fetchSummary({ days: daysSince(from), from, to, code });
      renderReport();
      $('rep-status').textContent = '';
    } catch (e) { $('rep-status').textContent = String(e && e.message || e); }
    finally { $('rep-body').classList.remove('loading'); }
  }
  function fillCodes(sel, codes, keep) {
    const v = keep != null ? keep : sel.value;
    sel.replaceChildren(h('option', { value: '', text: 'All visits' }), ...codes.map(c => h('option', { value: c, text: c })));
    sel.value = codes.includes(v) ? v : '';
  }
  function monthColumns(host, months) {
    const W = Math.max(260, host.clientWidth || 600), H = 180, pad = { l: 38, r: 8, t: 18, b: 24 };
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b, max = niceMax(Math.max(1, ...months.map(m => m.visits)));
    const band = iw / Math.max(1, months.length), bw = Math.min(24, band * 0.6);
    const Y = v => pad.t + ih - v / max * ih;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
      'aria-label': 'Visits to activities by month: ' + months.map(m => `${m.month} ${m.visits}`).join(', ') });
    for (let k = 0; k <= 4; k++) {
      const v = max * k / 4, y = Y(v);
      svg.append(s('line', { x1: pad.l, x2: W - pad.r, y1: y, y2: y, stroke: k ? 'var(--grid)' : 'var(--axis)', 'stroke-width': 1 }));
      const t = s('text', { x: pad.l - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)', style: 'font-variant-numeric:tabular-nums' }); t.textContent = fmt(v); svg.append(t);
    }
    const peak = months.reduce((b, m, i) => m.visits > months[b].visits ? i : b, 0), every = Math.ceil(months.length / (W < 500 ? 4 : 12));
    months.forEach((m, i) => {
      const x = pad.l + band * i + (band - bw) / 2, y = Y(m.visits), hgt = pad.t + ih - y, r = Math.min(4, hgt / 2, bw / 2);
      const d = hgt > 0 ? `M${x},${pad.t + ih}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${pad.t + ih}Z` : '';
      const g = s('g', { tabindex: '0', role: 'img', 'aria-label': `${m.month}: ${m.visits} visits, ${fmt(m.minutes / 60)} hours, ${m.completes} levels completed` });
      g.append(s('rect', { x: pad.l + band * i, y: pad.t, width: band, height: ih, fill: 'transparent' }));
      if (d) g.append(s('path', { d, fill: 'var(--s1)' }));
      if (i === peak || i === months.length - 1) { const t = s('text', { x: x + bw / 2, y: y - 5, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-2)' }); t.textContent = fmt(m.visits); g.append(t); }
      if (i % every === 0 || i === months.length - 1) { const t = s('text', { x: x + bw / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }); t.textContent = MON[+m.month.slice(5) - 1] + (i === 0 || m.month.endsWith('-01') ? ' ' + m.month.slice(2, 4) : ''); g.append(t); }
      hoverable(g, () => [m.month, [{ v: fmt(m.visits), l: 'visits' }, { v: fmt(m.minutes / 60), l: 'hours active', c: 'transparent' }, { v: fmt(m.completes), l: 'levels completed', c: 'transparent' },
        { v: fmt(m.debriefs + m.ingame), l: 'answers', c: 'transparent' }]]);
      svg.append(g);
    });
    host.replaceChildren(svg);
  }
  function renderReport() {
    const R = report, N = RP.numbers(R, SPECS, TITLES), body = $('rep-body');
    const full = n => n == null ? '–' : Math.round(n).toLocaleString('en-US');   // exact figures for proposals
    if (!R.code) fillCodes($('rep-code'), N.codes.map(c => c.code));
    const range = `${dLong(N.since)} – ${dLong(N.until)}`;
    const tile = (k, v, d, hero) => h('div', { class: 'tile' + (hero ? ' hero' : '') }, h('div', { class: 'k', text: k }), h('div', { class: 'v' }, v), d ? h('div', { class: 'd', text: d }) : null);
    const card = (title, note, ...kids) => h('section', { class: 'card' + (/^(Growth|Answers right|Activities|About)/.test(title) ? ' half' : '') }, h('h2', { text: title }), note ? h('p', { class: 'note', text: note }) : null, ...kids);
    const months = h('div', { class: 'chart' });
    const objRows = N.objectives.slice().sort((a, b) => b.n - a.n).slice(0, 14).sort((a, b) => b.acc - a.acc);
    body.replaceChildren(
      h('div', { class: 'rep-head' }, h('h2', {}, 'Qublitz outreach report', R.demo ? h('span', { class: 'rep-demo', text: 'DEMO DATA' }) : null),
        h('p', { text: `${range}${N.code ? ' · outreach event ' + N.code : ''} · https://mvwf.github.io/qublitz/` }),
        h('p', { text: 'Aggregate, anonymous counts. Visits are page visits, not unique people (there are no identifiers).' })),
      h('div', { class: 'kpis rep' },
        tile('Page visits to activities', full(N.visits), `not unique people · plus ${fmt(N.hubVisits)} to the hub`, true),
        tile('Activities used', fmt(N.activities.length), N.activities.slice(0, 3).map(a => a.title).join(', ') + (N.activities.length > 3 ? ', …' : '')),
        tile('Hours of active use', full(N.hours), 'a lower bound'),
        tile('Levels or missions completed', full(N.completes), ''),
        tile('Questions answered', full(N.answers), `${full(N.ingame)} in-game answers · ${full(N.debriefs)} debriefs`),
        tile('Answers right', pct(N.correct), 'across all learning objectives'),
        tile('Fun · learned', [fix1(N.fun), h('small', { text: ' · ' }), fix1(N.learned)], 'average ratings out of 5'),
        tile('Lab arrivals', full(N.labArrivals), 'visits to the Virtual Qubit Lab from another activity')),
      h('div', { class: 'grid' },
        card('Growth by month', 'Visits to activities per calendar month in the range (partial months at either end).', months,
          h('details', { class: 'tv' }, h('summary', { text: 'Table view' }), h('div', { class: 'tscroll' },
            table(['Month', 'Visits', 'Hours', 'Levels', 'Debriefs', 'In-game answers'], N.months.map(m => [m.month, fmt(m.visits), fmt(m.minutes / 60), fmt(m.completes), fmt(m.debriefs), fmt(m.ingame)]))))),
        card('Answers right by learning objective', `The ${objRows.length} objectives with the most answers, in-game and debrief together.`,
          objRows.length ? barRows(objRows.map(o => ({ label: `${o.gameTitle}: ${o.text}`, frac: o.acc, val: `${pct(o.acc)} of ${fmt(o.n)}`,
            aria: `${o.gameTitle}, ${o.text}: ${pct(o.acc)} right of ${o.n} answers`, tip: [o.gameTitle, [{ v: pct(o.acc), l: o.text }, { v: fmt(o.n), l: 'answers', c: 'transparent' }]] })), { ref: 0.5, wrap: true })
            : h('div', { class: 'empty', text: 'No answers in this range.' })),
        card('Outreach events', 'One row per event code in the range.', N.codes.length ? h('div', { class: 'tscroll' },
          table(['Event', 'Visits', 'Active minutes', 'Levels', 'Answers right', 'Fun', 'Learned'],
            N.codes.map(c => [c.code, fmt(c.visits), fmt(c.minutes), fmt(c.completes), pct(c.correct), fix1(c.fun), fix1(c.learned)]))) :
          h('div', { class: 'empty', text: 'No visits with an event code in this range.' })),
        card('Activities', 'Visits and active hours per activity.', h('div', { class: 'tscroll' },
          table(['Activity', 'Visits', 'Hours'], N.activities.map(a => [a.title, fmt(a.visits), fix1(a.minutes / 60)])))),
        card('About these numbers', '', h('ul', { class: 'caveats' }, RP.CAVEATS.map(c => h('li', { text: c })),
          h('li', { text: `Generated ${dLong(R.generated.slice(0, 10))} from the outreach usage dashboard.` + (R.demo ? ' DEMO DATA: synthetic, not real usage.' : '') })))));
    monthColumns(months, N.months);
  }

  // ── redesign briefs and before/after ─────────────────────────────────────────────────────────────────────
  function gameOptions(sel, keep) {
    const ids = Object.keys(summary.games).filter(g => g !== 'hub' && summary.games[g].opens).sort(byOpens);
    GAMES.forEach(g => { if (g.id !== 'hub' && !ids.includes(g.id)) ids.push(g.id); });
    const v = keep || sel.value;
    sel.replaceChildren(...ids.map(g => h('option', { value: g, text: title(g) })));
    if (ids.includes(v)) sel.value = v;
  }
  function fillRedesign() {
    gameOptions($('brief-game')); gameOptions($('cmp-game'));
    const sel = $('cmp-entry'), cur = sel.value;
    sel.replaceChildren(h('option', { value: '', text: 'Pick a date…' }),
      ...improvements.map((it, i) => /^\d{4}-\d{2}-\d{2}$/.test(it.date || '') ? h('option', { value: String(i), text: `${it.demo ? 'DEMO · ' : ''}${it.date} · ${title(it.game || '')}: ${String(it.change || '').slice(0, 60)}` }) : null));
    sel.value = cur;
    fillTargets();
  }
  function fillTargets() {
    const g = $('cmp-game').value, G = (summary.games || {})[g] || {}, spec = SPECS[g] || {};
    const lv = $('cmp-level'), ob = $('cmp-obj'), lvKeep = lv.value, obKeep = ob.value;
    lv.replaceChildren(h('option', { value: '', text: 'All levels' }), ...Object.keys(G.levels || {}).map(l => h('option', { value: l, text: l })));
    lv.value = (G.levels || {})[lvKeep] ? lvKeep : '';
    ob.replaceChildren(h('option', { value: '', text: 'All questions' }), ...(spec.objectives || []).map(o => h('option', { value: o.id, text: plain(o.text).slice(0, 70) })));
    ob.value = (spec.objectives || []).some(o => o.id === obKeep) ? obKeep : '';
  }
  function buildBrief(game) {
    if (game) $('brief-game').value = game;
    const g = $('brief-game').value;
    if (!g) return;
    const B = RP.brief(summary, SPECS, TITLES, g, { demo: DEMO });
    $('brief-md').value = `# ${B.title}\n\n${B.body}\n`;
    const link = RP.issueUrl(B.title, B.body, 7000);
    $('brief-issue').href = link.url;
    $('brief-status').textContent = link.truncated ? 'The issue link holds a shortened brief (URL length limit); paste the full text from the box.' : 'Opens a prefilled issue labelled feedback on GitHub.';
    $('brief-out').hidden = false;
    $('brief-copy').onclick = () => copyText($('brief-md').value, $('brief-status'));
  }
  async function runCompare() {
    const date = $('cmp-date').value, g = $('cmp-game').value, out = $('cmp-out');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || date > today()) { $('cmp-status').textContent = 'Pick a change date up to today.'; return; }
    const w = RP.windows(date, today(), 180);
    $('cmp-status').textContent = 'Loading…';
    try {
      const [b, a] = await Promise.all([fetchSummary({ days: daysSince(w.before.from), from: w.before.from, to: w.before.to }),
        fetchSummary({ days: daysSince(w.after.from), from: w.after.from, to: w.after.to })]);
      const rows = RP.compare(b, a, { game: g, level: $('cmp-level').value, objective: $('cmp-obj').value }, SPECS);
      const val = (r, x) => x.v == null ? '–' : r.unit === 'pct' ? pct(x.v) : r.unit === 'rating' ? fix1(x.v) + ' / 5' : fix1(x.v) + ' min';
      const frac = (r, x) => x.v == null ? 0 : r.unit === 'pct' ? x.v : r.unit === 'rating' ? (x.v - 1) / 4 : Math.min(1, x.v / 10);
      const delta = r => { if (!r.enough) return '–'; const d = r.after.v - r.before.v; const t = r.unit === 'pct' ? Math.round(d * 100) + ' pts' : r.unit === 'rating' ? d.toFixed(1) : d.toFixed(1) + ' min'; return (d > 0 ? '+' : d < 0 ? '−' : '±') + t.replace('-', ''); };
      const side = (r, x, cls) => h('div', { class: 'side' }, h('div', {}, h('span', { class: 'num', text: val(r, x) }), h('span', { class: 'sub', text: `  n = ${fmt(x.n)}` })),
        h('div', { class: 'barbox', 'aria-hidden': 'true' }, h('div', { class: 'bar ' + cls, style: { width: (frac(r, x) * 100).toFixed(1) + '%' } })));
      out.replaceChildren(
        h('p', { class: 'note', text: `${title(g)} · before: ${dLong(w.before.from)} – ${dLong(w.before.to)} · after: ${dLong(w.after.from)} – ${dLong(w.after.to)} (${w.days} days each)` + (DEMO ? ' · DEMO DATA' : '') }),
        h('div', { class: 'cmp', role: 'table', 'aria-label': 'Before and after' },
          h('div', { class: 'cmp-row head', role: 'row' }, h('span', { role: 'columnheader', class: 'lab', text: 'Metric' }), h('span', { role: 'columnheader', text: 'Before' }), h('span', { role: 'columnheader', text: 'After' }), h('span', { role: 'columnheader', class: 'delta', text: 'Change' })),
          rows.map(r => h('div', { class: 'cmp-row' + (r.enough ? '' : ' thin'), role: 'row' },
            h('span', { class: 'lab', role: 'rowheader', text: r.label }), side(r, r.before, 'before'), side(r, r.after, 'after'), h('span', { class: 'delta', text: delta(r) }),
            r.enough ? null : h('span', { class: 'why', text: `too few answers to compare (needs ${r.unit === 'rating' ? RP.COMPARE.MIN_RATINGS : RP.COMPARE.MIN_N} on each side)` })))),
        h('p', { class: 'note', text: 'Before and after are different visitors and seasons; treat a change as a signal to look closer, not as proof.' }));
      $('cmp-status').textContent = '';
    } catch (e) { $('cmp-status').textContent = String(e && e.message || e); }
  }
  function initRedesign() {
    $('brief-go').addEventListener('click', () => buildBrief());
    $('cmp-game').addEventListener('change', fillTargets);
    $('cmp-go').addEventListener('click', runCompare);
    $('cmp-entry').addEventListener('change', () => {
      const it = improvements[+$('cmp-entry').value];
      if (!it) return;
      $('cmp-date').value = it.date; if (it.game) $('cmp-game').value = it.game; fillTargets();
      $('cmp-level').value = it.level || ''; $('cmp-obj').value = it.objective || '';
      runCompare();
    });
  }
  function openBrief(game) { showTab('redesign'); buildBrief(game); $('brief-h').scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  function openCompare(i) {
    showTab('redesign'); $('cmp-entry').value = String(i);
    $('cmp-entry').dispatchEvent(new Event('change')); $('cmp-h').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── states ─────────────────────────────────────────────────────────────────────────────────────────────
  function setBanner(...nodes) { $('banner').replaceChildren(...nodes); }
  function offState() {
    $('main').hidden = true;
    $('days').closest('.filters').hidden = true;
    $('tabs').hidden = true;
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
      summary = await fetchSummary({ days: n });
      if (since) summary.since = since > summary.since ? since : summary.since;
      if (!DEMO) setBanner();
      showTab(tab);
      renderAll();
      if (tab === 'redesign') fillRedesign();
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
  // Every visit draws from its own seeded generator (day, page, visit number), so any date range or event-code
  // filter gives the same visits as the full history: before/after windows and per-event reports stay consistent.
  function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const hash = str => { let x = 2166136261; for (let i = 0; i < str.length; i++) { x ^= str.charCodeAt(i); x = Math.imul(x, 16777619); } return x >>> 0; };
  const DEMO_CHANGES = { beamline: 45, arena: 90 };   // days ago of the two DEMO "you said, we did" entries
  function demoSummary(o) {
    const now = new Date(), today = now.toISOString().slice(0, 10);
    const start = o.from || new Date(now.getTime() - o.days * 86400000).toISOString().slice(0, 10);
    const end = o.to && o.to < today ? o.to : today, only = o.code || '';
    const W = { hub: 2.0, 'qubit-lab': 1.3, copter: 1.2, beamline: 1.1, rider: 1.0, arena: 0.8, sonify: 0.6, trio: 0.45, string: 0.45, spectrogram: 0.4, drum: 0.35, fridge: 0.3, 'ep-tpd': 0.25 };
    const LV = { beamline: ['first-light', 'half-turn', 'phase', 'order', 'crosstalk', 'finale'], copter: ['hover', 'tilt', 'xyz', 'precise'], rider: ['1-1', '1-2', '1-3', '2-1', '2-2'],
      arena: ['training', 'duel-1', 'duel-2', 'mission'] };
    const EVENTS = [{ code: 'SCIFAIR26', ago: 12, size: 70 }, { code: 'ENGS53-W26', ago: 40, size: 45 }, { code: 'OPENHOUSE', ago: 75, size: 55 },
      { code: 'HSVISIT', ago: 130, size: 40 }, { code: 'SUMMERCAMP', ago: 200, size: 60 }, { code: 'LIBRARY', ago: 290, size: 30 }];
    const STEPS = { 'qubit-lab': ['rabi', 'ramsey', 't1', 'echo'], sonify: ['upload', 'play'], trio: ['heat', 'wave', 'schrodinger'] };
    const STOP_AT = { 'beamline:crosstalk': '|third-mirror', 'rider:2-2': '|boss' };
    const out = { generated: now.toISOString(), days: o.days, since: start, until: end, code: only, demo: true, games: {}, byDay: {}, byMonth: {}, codes: {}, devices: {},
      codeGames: {}, codeQuiz: {}, codeTotals: {}, quiz: {}, labFrom: {}, totalMinutes: 0 };
    const G = id => out.games[id] || (out.games[id] = { opens: 0, starts: 0, completes: 0, wins: 0, losses: 0, quizOpens: 0, leaves: 0, durSum: 0, levels: {}, events: {}, quits: {}, completedPerVisit: {} });
    const MO = (d, id) => { const M = out.byMonth[d.slice(0, 7)] = out.byMonth[d.slice(0, 7)] || {}; return M[id] || (M[id] = { opens: 0, minutes: 0, completes: 0, debriefs: 0, ingame: 0 }); };
    const quizzable = id => id !== 'hub' && (LV[id] || (SPECS[id] && SPECS[id].quiz && SPECS[id].quiz.length) || ['qubit-lab', 'trio'].includes(id));
    // question items only where a learning file has them (the rest still get ratings), so no question is made up
    const items = id => (SPECS[id] && SPECS[id].quiz && SPECS[id].quiz.length) ? SPECS[id].quiz.map(q => ({ id: q.id, obj: q.objective, k: (q.choices || []).length || 4, ans: q.answer })) : [];
    const diff = (id, q, qi) => { const r = rng(hash('item:' + id + ':' + q.id)); const p = (qi === 1 && LV[id]) || (id === 'arena' && qi === 0) ? 0.34 : 0.55 + r() * 0.35;
      return { p, fav: (q.ans + 1 + Math.floor(r() * (q.k - 1))) % q.k }; };
    const rate = { copter: [4.4, 3.6], beamline: [4.1, 4.0], rider: [4.3, 3.3], 'qubit-lab': [3.4, 4.2], arena: [4.5, 2.7], trio: [3.0, 3.9] };
    const blankQ = () => ({ n: 0, ingame: 0, sc: 0, it: 0, f: [0, 0], l: [0, 0], items: {} });
    for (const d of dayList(start, end)) {
      const ago = Math.round((Date.parse(today) - Date.parse(d)) / 86400000), wd = new Date(d + 'T00:00:00Z').getUTCDay();
      const base = (wd === 0 || wd === 6 ? 4 : 9) * (1 + 0.4 * Math.sin(ago / 9)) * (0.55 + 0.45 * Math.max(0, 1 - ago / 400));   // slow growth
      const ev = EVENTS.find(e => e.ago === ago);
      for (const id in W) {
        const dr = rng(hash(d + ':' + id)), pois = m => { let L = Math.exp(-m), k = 0, p = 1; do { k++; p *= dr(); } while (p > L && k < 500); return k - 1; };
        const k = pois(base * W[id] / 2) + (ev ? pois(ev.size * W[id] / 6) : 0);
        for (let v = 0; v < k; v++) {
          const rand = rng(hash(d + ':' + id + ':' + v));
          const code = ev && rand() < 0.85 ? ev.code : '';
          if (only && code !== only) continue;
          const g = G(id), mo = MO(d, id);
          g.opens++; mo.opens++; g.events.open = (g.events.open || 0) + 1;
          (out.byDay[d] = out.byDay[d] || {})[id] = (out.byDay[d][id] || 0) + 1;
          if (code) { out.codes[code] = (out.codes[code] || 0) + 1; const cg = out.codeGames[code] = out.codeGames[code] || {}; cg[id] = (cg[id] || 0) + 1; }
          const CT = code ? (out.codeTotals[code] = out.codeTotals[code] || { minutes: 0, completes: 0 }) : null;
          const dev = rand() < (code ? 0.62 : 0.38) ? 'touch' : 'desktop'; out.devices[dev] = (out.devices[dev] || 0) + 1;
          let minutes = id === 'hub' ? 0.3 + rand() * 0.6 : id === 'sonify' ? 0.3 + rand() * 1.1 : 1 + rand() * 9;
          let place = 'menu', done = 0, ingameAsked = false;
          // the DEMO change log: a hint on beamline's crosstalk level, and a shorter arena debrief
          const hard = { 'beamline:crosstalk': ago < DEMO_CHANGES.beamline ? 0.62 : 0.3, 'rider:2-2': 0.35 };
          if (id !== 'hub' && rand() < 0.82) {
            g.starts++;
            if (STEPS[id]) place = STEPS[id][Math.min(STEPS[id].length - 1, Math.floor(rand() * rand() * STEPS[id].length * 1.6))];
            for (const lv of LV[id] || []) {
              const L = g.levels[lv] = g.levels[lv] || { starts: 0, completes: 0, starsSum: 0 };
              L.starts++; place = lv;
              if (rand() > (hard[id + ':' + lv] != null ? hard[id + ':' + lv] : 0.85)) { if (STOP_AT[id + ':' + lv] && rand() < 0.7) place += STOP_AT[id + ':' + lv]; break; }
              L.completes++; g.completes++; mo.completes++; if (CT) CT.completes++; done++; L.starsSum += 1 + Math.floor(rand() * 3);
              if (lv === 'half-turn' || lv === 'tilt' || lv === '1-2' || lv === 'duel-1') ingameAsked = true;
              if (rand() < 0.25) break;
            }
          }
          if (ingameAsked && items(id).length) {   // one question asked inside the game
            const q = items(id)[0], Q = out.quiz[id] = out.quiz[id] || blankQ();
            const I = Q.items[q.id] = Q.items[q.id] || { objective: q.obj, n: 0, correct: 0, picks: {}, wrong: {}, ingame: { n: 0, correct: 0 } };
            const ok = rand() < 0.74, pick = ok ? q.ans : (q.ans + 1) % q.k;
            Q.ingame++; mo.ingame++; Q.sc += ok ? 1 : 0; Q.it++; I.n++; I.ingame.n++; I.picks[pick] = (I.picks[pick] || 0) + 1;
            if (code) { const CQ = out.codeQuiz[code] = out.codeQuiz[code] || blankQ(); CQ.ingame++; CQ.sc += ok ? 1 : 0; CQ.it++; }
            if (ok) { I.correct++; I.ingame.correct++; } else I.wrong[pick] = (I.wrong[pick] || 0) + 1;
          }
          if (id !== 'qubit-lab' && id !== 'hub' && rand() < (id === 'copter' || id === 'rider' ? 0.16 : 0.07)) {
            g.events.to_lab = (g.events.to_lab || 0) + 1;
            if (rand() < 0.72) out.labFrom[id] = (out.labFrom[id] || 0) + 1;
          }
          if (quizzable(id) && rand() < (id === 'arena' ? 0.3 : 0.24)) {
            g.quizOpens++;
            const sendP = id === 'qubit-lab' ? 0.4 : id === 'arena' ? (ago < DEMO_CHANGES.arena ? 0.8 : 0.42) : 0.78;
            if (rand() < sendP) {
              const Q = out.quiz[id] = out.quiz[id] || blankQ();
              const CQ = code ? (out.codeQuiz[code] = out.codeQuiz[code] || blankQ()) : null;
              let right = 0; const all = items(id), its = all.slice(ingameAsked ? 1 : 0);
              its.forEach(q => {
                const df = diff(id, q, all.indexOf(q));
                const I = Q.items[q.id] = Q.items[q.id] || { objective: q.obj, n: 0, correct: 0, picks: {}, wrong: {}, ingame: { n: 0, correct: 0 } };
                const ok = rand() < df.p;
                let pick = q.ans;
                if (!ok) { pick = rand() < 0.65 ? df.fav : (q.ans + 1 + Math.floor(rand() * (q.k - 1))) % q.k; if (pick === q.ans) pick = df.fav; I.wrong[pick] = (I.wrong[pick] || 0) + 1; }
                I.n++; if (ok) { I.correct++; right++; } I.picks[pick] = (I.picks[pick] || 0) + 1;
              });
              const rr = rate[id] || [3.8, 3.7];
              const fun = Math.max(1, Math.min(5, Math.round(rr[0] + (rand() - 0.5) * 2.4))), lea = Math.max(1, Math.min(5, Math.round(rr[1] + (rand() - 0.5) * 2.4)));
              for (const A of [Q, CQ]) if (A) { A.n++; A.sc += right; A.it += its.length; A.f[0] += fun; A.f[1]++; A.l[0] += lea; A.l[1]++; }
              mo.debriefs++;
              minutes += 1.2;
            }
          }
          g.quits[place] = (g.quits[place] || 0) + 1; g.completedPerVisit[done] = (g.completedPerVisit[done] || 0) + 1;
          if (rand() < 0.9) { g.leaves++; g.durSum += minutes * 60; mo.minutes += minutes; if (CT) CT.minutes += minutes; }
        }
      }
    }
    const r1 = x => Math.round(x * 10) / 10;
    for (const id in out.games) { const g = out.games[id]; g.avgMinutes = g.leaves ? Math.round(g.durSum / g.leaves / 6) / 10 : null; g.totalMinutes = r1(g.durSum / 60); out.totalMinutes += g.durSum / 60; delete g.durSum; }
    out.totalMinutes = r1(out.totalMinutes);
    Object.values(out.byMonth).forEach(M => Object.values(M).forEach(x => { x.minutes = r1(x.minutes); }));
    Object.values(out.codeTotals).forEach(x => { x.minutes = r1(x.minutes); });
    const fin = A => { A.avgScore = A.it ? Math.round(A.sc / A.it * 1000) / 1000 : null; A.avgFun = A.f[1] ? Math.round(A.f[0] / A.f[1] * 100) / 100 : null;
      A.avgLearned = A.l[1] ? Math.round(A.l[0] / A.l[1] * 100) / 100 : null; delete A.sc; delete A.it; delete A.f; delete A.l; if (!A.items) delete A.items; };
    Object.values(out.quiz).forEach(fin); Object.values(out.codeQuiz).forEach(fin);
    return out;
  }
  function demoImprovements() {
    const d = k => new Date(Date.now() - k * 86400000).toISOString().slice(0, 10);
    return [
      { date: d(DEMO_CHANGES.beamline), game: 'beamline', level: 'crosstalk', change: 'Added a hint to the crosstalk level and moved it after “order”.', because: 'level crosstalk: 31% of starts completed', demo: true },
      { date: d(DEMO_CHANGES.arena), game: 'arena', change: 'Shortened the debrief from six questions to four.', because: 'debrief drop-off: 38% of openings sent', demo: true },
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
  initTabs(); initReport(); initRedesign();
  if (!DEMO && !ENDPOINT) { offState(); return; }
  Promise.all([loadSpecs(), loadImprovements()]).then(load);
  let rt = 0, lastW = 0;
  new ResizeObserver(ents => {
    const w = Math.round(ents[0].contentRect.width);
    if (w === lastW) return; lastW = w;
    clearTimeout(rt); rt = setTimeout(() => { renderAll(); if (report && tab === 'report') renderReport(); }, 120);
  }).observe(document.body);
})();
