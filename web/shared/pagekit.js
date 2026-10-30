/* pagekit.js: two small helpers shared by the demo pages. No dependencies; scripts/build_standalone.py
 * inlines it into the single-file downloads.
 *
 *   PageKit.copyLink(button, path, query)
 *     Copies a link that reopens the page with the given settings. In the offline download the link
 *     points at the live site, since a file:// path means nothing to whoever receives it.
 *
 *   PageKit.guided({ steps, apply, before, key })
 *     Predict-then-reveal checkpoints, shown when the page is opened with ?guided=1. Each step sets
 *     the page up (apply(step.setup)), hides the plots that would give the answer away (step.cover,
 *     a list of selectors), asks for a prediction (step.choices with step.answer = index, or a
 *     number with step.answer, step.tol and step.unit), and only then reveals: apply(step.reveal),
 *     uncover, and show step.explain. Progress is kept in localStorage when the browser allows it.
 */
(function (root) {
  'use strict';
  const SITE = 'https://mvwf.github.io/qublitz/';
  const doc = root.document;

  function pageURL(path, query) {
    const q = query ? (query[0] === '?' ? query : '?' + query) : '';
    if (root.QUBLITZ_STANDALONE || location.protocol === 'file:') return SITE + path + q;
    return location.origin + location.pathname + q;
  }

  async function copyLink(button, path, query) {
    const url = pageURL(path, query), label = button.textContent;
    try { history.replaceState(null, '', (query ? (query[0] === '?' ? query : '?' + query) : location.pathname) + location.hash); }
    catch (e) { /* file:// pages can refuse replaceState */ }
    try { await navigator.clipboard.writeText(url); button.textContent = 'Copied ✓'; }
    catch (e) { root.prompt('Copy this link:', url); }
    setTimeout(() => { button.textContent = label; }, 1500);
    return url;
  }

  // URLSearchParams with typed, range-checked getters, so a mangled link falls back to defaults.
  function params(search) {
    const q = new URLSearchParams(search ?? location.search);
    return {
      has: k => q.has(k),
      str: (k, allowed) => { const v = q.get(k); return v != null && (!allowed || allowed.includes(v)) ? v : null; },
      num: (k, lo, hi) => { const v = parseFloat(q.get(k)); return Number.isFinite(v) && v >= lo && v <= hi ? v : null; },
      list: (k, n) => { const s = q.get(k); if (!s) return null; const v = s.split(',').map(Number);
        return v.every(Number.isFinite) && (!n || v.length === n) ? v : null; },
      raw: q,
    };
  }

  /* ---------------- guided mode ---------------- */
  const CSS = `
  .gd-panel{border:2px solid var(--green,#00693e);border-radius:12px;background:#fff;padding:14px 16px 14px;margin:0 0 16px;
    font-family:Helvetica,Arial,sans-serif;color:var(--ink,#1b2a24);box-shadow:0 6px 22px rgba(0,105,62,.10)}
  .gd-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
  .gd-head b{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--green,#00693e)}
  .gd-dots{display:flex;gap:5px;flex:1}
  .gd-dots i{width:22px;height:6px;border-radius:3px;background:var(--line,#d8e2dc)}
  .gd-dots i.done{background:var(--green2,#12855a)} .gd-dots i.miss{background:#d9a521} .gd-dots i.cur{outline:2px solid var(--green,#00693e);outline-offset:1px}
  .gd-panel h3{margin:2px 0 6px;font-size:17px}
  .gd-q{font-size:14px;line-height:1.5;margin:0 0 10px;max-width:900px}
  .gd-choices{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 10px}
  .gd-choices button,.gd-panel .gd-btn{font:inherit;font-size:13.5px;border:1px solid var(--line,#d8e2dc);background:#fff;color:inherit;
    border-radius:8px;padding:7px 12px;cursor:pointer}
  .gd-choices button[aria-pressed=true]{border-color:var(--green,#00693e);background:#e3f3ea;font-weight:600}
  .gd-choices button.right{border-color:var(--green,#00693e);background:#e3f3ea}
  .gd-choices button.wrong{border-color:#c84b31;background:#fdeae4}
  .gd-choices button:disabled{cursor:default;color:inherit}
  .gd-num{display:flex;align-items:center;gap:8px;margin:0 0 10px;font-size:14px}
  .gd-num input{font:inherit;font-size:14px;width:120px;padding:6px 8px;border:1px solid var(--line,#d8e2dc);border-radius:7px}
  .gd-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .gd-panel .gd-btn.primary{background:var(--green,#00693e);border-color:var(--green,#00693e);color:#fff}
  .gd-panel .gd-btn:disabled{opacity:.45;cursor:default}
  .gd-panel .gd-exit{margin-left:auto;font-size:12.5px;color:var(--muted,#5a6b63)}
  .gd-fb{margin:10px 0 10px;padding:10px 12px;border-radius:8px;background:var(--panel,#f5f9f6);font-size:13.8px;line-height:1.5;max-width:900px}
  .gd-fb .verdict{font-weight:700;margin-right:6px}
  .gd-fb .verdict.ok{color:var(--green,#00693e)} .gd-fb .verdict.no{color:#b5531e}
  .gd-cover{position:relative}
  .gd-cover>*{filter:blur(9px) grayscale(.6);pointer-events:none;user-select:none}
  .gd-cover::after{content:'Make your prediction first';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    font:600 15px Helvetica,Arial,sans-serif;color:var(--green,#00693e);background:rgba(245,249,246,.35);border-radius:10px;z-index:5}
  .gd-entry{font-size:12.5px;margin-left:8px}
  @media (prefers-reduced-motion: no-preference){ .gd-cover>*{transition:filter .35s} }`;

  function el(tag, attrs, html) {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (html != null) e.innerHTML = html;
    return e;
  }

  function guided(opts) {
    const { steps, apply, before, key } = opts;
    if (!doc.getElementById('gd-css')) doc.head.appendChild(el('style', { id: 'gd-css' }, CSS));
    const store = 'qublitz-guided:' + key;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(store) || 'null'); } catch (e) { saved = null; }
    const answers = (saved && Array.isArray(saved.answers) && saved.answers.length === steps.length) ? saved.answers : steps.map(() => null);
    let k = saved && saved.k >= 0 && saved.k <= steps.length ? saved.k : 0;
    const save = () => { try { localStorage.setItem(store, JSON.stringify({ k, answers })); } catch (e) { /* storage off */ } };

    const panel = el('section', { class: 'gd-panel', 'aria-live': 'polite' });
    before.parentNode.insertBefore(panel, before);
    let covered = [];

    function cover(sels) {
      uncover();
      covered = (sels || []).flatMap(s => [...doc.querySelectorAll(s)]);
      covered.forEach(c => c.classList.add('gd-cover'));
    }
    function uncover() { covered.forEach(c => c.classList.remove('gd-cover')); covered = []; }
    const isRight = (s, a) => a != null && (s.choices ? a === s.answer : Math.abs(a - s.answer) <= s.tol);
    const fmt = (s, v) => s.choices ? s.choices[v] : `${v}${s.unit ? ' ' + s.unit : ''}`;

    function dots() {
      return steps.map((s, i) => `<i class="${i === k ? 'cur ' : ''}${answers[i] == null ? '' : isRight(s, answers[i]) ? 'done' : 'miss'}" title="Step ${i + 1}"></i>`).join('');
    }

    function head(label) {
      return `<div class="gd-head"><b>${label}</b><div class="gd-dots">${dots()}</div>
        <a href="#" class="gd-exit" data-act="exit">Leave guided mode</a></div>`;
    }

    function showStep(fresh) {
      const s = steps[k];
      const done = answers[k] != null;
      if (fresh || !done) apply(s.setup || {});
      if (done) { uncover(); apply(s.reveal || {}); } else cover(s.cover);
      let pick = done ? answers[k] : null;
      const input = s.choices
        ? `<div class="gd-choices" role="group">${s.choices.map((c, i) =>
            `<button type="button" data-i="${i}" aria-pressed="${pick === i}" ${done ? 'disabled' : ''}
              class="${done ? (i === s.answer ? 'right' : i === pick ? 'wrong' : '') : ''}">${c}</button>`).join('')}</div>`
        : `<label class="gd-num">My prediction <input type="number" step="any" ${done ? `value="${pick}" disabled` : ''}> ${s.unit || ''}</label>`;
      const fb = done ? `<div class="gd-fb"><span class="verdict ${isRight(s, pick) ? 'ok' : 'no'}">${isRight(s, pick) ? 'Your prediction holds.' : 'Not quite.'}</span>
          You said <b>${fmt(s, pick)}</b>; the answer is <b>${fmt(s, s.answer)}</b>.<br>${s.explain}</div>` : '';
      panel.innerHTML = head(`Guided · step ${k + 1} of ${steps.length}`) +
        `<h3>${s.title}</h3>${s.intro ? `<p class="gd-q">${s.intro}</p>` : ''}<p class="gd-q"><b>Predict:</b> ${s.ask}</p>${input}${fb}
        <div class="gd-row">
          ${done ? '' : '<button type="button" class="gd-btn primary" data-act="lock" disabled>Lock in my prediction</button>'}
          ${k > 0 ? '<button type="button" class="gd-btn" data-act="back">◂ Back</button>' : ''}
          ${done ? `<button type="button" class="gd-btn primary" data-act="next">${k + 1 < steps.length ? 'Next ▸' : 'Finish'}</button>` : ''}
          ${done && s.try ? `<span class="gd-q" style="margin:0 0 0 6px">${s.try}</span>` : ''}
        </div>`;
      const lock = panel.querySelector('[data-act=lock]');
      if (!done) {
        if (s.choices) panel.querySelectorAll('.gd-choices button').forEach(b => b.addEventListener('click', () => {
          pick = +b.dataset.i;
          panel.querySelectorAll('.gd-choices button').forEach(x => x.setAttribute('aria-pressed', x === b));
          lock.disabled = false;
        }));
        else { const inp = panel.querySelector('.gd-num input');
          inp.addEventListener('input', () => { pick = parseFloat(inp.value); lock.disabled = !Number.isFinite(pick); });
          inp.addEventListener('keydown', e => { if (e.key === 'Enter' && !lock.disabled) lock.click(); }); }
        lock.addEventListener('click', () => { answers[k] = pick; save(); showStep(false); });
      }
    }

    function showSummary() {
      uncover();
      const right = steps.filter((s, i) => isRight(s, answers[i])).length;
      panel.innerHTML = head('Guided · done') + `<h3>${right} of ${steps.length} predictions held up</h3>
        <p class="gd-q">The ones that didn't are the interesting ones: go back to any step with the dots' order in mind, or explore
        freely below. Every control is unlocked now.</p>
        <div class="gd-row"><button type="button" class="gd-btn" data-act="back">◂ Back</button>
          <button type="button" class="gd-btn primary" data-act="restart">Start over</button></div>`;
    }

    function render(fresh) { if (k >= steps.length) showSummary(); else showStep(fresh); save(); }

    panel.addEventListener('click', e => {
      const a = e.target.closest('[data-act]'); if (!a) return;
      const act = a.dataset.act;
      if (act === 'exit') {
        e.preventDefault(); uncover(); panel.remove();
        try { const u = new URL(location.href); u.searchParams.delete('guided'); history.replaceState(null, '', u); } catch (err) { /* file:// */ }
        if (opts.onExit) opts.onExit();
        return;
      }
      if (act === 'next') k++;
      if (act === 'back') k = Math.max(0, k - 1);
      if (act === 'restart') { k = 0; answers.fill(null); }
      render(true);
      panel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
    render(true);
  }

  // A small "Guided tour" link for pages that support it, placed after `anchor`.
  function guidedEntry(anchor, start) {
    const a = el('a', { href: '?guided=1', class: 'gd-entry' }, 'Try the guided predictions ▸');
    a.addEventListener('click', e => {
      e.preventDefault();
      try { history.replaceState(null, '', '?guided=1' + location.hash); } catch (err) { /* file:// */ }
      a.remove(); start();
    });
    anchor.appendChild(a);
    return a;
  }

  const wantsGuided = () => new URLSearchParams(location.search).get('guided') === '1';

  root.PageKit = { SITE, pageURL, copyLink, params, guided, guidedEntry, wantsGuided };
})(typeof window !== 'undefined' ? window : globalThis);
