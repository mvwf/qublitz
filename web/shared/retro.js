/* retro.js: the QuBlitz Arena's retro look (deep navy, neon cyan / gold / magenta, Press Start 2P headings, Space
 * Grotesk text) for every page on the site: the hub, the Qublitz games and research tools, ENGS 23, privacy, dashboard.
 *
 * Load it in <head>, BEFORE Plotly, right after retro-fonts.js:
 *   <script src="../shared/retro-fonts.js"></script><script src="../shared/retro.js"></script>
 * It runs before first paint, so there is no flash of the old light theme. It
 *   1. adds class "retro retro-<theme>" to <html> and injects the theme's CSS (page tokens like --green, --ink and
 *      --panel are redefined, so most of each page's own rules follow automatically),
 *   2. wraps Plotly.newPlot / react / relayout / restyle / update so every chart gets a dark, see-through layout
 *      and the light theme's dark ink colours are swapped for their neon equivalents (physics untouched: only colours),
 *   3. adds a small theme switch to the site bar.
 *
 * Themes: arena (default), synthwave, phosphor, and classic (the original light Dartmouth look, handy for
 * projectors). Choose with ?theme=<name>, or with the switch; the choice is remembered in this browser only.
 * Pages can tweak themselves with selectors under html.retro (e.g. html.retro .myThing{...}).
 */
(function (root) {
  'use strict';
  const doc = root.document, html = doc.documentElement;
  const THEMES = ['arena', 'synthwave', 'phosphor', 'classic'];
  const LABEL = { arena: 'Arena', synthwave: 'Synthwave', phosphor: 'Phosphor', classic: 'Classic' };
  const KEY = 'qublitz_theme';

  let theme = 'arena';
  try { const s = root.localStorage.getItem(KEY); if (THEMES.includes(s)) theme = s; } catch (e) { /* storage blocked */ }
  try { const q = new URLSearchParams(root.location.search).get('theme'); if (THEMES.includes(q)) theme = q; } catch (e) { /* no URL */ }

  // ── palettes ───────────────────────────────────────────────────────────────────────────────────────────────────
  // Each theme maps the pages' own tokens (--green is the accent, --ink text, --muted secondary text, --line borders,
  // --panel tinted boxes, --night the game stages) plus the Arena tokens (--r-*).
  const P = {
    arena: {
      bg: '#070a16', bg2: '#0b1024', glowA: '#16203f', glowB: '#1c1448',
      ink: '#eaf0ff', dim: '#9aa3cf', accent: '#2fe0d0', accent2: '#5cf0e2', gold: '#f5b73c', rose: '#ff4d9d',
      indigo: '#8b7dff', red: '#ff5a72', blue: '#5fb4ff', green: '#3fe08a',
      panel: 'rgba(18,24,52,0.62)', panel2: 'rgba(28,36,76,0.70)', line: 'rgba(124,140,230,0.24)', lineHi: 'rgba(124,140,230,0.50)',
      input: 'rgba(6,9,22,0.75)', title: '#f5b73c', titleGlow: 'rgba(245,183,60,.45)', onAccent: '#061016',
    },
    synthwave: {
      bg: '#12051f', bg2: '#1d0a33', glowA: '#3a0f4f', glowB: '#4a1236',
      ink: '#fbeaff', dim: '#c7a6d9', accent: '#ff4fd8', accent2: '#ff86e6', gold: '#ffcf4a', rose: '#ff7a45',
      indigo: '#a98bff', red: '#ff5a72', blue: '#42d6ff', green: '#4dfcb0',
      panel: 'rgba(38,12,60,0.62)', panel2: 'rgba(56,18,84,0.70)', line: 'rgba(255,110,220,0.22)', lineHi: 'rgba(255,110,220,0.50)',
      input: 'rgba(16,4,28,0.78)', title: '#42d6ff', titleGlow: 'rgba(255,79,216,.55)', onAccent: '#1a0420',
    },
    phosphor: {
      bg: '#030a05', bg2: '#06130a', glowA: '#0b2614', glowB: '#082010',
      ink: '#c9ffd9', dim: '#6fbf8a', accent: '#39ff7a', accent2: '#8dffb0', gold: '#e8ff6a', rose: '#ffb347',
      indigo: '#7affd4', red: '#ff6b5a', blue: '#6af2ff', green: '#39ff7a',
      panel: 'rgba(6,26,12,0.70)', panel2: 'rgba(10,38,18,0.78)', line: 'rgba(57,255,122,0.20)', lineHi: 'rgba(57,255,122,0.48)',
      input: 'rgba(2,10,4,0.82)', title: '#39ff7a', titleGlow: 'rgba(57,255,122,.55)', onAccent: '#021006',
    },
  };

  const FONTS = (root.RETRO_FONTS || []).map(f =>
    `@font-face{font-family:'${f.family}';font-style:normal;font-weight:${f.weight};font-display:swap;` +
    `src:url(data:font/woff2;base64,${f.woff2}) format('woff2')}`).join('\n');
  const PIX = `'Press Start 2P',ui-monospace,monospace`, BODY = `'Space Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif`;

  function css(p) {
    return `${FONTS}
html.retro{color-scheme:dark;
  --green:${p.accent};--green2:${p.accent2};--ink:${p.ink};--muted:${p.dim};--line:${p.line};--panel:${p.panel2};
  --night:${p.bg};--accent:${p.red};--blue:${p.blue};--gold:${p.gold};--int:${p.red};--ext:${p.blue};--heat:${p.red};--wave:${p.blue};--sch:${p.accent};
  --cx:#ff7a8a;--cy:#7ee08a;--cz:#7fb8ff;--dx:#ff7a8a;--dy:#7ee08a;--dz:#7fb8ff;
  --r-bg:${p.bg};--r-panel:${p.panel};--r-panel2:${p.panel2};--r-line:${p.line};--r-line-hi:${p.lineHi};--r-ink:${p.ink};
  --r-dim:${p.dim};--r-accent:${p.accent};--r-gold:${p.gold};--r-rose:${p.rose};--r-indigo:${p.indigo};--r-input:${p.input};
  --r-pixel:${PIX};--r-body:${BODY};background:${p.bg}}
html.retro body{background:
  radial-gradient(1200px 800px at 15% -10%,${p.glowA} 0%,transparent 55%),
  radial-gradient(1100px 900px at 110% 8%,${p.glowB} 0%,transparent 55%),
  linear-gradient(160deg,${p.bg} 0%,${p.bg2} 55%,${p.bg} 100%);
  background-attachment:fixed;color:${p.ink};font-family:${BODY}}
html.retro body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.55;
  background-image:radial-gradient(1px 1px at 12% 18%,#fff8 50%,transparent 51%),radial-gradient(1px 1px at 64% 32%,#fff6 50%,transparent 51%),
  radial-gradient(1px 1px at 82% 72%,#fff7 50%,transparent 51%),radial-gradient(1px 1px at 32% 82%,#fff5 50%,transparent 51%),
  radial-gradient(1.5px 1.5px at 48% 8%,#fff9 50%,transparent 51%),radial-gradient(1px 1px at 92% 44%,#fff5 50%,transparent 51%);
  background-size:420px 420px}
html.retro .page{position:relative;z-index:1}
html.retro ::selection{background:${p.accent};color:${p.onAccent}}
html.retro a{color:${p.accent}}
html.retro a.back{color:${p.dim}} html.retro a.back:hover{color:${p.accent}}
html.retro h1{font-family:${PIX};font-weight:400;font-size:clamp(15px,2.6vw,22px);line-height:1.45;color:${p.title};
  letter-spacing:.02em;text-shadow:0 0 18px ${p.titleGlow},0 2px 0 rgba(0,0,0,.6);margin:10px 0 8px}
html.retro h1 span{font-family:${BODY};font-size:15px;color:${p.dim};text-shadow:none;letter-spacing:0}
html.retro .sub{color:${p.dim}}
html.retro .sub b,html.retro .hint b{color:${p.ink}}
html.retro h2,html.retro h3{color:${p.ink}}
html.retro .card{background:${p.panel};border:1px solid ${p.line};border-radius:12px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 10px 30px rgba(0,0,0,.35);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px)}
html.retro .card h2,html.retro .card>h2,html.retro .card>summary>h2{font-family:${PIX};font-weight:400;font-size:9.5px;
  letter-spacing:.08em;line-height:1.6;color:${p.accent};text-shadow:0 0 10px ${p.accent}55}
html.retro button,html.retro .btn{font-family:${BODY};background:rgba(255,255,255,.035);color:${p.ink};border:1px solid ${p.lineHi};border-radius:8px}
html.retro button:hover,html.retro .btn:hover{border-color:${p.accent};color:${p.accent};box-shadow:0 0 0 1px ${p.accent}33,0 0 14px ${p.accent}33}
html.retro button.primary,html.retro .btn.primary{background:${p.accent};border-color:${p.accent};color:${p.onAccent};font-weight:700;
  box-shadow:0 0 16px ${p.accent}55}
html.retro button.primary:hover{background:${p.accent2};color:${p.onAccent}}
html.retro button:focus-visible{outline:2px solid ${p.gold};outline-offset:2px}
html.retro input[type=number],html.retro input[type=text],html.retro select,html.retro textarea{background:${p.input};color:${p.ink};
  border:1px solid ${p.lineHi};font-family:${BODY}}
html.retro select option{background:${p.bg2};color:${p.ink}}
html.retro input:focus,html.retro select:focus,html.retro textarea:focus{outline:2px solid ${p.accent}44;border-color:${p.accent}}
html.retro input[type=range],html.retro input[type=checkbox],html.retro input[type=radio]{accent-color:${p.accent}}
html.retro .code{background:${p.input};border-color:${p.lineHi};color:${p.gold};font-family:${PIX};font-size:14px}
html.retro table th,html.retro table td{border-color:${p.line}}
html.retro .eq{color:${p.ink}}
html.retro hr{border-color:${p.line}}
html.retro details>summary{color:${p.ink}}
html.retro kbd{background:${p.input};border:1px solid ${p.lineHi};color:${p.ink};border-radius:4px;padding:0 4px}
html.retro .tabs{border-color:${p.line}}
html.retro .tab{color:${p.dim}}
html.retro .tab[aria-selected="true"],html.retro .tab.on,html.retro .tab.active{background:${p.panel};color:${p.accent};border-color:${p.line}}
html.retro .seg{border-color:${p.lineHi}}
html.retro .seg button{border-color:${p.line};background:transparent}
html.retro .seg button[aria-pressed="true"]{background:${p.accent};color:${p.onAccent}}
html.retro .stage{background:${p.bg};border:1px solid ${p.lineHi};box-shadow:0 0 0 1px rgba(0,0,0,.4),0 0 28px ${p.accent}22,0 14px 40px rgba(0,0,0,.55)}
html.retro .overlay{background:${p.bg}e6}
html.retro .overlay h2{font-family:${PIX};font-weight:400;font-size:clamp(16px,3vw,24px);line-height:1.4;color:${p.title};text-shadow:0 0 18px ${p.titleGlow}}
html.retro .overlay h2 small{font-family:${BODY};text-shadow:none;color:${p.dim}}
html.retro .js-plotly-plot .plotly .modebar-btn path{fill:${p.dim}}
html.retro .js-plotly-plot .plotly .modebar-btn:hover path,html.retro .js-plotly-plot .plotly .modebar-btn.active path{fill:${p.accent}}
html.retro .js-plotly-plot .plotly .modebar{background:transparent}
/* the shared site bar */
html.retro .qsitebar{font-family:${BODY};color:${p.dim};border-bottom-color:${p.line}}
html.retro .qsitebar a{color:${p.accent}}
html.retro .qsitebar a.full{background:${p.accent};color:${p.onAccent}}
html.retro .qsitebar .rt-theme{font:600 11.5px ${BODY};padding:2px 9px;border-radius:999px;border:1px solid ${p.lineHi};
  background:transparent;color:${p.dim};cursor:pointer}
html.retro .qsitebar .rt-theme:hover{color:${p.accent};border-color:${p.accent}}
/* the outreach kit: learning panel, question dialogs, offer toast, privacy line */
html.retro .ql-panel{background:${p.panel};border-color:${p.line};color:${p.ink};font-family:${BODY}}
html.retro .ql-panel h3{font-family:${PIX};font-weight:400;font-size:11px;line-height:1.6;color:${p.gold}}
html.retro .ql-panel h4,html.retro .ql-panel .ql-tag,html.retro .ql-panel .ql-fun,html.retro .ql-note{color:${p.dim}}
html.retro .ql-btn{font-family:${BODY};background:${p.accent};border-color:${p.accent};color:${p.onAccent}}
html.retro .ql-btn.ql-ghost{background:transparent;color:${p.accent}}
html.retro .ql-veil{background:rgba(2,4,10,.72)}
html.retro .ql-dlg,html.retro .ql-offer{background:${p.bg2};color:${p.ink};border:1px solid ${p.lineHi};font-family:${BODY};
  box-shadow:0 0 30px ${p.accent}22,0 16px 50px rgba(0,0,0,.6)}
html.retro .ql-dlg h2{font-family:${PIX};font-weight:400;font-size:13px;line-height:1.6;color:${p.gold}}
html.retro .ql-dlg .ql-sub{color:${p.dim}}
html.retro .ql-q{border-color:${p.line}}
html.retro .ql-q label:hover{background:rgba(255,255,255,.05)}
html.retro .ql-q .ql-why.ok{background:${p.green}1f} html.retro .ql-q .ql-why.no{background:${p.red}1f}
html.retro .ql-rate button{background:transparent;color:${p.ink};border-color:${p.lineHi};font-family:${BODY}}
html.retro .ql-rate button[aria-pressed="true"]{background:${p.accent};color:${p.onAccent};border-color:${p.accent}}
html.retro .ql-privacy{color:${p.dim};font-family:${BODY}} html.retro .ql-privacy button{color:${p.accent}}
/* PageKit guided tours */
html.retro .gd-panel{background:${p.panel};border-color:${p.accent};box-shadow:0 0 22px ${p.accent}22}
html.retro .gd-head b{font-family:${PIX};font-weight:400;font-size:9.5px;color:${p.accent}}
html.retro .gd-choices button,html.retro .gd-panel .gd-btn{background:transparent;color:${p.ink};border-color:${p.lineHi}}
html.retro .gd-choices button[aria-pressed=true],html.retro .gd-choices button.right{background:${p.green}22;border-color:${p.green}}
html.retro .gd-choices button.wrong{background:${p.red}22;border-color:${p.red}}
html.retro .gd-panel .gd-btn.primary{background:${p.accent};color:${p.onAccent};border-color:${p.accent}}
html.retro .gd-num input{background:${p.input};color:${p.ink};border-color:${p.lineHi}}
html.retro .gd-fb{background:${p.panel2}}
html.retro .gd-fb .verdict.ok{color:${p.green}} html.retro .gd-fb .verdict.no{color:${p.gold}}
${theme === 'phosphor' ? `html.retro body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:9;
  background:repeating-linear-gradient(0deg,rgba(0,0,0,.16) 0 1px,transparent 1px 3px);mix-blend-mode:multiply}
html.retro h1,html.retro .card h2{animation:rt-flicker 6s infinite steps(1)}
@keyframes rt-flicker{0%,96%,100%{opacity:1}97%{opacity:.82}98%{opacity:1}99%{opacity:.9}}
@media (prefers-reduced-motion:reduce){html.retro h1,html.retro .card h2{animation:none}}` : ''}
${theme === 'synthwave' ? `html.retro body{background:
  radial-gradient(900px 420px at 50% 115%,#ff7a4555 0%,transparent 60%),
  radial-gradient(1200px 800px at 15% -10%,${p.glowA} 0%,transparent 55%),
  radial-gradient(1100px 900px at 110% 8%,${p.glowB} 0%,transparent 55%),
  linear-gradient(180deg,${p.bg} 0%,${p.bg2} 70%,#2a0a3a 100%);background-attachment:fixed}
html.retro body::before{opacity:.35;background-image:
  linear-gradient(transparent 0 calc(100% - 1px),#ff4fd833 calc(100% - 1px)),linear-gradient(90deg,transparent 0 calc(100% - 1px),#ff4fd822 calc(100% - 1px));
  background-size:48px 48px;-webkit-mask-image:linear-gradient(to top,#000 0,transparent 45%);mask-image:linear-gradient(to top,#000 0,transparent 45%)}` : ''}`;
  }

  // ── Plotly: dark layouts and neon ink ───────────────────────────────────────────────────────────────────────────
  // Light-theme colours the pages use, mapped to readable equivalents on a dark background. Anything else is kept.
  function inkMap(p) {
    return {
      '#00693e': p.accent, '#12855a': p.accent2, '#1b2a24': p.ink, '#5a6b63': p.dim, '#222': p.ink, '#333': p.ink,
      '#444': p.dim, '#555': p.dim, '#666': p.dim, '#000': p.ink, '#000000': p.ink, 'black': p.ink, '#111': p.ink,
      '#2d6fb3': p.blue, '#1f77b4': p.blue, '#c84b31': p.red, '#d9a521': p.gold, '#d8e2dc': p.line, '#ccc': p.line,
      '#e5e5e5': p.line, '#eee': p.line, '#ddd': p.line, '#888': p.dim, '#999': p.dim, '#aaa': p.dim,
      '#7a4b12': p.gold, 'purple': '#c77dff', '#800080': '#c77dff', 'gray': '#9aa3b8', 'grey': '#9aa3b8', '#1f4a73': p.blue, '#b5531e': p.rose,
    };
  }
  const isColorKey = k => /colou?r$|^(bgcolor|bordercolor|gridcolor|zerolinecolor|linecolor|tickcolor|fillcolor|outlinecolor)$/i.test(k);
  const LIGHT_BG = /^(#fff(fff)?|white|#f[0-9a-f]{2}([0-9a-f]{3})?|rgba?\(\s*2[3-5]\d\s*,\s*2[3-5]\d\s*,\s*2[3-5]\d\b.*)$/i;

  function recolor(v, map, key, seen) {
    if (v == null || typeof v !== 'object' || seen.has(v)) return;
    seen.add(v);
    for (const k of Object.keys(v)) {
      const x = v[k];
      if (typeof x === 'string' && (isColorKey(k) || key === 'color')) {
        const m = map[x.trim().toLowerCase()];
        if (m) v[k] = m;
        else if (/bgcolor$/i.test(k) && LIGHT_BG.test(x.trim())) v[k] = 'rgba(0,0,0,0)';
      } else if (Array.isArray(x) && isColorKey(k)) {
        v[k] = x.map(c => (typeof c === 'string' && map[c.trim().toLowerCase()]) || c);
      } else if (x && typeof x === 'object' && !ArrayBuffer.isView(x) && !(Array.isArray(x) && typeof x[0] === 'number')) {
        recolor(x, map, k, seen);
      }
    }
  }
  function axisDefaults(a, p) {
    a.gridcolor = a.gridcolor && !LIGHT_BG.test(a.gridcolor) && !/^#(e|d|c)/i.test(a.gridcolor) ? a.gridcolor : 'rgba(124,140,230,0.14)';
    if (a.zerolinecolor == null || /^#(e|d|c|f)/i.test(a.zerolinecolor)) a.zerolinecolor = 'rgba(124,140,230,0.32)';
    if (a.linecolor == null || /^#(0|1|2|3|4|5|6)/i.test(a.linecolor) || a.linecolor === 'black') a.linecolor = p.line;
    a.tickfont = Object.assign({ color: p.dim }, a.tickfont); if (a.tickfont.color && inkMap(p)[a.tickfont.color]) a.tickfont.color = p.dim;
  }
  function darkLayout(layout, p) {
    const L = layout || {};
    recolor(L, inkMap(p), '', new WeakSet());
    L.paper_bgcolor = 'rgba(0,0,0,0)';
    if (!L.plot_bgcolor || LIGHT_BG.test(L.plot_bgcolor)) L.plot_bgcolor = 'rgba(4,6,16,0.45)';
    L.font = Object.assign({}, L.font, { family: BODY, color: (L.font && L.font.color && !inkMap(p)[L.font.color]) ? L.font.color : p.ink });
    for (const k of Object.keys(L)) if (/^(x|y)axis\d*$/.test(k) && L[k] && typeof L[k] === 'object') axisDefaults(L[k], p);
    if (!Object.keys(L).some(k => /^(x|y)axis/.test(k))) { L.xaxis = L.xaxis || {}; L.yaxis = L.yaxis || {}; axisDefaults(L.xaxis, p); axisDefaults(L.yaxis, p); }
    if (L.scene) for (const ax of ['xaxis', 'yaxis', 'zaxis']) { L.scene[ax] = L.scene[ax] || {}; axisDefaults(L.scene[ax], p); L.scene[ax].backgroundcolor = 'rgba(0,0,0,0)'; }
    if (L.legend) { L.legend.bgcolor = 'rgba(0,0,0,0)'; L.legend.font = Object.assign({ color: p.ink }, L.legend.font); }
    if (L.hoverlabel == null) L.hoverlabel = { bgcolor: p.bg2, bordercolor: p.accent, font: { color: p.ink, family: BODY } };
    if (L.polar) { L.polar.bgcolor = 'rgba(0,0,0,0)'; }
    return L;
  }
  function wrapPlotly(Pl) {
    if (!Pl || Pl.__retro || theme === 'classic') return Pl;
    const p = P[theme], map = inkMap(p);
    const fixData = d => { if (Array.isArray(d)) recolor({ d }, map, '', new WeakSet()); return d; };
    const fig = (fn) => function (gd, data, layout, config) {
      if (data && !Array.isArray(data) && typeof data === 'object' && data.data) {   // newPlot(gd, {data, layout, config})
        fixData(data.data); data.layout = darkLayout(data.layout, p); return fn.call(this, gd, data);
      }
      return fn.call(this, gd, fixData(data), darkLayout(layout, p), config);
    };
    if (Pl.newPlot) Pl.newPlot = fig(Pl.newPlot);
    if (Pl.react) Pl.react = fig(Pl.react);
    const fixObj = o => { if (o && typeof o === 'object') recolor(o, map, '', new WeakSet()); return o; };
    if (Pl.relayout) { const f = Pl.relayout; Pl.relayout = function (gd, a, b) { if (typeof a === 'object') fixObj(a); return f.call(this, gd, a, b); }; }
    if (Pl.restyle) { const f = Pl.restyle; Pl.restyle = function (gd, a, b, c) { if (typeof a === 'object') fixObj(a); return f.call(this, gd, a, b, c); }; }
    if (Pl.update) { const f = Pl.update; Pl.update = function (gd, a, b, c) { fixObj(a); fixObj(b); return f.call(this, gd, a, b, c); }; }
    Pl.__retro = true;
    return Pl;
  }

  // ── switch on ──────────────────────────────────────────────────────────────────────────────────────────────────
  if (theme !== 'classic') {
    html.classList.add('retro', 'retro-' + theme);
    const st = doc.createElement('style'); st.id = 'retro-css'; st.textContent = css(P[theme]);
    (doc.head || html).appendChild(st);
    const meta = doc.createElement('meta'); meta.name = 'theme-color'; meta.content = P[theme].bg; (doc.head || html).appendChild(meta);
    // Plotly may already be here, or arrive later (its UMD build assigns window.Plotly)
    let pl = root.Plotly;
    if (pl) wrapPlotly(pl);
    else {
      try {
        Object.defineProperty(root, 'Plotly', { configurable: true, enumerable: true,
          get() { return pl; }, set(v) { pl = wrapPlotly(v); } });
      } catch (e) { /* non-configurable: charts stay light */ }
    }
  }

  // ── page stylesheets: turn the light theme's hard-coded colours dark ───────────────────────────────────────────
  // Many pages write #fff or pale tints straight into their rules. Rather than patch every page, rewrite those
  // declarations in the CSSOM: near-white backgrounds become the theme's panels, pale tints become faint glows of the
  // same hue, dark text becomes light, and pale borders become theme lines. Saturated colours (gold chips, the
  // game's own neon) are left alone, and so is the text on them.
  function rgbOf(s) {
    s = (s || '').trim().toLowerCase();
    let m = s.match(/^#([0-9a-f]{3,8})$/);
    if (m) {
      let h = m[1]; if (h.length <= 4) h = h.split('').map(c => c + c).join('');
      return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16), a: h.length === 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1 };
    }
    m = s.match(/^rgba?\(\s*([\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/);
    if (m) return { r: +m[1], g: +m[2], b: +m[3], a: m[4] == null ? 1 : (m[4].endsWith('%') ? parseFloat(m[4]) / 100 : +m[4]) };
    if (s === 'white') return { r: 255, g: 255, b: 255, a: 1 };
    if (s === 'black') return { r: 0, g: 0, b: 0, a: 1 };
    return null;
  }
  function hsl({ r, g, b }) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn;
    let h = 0, s = 0;
    if (d) {
      s = d / (1 - Math.abs(2 * l - 1));
      h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; h *= 60; if (h < 0) h += 360;
    }
    return { h, s, l };
  }
  function darkenSheet(sheet, p) {
    let rules; try { rules = sheet.cssRules; } catch (e) { return; }   // cross-origin
    for (const r of rules) {
      if (r.cssRules && !r.style) { darkenSheet(r, p); continue; }      // @media and friends
      if (!r.style || /html\.retro/.test(r.selectorText || '')) continue;
      fixStyle(r.style, p);
    }
  }
  function fixStyle(st, p) {
    const bgProp = st.getPropertyValue('background-color') ? 'background-color' : (st.getPropertyValue('background') ? 'background' : '');
    const bgVal = bgProp ? st.getPropertyValue(bgProp).trim() : '';
    const bg = rgbOf(bgVal);
    let brightChip = false;
    if (bg && bg.a > 0.5) {
      const c = hsl(bg);
      if (c.l > 0.9 && c.s < 0.25) st.setProperty(bgProp, p.panel2, st.getPropertyPriority(bgProp));          // white-ish
      else if (c.l > 0.86) st.setProperty(bgProp, `hsla(${c.h.toFixed(0)},70%,60%,0.13)`, st.getPropertyPriority(bgProp)); // pale tint
      else if (c.l > 0.45) brightChip = true;                                                                    // a coloured chip
    }
    const col = rgbOf(st.getPropertyValue('color'));
    if (col && !brightChip) {
      const c = hsl(col);
      if (c.l < 0.3 && c.s < 0.25) st.setProperty('color', p.ink, st.getPropertyPriority('color'));
      else if (c.l < 0.4) st.setProperty('color', `hsl(${c.h.toFixed(0)},${Math.max(60, c.s * 100).toFixed(0)}%,74%)`, st.getPropertyPriority('color'));
      else if (c.l < 0.5 && c.s < 0.2) st.setProperty('color', p.dim, st.getPropertyPriority('color'));
    }
    for (const side of ['border-color', 'border-top-color', 'border-bottom-color', 'border-left-color', 'border-right-color']) {
      const bc = rgbOf(st.getPropertyValue(side));
      if (!bc) continue;
      const c = hsl(bc);
      if (c.l > 0.78) st.setProperty(side, c.s < 0.25 ? p.line : `hsla(${c.h.toFixed(0)},70%,62%,0.4)`, st.getPropertyPriority(side));
    }
  }
  if (theme !== 'classic') {
    const done = new WeakSet();
    const sweep = () => {
      for (const s of doc.styleSheets) {
        const owner = s.ownerNode;
        if (done.has(s) || (owner && owner.id === 'retro-css')) continue;
        done.add(s); darkenSheet(s, P[theme]);
      }
    };
    // inline style="" attributes written into the markup or by scripts
    const sweepInline = rootEl => {
      for (const el of (rootEl.querySelectorAll ? rootEl.querySelectorAll('[style]') : [])) {
        if (el.closest('svg') || el.dataset.rtDone) continue;
        el.dataset.rtDone = '1'; fixStyle(el.style, P[theme]);
      }
    };
    const kick = () => { sweep(); sweepInline(doc); };
    if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', kick); else kick();
    root.addEventListener('load', kick);
    try {
      new MutationObserver(ms => {
        let styles = false;
        for (const m of ms) for (const n of m.addedNodes) {
          if (n.nodeType !== 1) continue;
          if (n.tagName === 'STYLE' || n.tagName === 'LINK') styles = true;
          else if (!n.closest || !n.closest('.js-plotly-plot')) { if (n.getAttribute('style')) { n.removeAttribute('data-rt-done'); fixStyle(n.style, P[theme]); } sweepInline(n); }
        }
        if (styles) sweep();
      }).observe(html, { childList: true, subtree: true });
    } catch (e) { /* old browser: the static sweep is enough */ }
  }

  // ── light-theme canvases (opt-in) ────────────────────────────────────────────────────────────────────────────
  // A <canvas data-retro-ink> drawn with the light theme's colours gets them remapped as they are set: pale plot
  // backgrounds turn dark, pale gridlines become theme lines, dark text and axes turn light, and the Dartmouth
  // series colours become their neon versions. Other colours, and canvases without the attribute, are untouched.
  if (theme !== 'classic' && root.CanvasRenderingContext2D) {
    const p = P[theme], map = inkMap(p);
    const extra = { '#fbfdfc': 'rgba(4,6,16,0.55)', '#e4ede8': 'rgba(124,140,230,0.22)', '#b9c7c0': p.dim, '#9fb0a8': p.dim,
      'rgba(27,42,36,0.08)': 'rgba(124,140,230,0.10)', 'rgba(0,105,62,0.15)': p.accent + '33', 'rgba(200,75,49,0.12)': p.red + '2a' };
    const remap = (v, fill) => {
      if (typeof v !== 'string') return v;
      const k = v.trim().toLowerCase().replace(/\s+/g, '');
      if (k === '#fff' || k === '#ffffff' || k === 'white') return v;   // pure white is a highlight: keep it
      if (extra[k]) return extra[k];
      if (map[k]) return map[k];
      const c = rgbOf(k); if (!c || c.a < 0.3) return v;
      const h = hsl(c);
      if (fill && h.l > 0.93 && h.s < 0.3) return 'rgba(4,6,16,0.55)';
      if (!fill && h.l > 0.85 && h.s < 0.3) return 'rgba(124,140,230,0.22)';
      if (h.l < 0.35 && h.s < 0.3) return p.ink;
      return v;
    };
    const proto = root.CanvasRenderingContext2D.prototype;
    for (const [prop, fill] of [['fillStyle', true], ['strokeStyle', false]]) {
      const d = Object.getOwnPropertyDescriptor(proto, prop);
      if (!d || !d.set) continue;
      Object.defineProperty(proto, prop, { configurable: true, enumerable: d.enumerable, get: d.get,
        set(v) { d.set.call(this, this.canvas && this.canvas.hasAttribute && this.canvas.hasAttribute('data-retro-ink') ? remap(v, fill) : v); } });
    }
  }

  // the theme switch, added to the site bar once it exists
  function addSwitch() {
    const bar = doc.querySelector('.qsitebar');
    if (!bar || bar.querySelector('.rt-theme')) return;
    const b = doc.createElement('button');
    b.type = 'button'; b.className = 'rt-theme';
    b.title = 'Change the look (Arena, Synthwave, Phosphor, Classic). Remembered in this browser only.';
    b.textContent = '◐ ' + LABEL[theme];
    if (theme === 'classic') b.style.cssText = 'font:600 11.5px Helvetica,Arial,sans-serif;padding:2px 9px;border-radius:999px;border:1px solid #d8e2dc;background:#fff;color:#5a6b63;cursor:pointer';
    b.addEventListener('click', () => {
      const next = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
      try { root.localStorage.setItem(KEY, next); } catch (e) { /* not remembered */ }
      const u = new URL(root.location.href); u.searchParams.delete('theme');
      if (u.href === root.location.href) root.location.reload(); else root.location.replace(u.href);
    });
    const sp = bar.querySelector('.sp');
    if (sp && sp.nextSibling) bar.insertBefore(b, sp.nextSibling); else bar.appendChild(b);
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', addSwitch); else addSwitch();

  root.RETRO = { theme, themes: THEMES.slice(), palette: P[theme] || null };
})(window);
