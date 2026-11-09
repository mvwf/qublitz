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
 * Themes: arena (default), synthwave, phosphor, classic (the original light Dartmouth look, handy for projectors)
 * and dark (Classic dark: plain neutral dark, no retro fonts). Choose with ?theme=<name> (light is an alias of
 * classic, as in Q-Vault), or with the theme menu in the site bar; the choice is remembered in this browser only and
 * is shared with Q-Vault on the same origin.
 * Pages can tweak themselves with selectors under html.retro (e.g. html.retro .myThing{...}).
 */
(function (root) {
  'use strict';
  const doc = root.document, html = doc.documentElement;
  const THEMES = ['arena', 'synthwave', 'phosphor', 'classic', 'dark'];
  const LABEL = { arena: 'Arena', synthwave: 'Synthwave', phosphor: 'Phosphor', classic: 'Classic light', dark: 'Classic dark' };
  const BLURB = { arena: 'Neon on deep navy, pixel titles', synthwave: 'Magenta and sunset grid', phosphor: 'Green CRT with scanlines',
    classic: 'The original light look, good for projectors', dark: 'Plain dark, no retro fonts' };
  const KEY = 'qublitz_theme', QVAULT_KEY = 'qvault-theme';   // Q-Vault (mvwf.github.io/MARQUIS_Q_VAULT) shares the origin
  const ALIAS = { light: 'classic' };                         // Q-Vault's name for Classic light
  const pick = v => (THEMES.includes(v) ? v : ALIAS[v] || null);
  const read = k => { try { return root.localStorage.getItem(k); } catch (e) { return null; } };

  // Order of precedence: ?theme= in the link, then this site's saved choice, then a choice made in Q-Vault, then Arena.
  let theme = pick(read(KEY)) || pick(read(QVAULT_KEY)) || 'arena';
  try { const q = pick(new URLSearchParams(root.location.search).get('theme')); if (q) theme = q; } catch (e) { /* no URL */ }

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
    // Classic dark: Q-Vault's neutral dark surfaces with the Classic look's green accent; no pixel fonts, glows or stars.
    dark: {
      plain: true, bg: '#121211', bg2: '#1b1b1a', glowA: '#121211', glowB: '#121211',
      ink: '#f1f0ec', dim: '#bab9af', accent: '#4cc38a', accent2: '#6fd3a2', gold: '#e0b44a', rose: '#e8699b',
      indigo: '#9d8cf0', red: '#ef7070', blue: '#5d9cec', green: '#4cc38a',
      panel: '#1b1b1a', panel2: '#242422', line: '#33332f', lineHi: '#4a4a45',
      input: '#121211', title: '#4cc38a', titleGlow: 'transparent', onAccent: '#0b0b0b',
    },
  };

  const FONTS = (root.RETRO_FONTS || []).map(f =>
    `@font-face{font-family:'${f.family}';font-style:normal;font-weight:${f.weight};font-display:swap;` +
    `src:url(data:font/woff2;base64,${f.woff2}) format('woff2')}`).join('\n');
  const HELV = `Helvetica,'Helvetica Neue',Arial,sans-serif`;
  const PIX0 = `'Press Start 2P',ui-monospace,monospace`, BODY0 = `'Space Grotesk',system-ui,-apple-system,'Segoe UI',sans-serif`;

  function css(p) {
    const PIX = p.plain ? HELV : PIX0, BODY = p.plain ? HELV : BODY0;
    return `${FONTS}
html.retro{color-scheme:dark;
  --green:${p.accent};--on-green:${p.onAccent};--green2:${p.accent2};--ink:${p.ink};--muted:${p.dim};--line:${p.line};--panel:${p.panel2};
  --night:${p.bg};--accent:${p.red};--blue:${p.blue};--gold:${p.gold};--int:${p.red};--ext:${p.blue};--heat:${p.red};--wave:${p.blue};--sch:${p.accent};
  --cx:#ff5fa8;--cy:#40e6b4;--cz:#8fb0ff;--dx:#ff5fa8;--dy:#40e6b4;--dz:#8fb0ff;
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
/* card headings: the body font (the pixel font at 9.5-11 px was hard to read), uppercase and letter-spaced like the original */
html.retro .card h2,html.retro .card>h2,html.retro .card>summary>h2{font-family:${BODY};font-weight:700;font-size:14px;
  letter-spacing:.07em;text-transform:uppercase;line-height:1.45;color:${p.accent};text-shadow:0 0 10px ${p.accent}55}
html.retro button,html.retro .btn{font-family:${BODY};background:rgba(255,255,255,.035);color:${p.ink};border:1px solid ${p.lineHi};border-radius:8px}
/* hover glow, except on buttons that are already a bright fill (gold Start/Fire, held pads, pressed chips) */
html.retro button:not(.go):not(.fire):not(.on):not([aria-pressed="true"]):not([aria-selected="true"]):not(.primary):hover,html.retro .btn:not(.primary):hover{border-color:${p.accent};color:${p.accent};box-shadow:0 0 0 1px ${p.accent}33,0 0 14px ${p.accent}33}
html.retro button.primary,html.retro .btn.primary{background:${p.accent};border-color:${p.accent};color:${p.onAccent};font-weight:700;
  box-shadow:0 0 16px ${p.accent}55}
html.retro button.primary:hover{background:${p.accent2};color:${p.onAccent}}
html.retro button:focus-visible,html.retro a:focus-visible,html.retro summary:focus-visible,html.retro [tabindex]:focus-visible,
html.retro input:focus-visible,html.retro select:focus-visible,html.retro textarea:focus-visible{outline:2px solid ${p.gold};outline-offset:2px}
html.retro input[type=number],html.retro input[type=text],html.retro select,html.retro textarea{background:${p.input};color:${p.ink};
  border:1px solid ${p.lineHi};font-family:${BODY}}
html.retro select option{background:${p.bg2};color:${p.ink}}
html.retro input:focus,html.retro select:focus,html.retro textarea:focus{border-color:${p.accent}}
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
/* small labels on bright fills: dark text (white on cyan/sky/red fails contrast) */
html.retro .tab[aria-selected="true"] .n,html.retro .regime .tag,html.retro .budget i,html.retro .chip[aria-pressed="true"],
html.retro .ctrl button.on,html.retro .ctrl button.on small,html.retro .segbar div{color:${p.onAccent}}
html.retro .chip[aria-pressed="true"] .n{opacity:.8}
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
html.retro .qfund{font-family:${BODY};color:${p.dim};border-top-color:${p.line}}
html.retro .qfund a{color:${p.accent}}
html.retro .qsitebar a.qv{background:${p.accent}1f}
html.retro .qv-real{background:${p.accent}14;border-color:${p.line}}
html.retro .qv-real a{color:${p.accent}}
/* the outreach kit: learning panel, question dialogs, offer toast, privacy line */
html.retro .ql-panel{background:${p.panel};border-color:${p.line};color:${p.ink};font-family:${BODY}}
html.retro .ql-panel .ql-h{font-family:${BODY};font-weight:700;font-size:16px;line-height:1.45;color:${p.gold};text-transform:none;letter-spacing:normal;text-shadow:none;animation:none}
html.retro .ql-panel .ql-sh,html.retro .ql-panel .ql-tag,html.retro .ql-panel .ql-fun,html.retro .ql-note{color:${p.dim}}
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
html.retro .ql-ch{background:${p.bg2};color:${p.ink};border-color:${p.accent};font-family:${BODY};box-shadow:0 0 22px ${p.accent}22,0 10px 30px rgba(0,0,0,.5)}
html.retro .ql-ch>button.ql-ch-head{color:${p.gold}} html.retro .ql-ch-bar{background:${p.line}} html.retro .ql-ch-bar i{background:${p.accent}}
html.retro .ql-ch li{border-color:${p.line}} html.retro .ql-ch li .ql-ck{border-color:${p.dim};color:${p.onAccent}}
html.retro .ql-ch li.done .ql-ck{background:${p.green};border-color:${p.green}} html.retro .ql-ch li.done .ql-ct{color:${p.dim}}
html.retro .ql-ch .ql-hint{background:${p.gold}22;color:${p.ink}} html.retro .ql-ch .ql-ch-foot{color:${p.dim}} html.retro .ql-ch .ql-ch-foot button{color:${p.accent}}
/* PageKit guided tours */
html.retro .gd-panel{background:${p.panel};border-color:${p.accent};box-shadow:0 0 22px ${p.accent}22}
html.retro .gd-head b{font-family:${BODY};font-weight:700;font-size:14px;color:${p.accent}}
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
  background-size:48px 48px;-webkit-mask-image:linear-gradient(to top,#000 0,transparent 45%);mask-image:linear-gradient(to top,#000 0,transparent 45%)}` : ''}
${p.plain ? `html.retro body::before{display:none}
html.retro *{text-shadow:none!important}
html.retro body{background:${p.bg}}
html.retro h1{font-weight:700;font-size:24px;letter-spacing:.01em;line-height:1.3;margin:4px 0 2px}
html.retro .card h2,html.retro .card>h2,html.retro .card>summary>h2{font-size:13px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:${p.dim}}
html.retro .overlay h2{font-weight:700} html.retro .card{backdrop-filter:none;-webkit-backdrop-filter:none;box-shadow:none}
html.retro .stage{box-shadow:0 8px 30px rgba(0,0,0,.4)} html.retro button.primary{box-shadow:none}` : ''}`;
  }

  // ── Plotly: dark layouts and neon ink ───────────────────────────────────────────────────────────────────────────
  // Light-theme colours the pages use, mapped to readable equivalents on a dark background. Anything else is kept.
  function inkMap(p) {
    return {
      '#00693e': p.accent, '#12855a': p.accent2, '#1b2a24': p.ink, '#5a6b63': p.dim, '#222': p.ink, '#333': p.ink,
      '#444': p.dim, '#555': p.dim, '#666': p.dim, '#000': p.ink, '#000000': p.ink, 'black': p.ink, '#111': p.ink,
      '#2d6fb3': p.blue, '#1f77b4': p.blue, '#c84b31': p.red, '#d9a521': p.gold, '#d8e2dc': p.line, '#ccc': p.line,
      '#e5e5e5': p.line, '#eee': p.line, '#ddd': p.line, '#888': p.dim, '#999': p.dim, '#aaa': p.dim,
      '#7a4b12': p.gold, '#7d8f86': p.blue, '#c3cec8': 'rgba(154,163,207,0.40)', '#c9d6cf': p.lineHi,
      '#9fb0a8': p.ink, '#c9a26b': p.gold, '#a58fc0': '#c77dff',
      'magenta': p.rose, 'cyan': p.accent, 'chartreuse': p.gold, 'red': p.red, '#ff0000': p.red, '#f00': p.red, 'purple': '#c77dff', '#800080': '#c77dff', 'gray': '#9aa3b8', 'grey': '#9aa3b8', '#1f4a73': p.blue, '#b5531e': p.rose,
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
    L.font = Object.assign({}, L.font, { family: p.plain ? HELV : BODY0, color: (L.font && L.font.color && !inkMap(p)[L.font.color]) ? L.font.color : p.ink });
    for (const k of Object.keys(L)) if (/^(x|y)axis\d*$/.test(k) && L[k] && typeof L[k] === 'object') axisDefaults(L[k], p);
    if (!Object.keys(L).some(k => /^(x|y)axis/.test(k))) { L.xaxis = L.xaxis || {}; L.yaxis = L.yaxis || {}; axisDefaults(L.xaxis, p); axisDefaults(L.yaxis, p); }
    if (L.scene) for (const ax of ['xaxis', 'yaxis', 'zaxis']) { L.scene[ax] = L.scene[ax] || {}; axisDefaults(L.scene[ax], p); L.scene[ax].backgroundcolor = 'rgba(0,0,0,0)'; }
    if (L.legend) { if (!L.legend.bgcolor || LIGHT_BG.test(L.legend.bgcolor)) L.legend.bgcolor = 'rgba(0,0,0,0)'; L.legend.font = Object.assign({ color: p.ink }, L.legend.font); }
    if (L.hoverlabel == null) L.hoverlabel = { bgcolor: p.bg2, bordercolor: p.accent, font: { color: p.ink, family: p.plain ? HELV : BODY0 } };
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
      // a bright chip whose text colour comes from elsewhere would get the theme's light ink: give it dark text
      if (brightChip && c.l > 0.6 && !st.getPropertyValue('color')) st.setProperty('color', '#14110a');
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

  // ── reading type (every theme, Classic included) ────────────────────────────────────────────────────────────────
  // 1. A reading scale: body 16 px, ledes (.sub) 17 px, card text 15 px, hints and notes 14 px.
  // 2. A floor on what actually renders: text under 13 px is raised to 13 px, and uppercase letter-spaced labels to
  //    12 px. Figures are sized to fit, so math (sub, sup, stacked .fr / .frac fractions), SVG, canvas, Plotly charts
  //    and game stages (.stage) are skipped. Content added later is floored too (a MutationObserver, batched per
  //    frame). Opt out per element, or for a whole subtree, with data-qt="keep". Floored elements get data-qt="floor".
  const typeCss = doc.createElement('style'); typeCss.id = 'retro-type-css';
  typeCss.textContent = `
html body { font-size: 16px; }
html .sub { font-size: 17px; line-height: 1.55; max-width: 68ch; }
html .card { font-size: 15px; }
html .hint, html .note { font-size: 14px; }
html .card th { white-space: normal; }   /* with the floor, a header row may need two lines in a narrow card */`;
  (doc.head || html).appendChild(typeCss);
  const QT_SKIP = 'svg,canvas,.js-plotly-plot,.stage,sub,sup,.fr,.frac,[data-qt="keep"]';
  const QT_FIELD = /^(SELECT|TEXTAREA)$/;
  const hasText = el => { for (let n = el.firstChild; n; n = n.nextSibling) if (n.nodeType === 3 && n.nodeValue.trim()) return true; return false; };
  function floorOne(el) {
    if (el.nodeType !== 1 || el.dataset.qt === 'floor' || /^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE|OPTION)$/.test(el.tagName)) return;
    const field = QT_FIELD.test(el.tagName) || (el.tagName === 'INPUT' && /^(text|number|search|url|email|tel|)$/.test(el.type || ''));
    if (!field && !hasText(el)) return;
    if (el.closest(QT_SKIP)) return;
    const s = root.getComputedStyle(el), px = parseFloat(s.fontSize);
    if (!(px > 0)) return;
    const min = s.textTransform === 'uppercase' && parseFloat(s.letterSpacing) > 0 ? 12 : 13;
    if (px < min - 0.01) { el.style.setProperty('font-size', min + 'px', 'important'); el.dataset.qt = 'floor'; }
  }
  function floorTree(rootEl) {
    if (!rootEl || rootEl.nodeType !== 1) return;
    floorOne(rootEl);
    for (const el of rootEl.querySelectorAll('*')) floorOne(el);
  }
  const floorAll = () => { if (doc.body) floorTree(doc.body); };
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', floorAll); else floorAll();
  root.addEventListener('load', floorAll);
  let qtTimer = 0;
  root.addEventListener('resize', () => { clearTimeout(qtTimer); qtTimer = setTimeout(floorAll, 250); });
  try {
    // new elements are checked with their subtree; a live readout whose text changes is checked once, not every frame
    const queued = new Set(), seen = new WeakSet(); let qtFrame = 0;
    const flush = () => { qtFrame = 0; for (const n of queued) if (n.isConnected) floorTree(n); queued.clear(); };
    new MutationObserver(ms => {
      for (const m of ms) for (const n of m.addedNodes) {
        if (n.nodeType === 1) queued.add(n);
        else if (n.nodeType === 3 && !seen.has(m.target)) { seen.add(m.target); queued.add(m.target); }
      }
      if (queued.size && !qtFrame) qtFrame = (root.requestAnimationFrame || setTimeout)(flush);
    }).observe(html, { childList: true, subtree: true });
  } catch (e) { /* old browser: the load-time pass is enough */ }

  // ── phones and tablets (every theme, Classic included) ──────────────────────────────────────────────────────────
  // 1. Finger-sized controls on touch screens (pointer: coarse): 40 px buttons and fields, taller sliders, bigger boxes.
  // 2. Plotly charts: drag-to-zoom would swallow the swipe that scrolls the page, so on touch screens 2D charts start in
  //    "no drag" mode (tap still shows values; the chart's toolbar still zooms). 3D scenes keep drag-to-rotate.
  // 3. On narrow screens a long intro paragraph under the title folds to three lines with a "More" button, so the
  //    activity itself is on the first screen.
  const TOUCH = !!(root.matchMedia && root.matchMedia('(pointer: coarse)').matches);
  const touchCss = doc.createElement('style'); touchCss.id = 'retro-touch-css';
  touchCss.textContent = `
@media (pointer: coarse) {
  button, .btn, select, input[type=number], input[type=text], input[type=search], input[type=url] { min-height: 40px; }
  .seg button, .tab, .chip { min-height: 40px; }
  input[type=range] { min-height: 32px; }
  input[type=checkbox], input[type=radio] { width: 24px; height: 24px; vertical-align: middle; }
  .qsitebar a, a.back, a#back { display: inline-block; padding: 8px 0; }   /* text links that stand alone get a finger-sized target */
  .rt-theme { height: 36px; }
  button { touch-action: manipulation; -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; }
  .pad button, .ctrl button, .stagebtn { touch-action: none; }   /* hold-to-drive buttons: a slight drift must not turn into a scroll */
  .js-plotly-plot .plotly .modebar-btn { padding: 6px 4px; }
}
.rt-clamp { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 3; overflow: hidden; }
.rt-more { display: inline-block; margin: 0 0 10px; font: 600 13px Helvetica, Arial, sans-serif; background: none; border: 0;
  padding: 6px 0; color: inherit; text-decoration: underline; cursor: pointer; min-height: 0; }
.rt-more[hidden] { display: none; }`;
  (doc.head || html).appendChild(touchCss);
  if (TOUCH) {
    const touchPlotly = Pl => {
      if (!Pl || Pl.__touch) return Pl;
      const fix = (layout, config) => {
        const L = layout || {};
        if (!Object.keys(L).some(k => /^scene\d*$/.test(k)) && L.dragmode == null) L.dragmode = false;
        return [L, Object.assign({ scrollZoom: false }, config || {})];
      };
      const wrap = fn => function (gd, data, layout, config) {
        if (data && !Array.isArray(data) && typeof data === 'object' && data.data) { const [L, C] = fix(data.layout, data.config); data.layout = L; data.config = C; return fn.call(this, gd, data); }
        const [L, C] = fix(layout, config); return fn.call(this, gd, data, L, C);
      };
      if (Pl.newPlot) Pl.newPlot = wrap(Pl.newPlot);
      if (Pl.react) Pl.react = wrap(Pl.react);
      Pl.__touch = true;
      return Pl;
    };
    // chain onto whatever setter retro already installed (or install one), so both wrappers apply in either order
    const d = Object.getOwnPropertyDescriptor(root, 'Plotly');
    if (root.Plotly) touchPlotly(root.Plotly);
    else if (d && d.set) Object.defineProperty(root, 'Plotly', { configurable: true, enumerable: true, get: d.get, set(v) { d.set(v); touchPlotly(d.get()); } });
    else { let pl; try { Object.defineProperty(root, 'Plotly', { configurable: true, enumerable: true, get() { return pl; }, set(v) { pl = touchPlotly(v); } }); } catch (e) { /* fine */ } }
  }
  function foldIntro() {
    if (!root.matchMedia || !root.matchMedia('(max-width: 640px)').matches) return;
    const h1 = doc.querySelector('h1'); if (!h1) return;
    let p = h1.nextElementSibling;
    while (p && !(p.tagName === 'P' && p.classList.contains('sub'))) { if (p.tagName !== 'P' && p.tagName !== 'SPAN') return; p = p.nextElementSibling; }
    if (!p || p.dataset.rtFold || p.textContent.trim().length < 260) return;
    p.dataset.rtFold = '1'; p.classList.add('rt-clamp');
    const b = doc.createElement('button'); b.type = 'button'; b.className = 'rt-more'; b.textContent = 'More ▾'; b.setAttribute('aria-expanded', 'false');
    b.addEventListener('click', () => { const open = p.classList.toggle('rt-clamp') === false; b.textContent = open ? 'Less ▴' : 'More ▾'; b.setAttribute('aria-expanded', String(open)); });
    p.after(b);
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', foldIntro); else foldIntro();

  // ── sideways-scrolling boxes reachable by keyboard (every theme) ─────────────────────────────────────────────────
  // A box with overflow-x auto/scroll that is wider inside than out, with nothing focusable in it, cannot be scrolled
  // from the keyboard. Such a box gets tabindex="0" (marked data-rt-scroll, so it is taken away again when the box no
  // longer overflows); a tabindex the page set itself is left alone. Rescanned on load, on resize and when elements
  // are added (tables filled after load; text-only updates do not count), at most once per 300 ms.
  const FOCUSABLE = 'a[href],button,input,select,textarea,iframe,summary,[contenteditable]:not([contenteditable="false"]),[tabindex]:not([tabindex="-1"])';
  function scanScrollers() {
    const body = doc.body; if (!body) return;
    for (const el of body.getElementsByTagName('*')) {
      const ours = el.hasAttribute('data-rt-scroll');
      if (!ours && el.hasAttribute('tabindex')) continue;
      let need = false;
      if (el.scrollWidth > el.clientWidth + 1) {
        const ox = root.getComputedStyle(el).overflowX;
        need = (ox === 'auto' || ox === 'scroll') && !el.querySelector(FOCUSABLE);
      }
      if (need && !ours) { el.setAttribute('tabindex', '0'); el.setAttribute('data-rt-scroll', ''); }
      else if (!need && ours) { el.removeAttribute('tabindex'); el.removeAttribute('data-rt-scroll'); }
    }
  }
  let scanTimer = 0, scanLast = 0;
  const scanSoon = () => {
    if (scanTimer) return;
    scanTimer = setTimeout(() => { scanTimer = 0; scanLast = Date.now(); scanScrollers(); }, Math.max(0, 300 - (Date.now() - scanLast)));
  };
  const startScan = () => {
    scanSoon();
    if (root.MutationObserver && doc.body) new MutationObserver(ms => { if (ms.some(m => [...m.addedNodes].some(n => n.nodeType === 1))) scanSoon(); }).observe(doc.body, { childList: true, subtree: true });
  };
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', startScan); else startScan();
  root.addEventListener('load', scanSoon);
  root.addEventListener('resize', scanSoon);

  // ── the theme switch: a button in the site bar (as in Q-Vault) that opens a menu of every theme ─────────────────
  const SW = {   // swatches for the menu: background, accent, title colour
    arena: ['#070a16', '#2fe0d0', '#f5b73c'], synthwave: ['#12051f', '#ff4fd8', '#42d6ff'], phosphor: ['#030a05', '#39ff7a', '#e8ff6a'],
    classic: ['#ffffff', '#00693e', '#d8e2dc'], dark: ['#121211', '#4cc38a', '#5d9cec'] };
  const cur = theme === 'classic' ? null : P[theme];
  const ui = doc.createElement('style'); ui.id = 'retro-switch-css';
  const c = cur || { ink: '#1b2a24', dim: '#5a6b63', accent: '#00693e', lineHi: '#c9d6cf', bg2: '#ffffff', line: '#d8e2dc' };
  ui.textContent = `
.rt-wrap{position:relative;display:inline-flex}
.rt-theme{display:inline-flex;align-items:center;gap:6px;height:28px;padding:0 11px;border-radius:999px;cursor:pointer;
  font:600 13px ${cur && !cur.plain ? BODY0 : HELV};border:1px solid ${c.lineHi};background:transparent;color:${c.dim};white-space:nowrap}
.rt-theme:hover,.rt-theme[aria-expanded="true"]{color:${c.accent};border-color:${c.accent}}
.rt-theme svg{width:14px;height:14px;flex:none}
.rt-menu{position:absolute;right:0;top:calc(100% + 6px);z-index:100000;min-width:250px;padding:6px;border-radius:12px;
  background:${c.bg2};border:1px solid ${c.lineHi};box-shadow:0 14px 40px rgba(0,0,0,.35);font:13px ${HELV}}
.rt-menu[hidden]{display:none}
.rt-menu p{margin:4px 8px 6px;font-size:13px;color:${c.dim}}
.rt-wrap .rt-menu button{display:flex;align-items:center;gap:10px;width:100%;text-align:left;padding:7px 8px;border:0!important;border-radius:8px;
  background:transparent!important;color:${c.ink}!important;cursor:pointer;font:inherit;box-shadow:none!important}
.rt-wrap .rt-menu button:hover,.rt-wrap .rt-menu button:focus-visible{background:${c.line}!important;outline:none}
.rt-menu button[aria-checked="true"]{font-weight:700}
.rt-menu button[aria-checked="true"]::after{content:"✓";margin-left:auto;color:${c.accent}}
.rt-menu .sw{display:inline-flex;flex:none;width:34px;height:20px;border-radius:5px;overflow:hidden;border:1px solid ${c.lineHi}}
.rt-menu .sw i{flex:1} .rt-menu small{display:block;font-size:13px;color:${c.dim};font-weight:400}
@media (max-width:560px){.rt-theme .rt-name{display:none}}`;
  (doc.head || html).appendChild(ui);

  function choose(t) {
    try { root.localStorage.setItem(KEY, t); } catch (e) { /* not remembered */ }
    // the page re-themes on load (stylesheets, plots and canvases are all set up then), so reload where you were
    try { root.sessionStorage.setItem('qublitz_scroll', JSON.stringify({ p: root.location.pathname, y: root.scrollY })); } catch (e) { /* fine */ }
    const u = new URL(root.location.href); u.searchParams.delete('theme');
    if (u.href === root.location.href) root.location.reload(); else root.location.replace(u.href);
  }
  function addSwitch() {
    const bar = doc.querySelector('.qsitebar');
    if (!bar || bar.querySelector('.rt-theme')) return;
    const wrap = doc.createElement('span'); wrap.className = 'rt-wrap';
    wrap.innerHTML = `<button type="button" class="rt-theme" aria-haspopup="true" aria-expanded="false"
      aria-label="Theme: ${LABEL[theme]}. Choose a theme" title="Theme: ${LABEL[theme]}. Click to choose another">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/></svg>
      <span class="rt-name">${LABEL[theme]}</span></button>
      <div class="rt-menu" role="menu" aria-label="Themes" hidden><p>Theme · remembered in this browser</p>${THEMES.map(t =>
        `<button type="button" role="menuitemradio" aria-checked="${t === theme}" data-t="${t}"><span class="sw">${SW[t].map(col =>
          `<i data-rt-done="1" style="background:${col}"></i>`).join('')}</span><span>${LABEL[t]}<small>${BLURB[t]}</small></span></button>`).join('')}</div>`;
    const btn = wrap.querySelector('.rt-theme'), menu = wrap.querySelector('.rt-menu');
    const items = () => [...menu.querySelectorAll('button')];
    const open = on => { menu.hidden = !on; btn.setAttribute('aria-expanded', String(on)); if (on) (menu.querySelector('[aria-checked="true"]') || items()[0]).focus({ preventScroll: true }); };
    btn.addEventListener('click', () => open(menu.hidden));
    menu.addEventListener('click', e => { const b = e.target.closest('button[data-t]'); if (!b) return; open(false); if (b.dataset.t !== theme) choose(b.dataset.t); });
    menu.addEventListener('keydown', e => {
      const list = items(), i = list.indexOf(doc.activeElement);
      if (e.key === 'Escape') { open(false); btn.focus(); }
      else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); list[(i + (e.key === 'ArrowDown' ? 1 : list.length - 1)) % list.length].focus({ preventScroll: true }); }
    });
    doc.addEventListener('click', e => { if (!wrap.contains(e.target)) open(false); });
    const sp = bar.querySelector('.sp');
    if (sp && sp.nextSibling) bar.insertBefore(wrap, sp.nextSibling); else bar.appendChild(wrap);
  }
  // after a theme change, come back to the same place on the page
  root.addEventListener('load', () => {
    try {
      const v = JSON.parse(root.sessionStorage.getItem('qublitz_scroll') || 'null'); root.sessionStorage.removeItem('qublitz_scroll');
      if (v && v.p === root.location.pathname) { root.scrollTo(0, v.y); setTimeout(() => root.scrollTo(0, v.y), 350); }
    } catch (e) { /* fine */ }
  });
  // the bar may be added during DOMContentLoaded (pages without <body>), so try again just after it and at load
  const trySwitch = () => { addSwitch(); setTimeout(addSwitch, 0); };
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', trySwitch); else trySwitch();
  root.addEventListener('load', addSwitch);

  root.RETRO = { theme, themes: THEMES.slice(), labels: Object.assign({}, LABEL), palette: P[theme] || null, choose };
})(window);
