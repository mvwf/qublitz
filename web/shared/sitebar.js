/* sitebar.js: a slim bar at the top of each page with links to the published site, Coherence Vault (the MARQUIS database of
 * real device numbers), the GitHub repository and, when the page is embedded (for example in a Google Site) or is the downloaded
 * offline copy, a button that opens this page full size in its own tab.
 *
 * Include it as <script src="…/shared/sitebar.js"></script> where the bar should appear; it inserts
 * itself in place of that tag. The page's path on the site comes from where this script was loaded
 * from, or from window.QUBLITZ_PATH, which scripts/build_standalone.py sets in the offline copies.
 * It also adds the funding line at the foot of every page (FUNDING below; the hub shows the full acknowledgment).
 */
(function () {
  'use strict';
  const SITE = 'https://mvwf.github.io/qublitz/', REPO = 'https://github.com/mvwf/qublitz', CVAULT = 'https://mvwf.github.io/CoherenceVault/';
  // Who pays for Qublitz. Keep in step with the funding section of index.html and the README.
  const FUNDING = 'Qublitz is supported by the Unitary Foundation and by the U.S. National Science Foundation through an ' +
    '<a href="https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2540460" target="_blank" rel="noopener">NSF CAREER award (No. 2540460)</a> and the ' +
    '<a href="https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2553633" target="_blank" rel="noopener">NSF MARQUIS Quantum Leap Challenge Institute (No. 2553633)</a>.';
  const me = document.currentScript;
  if (!me) return;

  // the page's path relative to the site root, e.g. "qublitz/rabi-rider.html" ("" for the hub)
  let path = window.QUBLITZ_PATH;
  if (path == null && me.src) {
    const root = me.src.replace(/shared\/sitebar\.js(?:[?#].*)?$/, '');
    path = location.href.startsWith(root) ? location.href.slice(root.length).replace(/[?#].*$/, '') : '';
  }
  path = (path || '').replace(/^index\.html$/, '');

  // Inside the hub's own viewer the hub already offers "Open in new tab", so the bar stays out of the way.
  let embedded = false, ownHub = false;
  try { embedded = window.self !== window.top; ownHub = embedded && window.top.location.origin === location.origin; }
  catch (e) { embedded = true; }   // a cross-origin parent, such as a Google Site
  if (ownHub) return;
  const offline = !!window.QUBLITZ_STANDALONE || location.protocol === 'file:';

  const css = document.createElement('style');
  css.textContent =
    '.qsitebar{display:flex;flex-wrap:wrap;align-items:center;gap:4px 14px;font:12.5px/1.5 Helvetica,"Helvetica Neue",Arial,sans-serif;' +
    'color:#5a6b63;padding:6px 0 8px;margin:0 0 6px;border-bottom:1px solid #d8e2dc}' +
    '.qsitebar a{color:#00693e;text-decoration:none;font-weight:600}.qsitebar a:hover{text-decoration:underline}' +
    '.qsitebar .sp{flex:1}' +
    '.qsitebar a.full{background:#00693e;color:#fff;border-radius:6px;padding:3px 10px}.qsitebar a.full:hover{background:#12855a;text-decoration:none}' +
    '.qsitebar.hub{padding:6px max(22px,calc((100% - 1180px) / 2 + 22px));margin:0;background:#f5f9f6}' +
    '.qsitebar a.qv{background:rgba(0,105,62,.09);border-radius:6px;padding:4px 8px;display:inline-block}' +
    '.qfund{font:12px/1.5 Helvetica,"Helvetica Neue",Arial,sans-serif;color:#5a6b63;border-top:1px solid #d8e2dc;margin:28px auto 0;padding:10px 16px 18px;max-width:1180px;text-align:center}' +
    '.qfund a{color:#00693e}';
  const bar = document.createElement('nav');
  bar.className = 'qsitebar' + (path ? '' : ' hub');
  bar.setAttribute('aria-label', 'Site links');
  // embedded or offline, links open a new tab so the page stays where it is; on the site itself they navigate
  const away = embedded || offline ? ' target="_blank" rel="noopener"' : '';
  const link = (href, text, cls) => `<a href="${href}"${away}${cls ? ` class="${cls}"` : ''}>${text}</a>`;
  bar.innerHTML =
    link(SITE, 'mvwf.github.io/qublitz') + '<span>·</span>' +
    `<a href="${CVAULT}" target="_blank" rel="noopener" class="qv" title="Coherence Vault: published T₁, T₂, quality factors and wiring of real superconducting devices, each linked to its paper">📚 Real device data: Coherence Vault ↗</a>` +
    '<span>·</span>' + link(REPO, 'Source on GitHub') + '<span class="sp"></span>' +
    (embedded || offline ? link(SITE + path, 'Open the full page ↗', 'full') : '');
  // Pages without an explicit <body> parse this script inside <head>, where the bar would be invisible: wait for the body.
  if (me.closest('head')) document.addEventListener('DOMContentLoaded', () => document.body.prepend(css, bar));
  else me.before(css, bar);
  // the full page opens with the settings in this page's URL, as they are at the moment of the click
  // the funding line at the foot of the page (the hub has its own, fuller section)
  const foot = () => {
    if (!path || document.querySelector('.qfund')) return;
    const f = document.createElement('footer'); f.className = 'qfund'; f.innerHTML = FUNDING; document.body.append(f);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', foot); else setTimeout(foot, 0);
  const full = bar.querySelector('a.full');
  if (full && !offline) full.addEventListener('click', () => { full.href = SITE + path + location.search + location.hash; });
})();
