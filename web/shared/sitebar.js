/* sitebar.js: a slim bar at the top of each page with links to the published site, the GitHub
 * repository and, when the page is embedded (for example in a Google Site) or is the downloaded
 * offline copy, a button that opens this page full size in its own tab.
 *
 * Include it as <script src="…/shared/sitebar.js"></script> where the bar should appear; it inserts
 * itself in place of that tag. The page's path on the site comes from where this script was loaded
 * from, or from window.QUBLITZ_PATH, which scripts/build_standalone.py sets in the offline copies.
 */
(function () {
  'use strict';
  const SITE = 'https://mvwf.github.io/qublitz/', REPO = 'https://github.com/mvwf/qublitz';
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
    '.qsitebar.hub{padding:6px max(22px,calc((100% - 1180px) / 2 + 22px));margin:0;background:#f5f9f6}';
  const bar = document.createElement('nav');
  bar.className = 'qsitebar' + (path ? '' : ' hub');
  bar.setAttribute('aria-label', 'Site links');
  // embedded or offline, links open a new tab so the page stays where it is; on the site itself they navigate
  const away = embedded || offline ? ' target="_blank" rel="noopener"' : '';
  const link = (href, text, cls) => `<a href="${href}"${away}${cls ? ` class="${cls}"` : ''}>${text}</a>`;
  bar.innerHTML =
    link(SITE, 'mvwf.github.io/qublitz') + '<span>·</span>' + link(REPO, 'Source on GitHub') + '<span class="sp"></span>' +
    (embedded || offline ? link(SITE + path, 'Open the full page ↗', 'full') : '');
  me.before(css, bar);
  // the full page opens with the settings in this page's URL, as they are at the moment of the click
  const full = bar.querySelector('a.full');
  if (full && !offline) full.addEventListener('click', () => { full.href = SITE + path + location.search + location.hash; });
})();
