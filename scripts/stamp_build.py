"""Append the build stamp to the site's outreach-config.js (run by .github/workflows/pages.yml).

Every event and question record then carries `gv` = '<date this page's files last changed>+<that commit>', so
the outreach data can be split by game version as the games keep changing. A page's gv changes only when one of
its own files changes, not on every site deploy. A page's files are its HTML, its
web/learning/<id>.js and the local scripts it loads (engines); the Arena's source is the vendored game.

Usage: python3 scripts/stamp_build.py _site   (needs the full git history: checkout with fetch-depth: 0)
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = sys.argv[1] if len(sys.argv) > 1 else '_site'


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def last_changed(paths):
    paths = [p for p in paths if os.path.exists(os.path.join(ROOT, p))]
    return git('log', '-1', '--format=%cs+%h', '--', *paths) if paths else ''


manifest = open(os.path.join(ROOT, 'web/learning/manifest.js'), encoding='utf-8').read()
pages = {}
for gid, path in re.findall(r"id:\s*'([a-z0-9-]+)'[^}]*?path:\s*'([^']*)'", manifest):
    src = 'pages/_assets/quantum_chess.html' if path == 'qublitz/arena.html' else ('web/' + (path or 'index.html'))
    files = [src, f'web/learning/{gid}.js']
    if src.startswith('web/') and os.path.exists(os.path.join(ROOT, src)):
        html = open(os.path.join(ROOT, src), encoding='utf-8').read()
        base = os.path.dirname(src)
        for s in re.findall(r'<script src="([^"]+)"></script>', html):
            if not re.match(r'(?:https?:)?//', s) and 'shared/' not in s and 'learning/' not in s:
                files.append(os.path.normpath(os.path.join(base, s)))
    pages[path] = last_changed(files)

stamp = {'sha': git('rev-parse', '--short', 'HEAD') or 'dev', 'built': git('log', '-1', '--format=%cs'), 'pages': pages}
with open(os.path.join(SITE, 'shared', 'outreach-config.js'), 'a', encoding='utf-8') as f:
    f.write('\n// Build stamp (scripts/stamp_build.py): the page version sent with every outreach record.\n')
    f.write('window.QUBLITZ_BUILD = ' + json.dumps(stamp, separators=(',', ':')) + ';\n')
print(json.dumps(stamp, indent=1))
