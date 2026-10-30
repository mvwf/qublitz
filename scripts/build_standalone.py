"""Bundle a web/ page and its local scripts into one self-contained HTML file.

The published Qubit Simulator offers this file as a download, so it runs offline
from disk in any browser with nothing to install. Every <script src="..."> with a
relative path is inlined; the build fails if the page still loads anything from
the network, so the download really is self-contained.

Usage (from the repo root):
    python scripts/build_standalone.py web/qublitz/qubit-simulator.html out.html
"""
import os
import re
import sys

SCRIPT_TAG = re.compile(r'<script src="([^"]+)"></script>')
REMOTE = re.compile(r'<(?:script|link)\b[^>]*\b(?:src|href)="(?:https?:)?//', re.IGNORECASE)
# Set before the page's own scripts run, so the page knows it is the downloaded copy.
MARKER = "<script>window.QUBLITZ_STANDALONE = true;</script>\n"


def inline(match, base):
    src = match.group(1)
    if re.match(r"(?:https?:)?//", src):
        sys.exit(f"build_standalone: {src} is remote; ship it under web/ instead")
    with open(os.path.join(base, src), encoding="utf-8") as f:
        code = f.read()
    # A literal "</script" inside the code would end the inline <script> element early.
    code = code.replace("</script", "<\\/script")
    return f"<script>/* {src} */\n{code}\n</script>"


def main(page, out):
    with open(page, encoding="utf-8") as f:
        html = f.read()
    base = os.path.dirname(page)
    first = SCRIPT_TAG.search(html)
    if not first:
        sys.exit(f"build_standalone: no <script src> tags in {page}")
    html = html[:first.start()] + MARKER + html[first.start():]
    html = SCRIPT_TAG.sub(lambda m: inline(m, base), html)
    if REMOTE.search(html):
        sys.exit(f"build_standalone: {page} still loads a remote script or stylesheet")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
