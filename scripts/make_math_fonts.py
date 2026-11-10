"""Build the self-hosted math face (issue #158): WOFF2 subsets of Latin Modern for web/vendor/fonts/.

The pages set equations in 'Latin Modern Roman'; web/shared/retro.js declares that family with these files, so
equations look the same on every machine without loading anything from a third-party server.

Sources (GUST Font License, see web/vendor/fonts/GUST-FONT-LICENSE.txt), from CTAN:
    https://mirrors.ctan.org/fonts/lm.zip       fonts/opentype/public/lm/lmroman10-{regular,italic,bold}.otf
    https://mirrors.ctan.org/fonts/lm-math.zip  opentype/latinmodern-math.otf
Latin Modern Roman has no lowercase Greek, primes, hbar, partial, nabla, integrals, angle brackets and the like;
those come from Latin Modern Math, which is subset to exactly the characters Latin Modern Roman lacks. Latin Modern
Math is upright only, so a second, smaller file for italic text holds its Greek, punctuation, letterlike symbols and
accents, with lowercase Greek (and the partial sign) mapped to the math italic letters (U+1D6FC-1D71B), as TeX sets
them; operators, arrows and brackets stay upright in italic text, from the first file.

Needs fonttools and brotli (pip install fonttools brotli).
Usage (from the repo root):
    python scripts/make_math_fonts.py <dir with the four .otf files> web/vendor/fonts
It prints each file's unicode-range, for the @font-face rules in web/shared/retro.js.
"""
import os
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

# Basic Latin, Latin-1, a few Latin Extended-A letters (dotless i, hbar, oe), combining accents, Greek, general
# punctuation (thin spaces, primes), super/subscripts, letterlike symbols (hbar), arrows, math operators,
# miscellaneous technical (ceilings, floors) and the angle brackets of bras and kets.
RANGES = [(0x20, 0x7E), (0xA0, 0xFF), (0x127, 0x127), (0x131, 0x131), (0x152, 0x153), (0x300, 0x36F),
          (0x370, 0x3FF), (0x2000, 0x206F), (0x2070, 0x209F), (0x20AC, 0x20AC), (0x2100, 0x214F),
          (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2300, 0x23FF), (0x27C0, 0x27EF)]
TEXT = ["regular", "italic", "bold"]  # bold italic is synthesized from the italic (three characters on one page)
SUFFIX = "qbsub"                      # renamed, as the GUST Font License requests for derived works
# the part of the math subset that gets an italic file: accents, Greek, punctuation, letterlike symbols, partial
ITALIC_RANGES = [(0x300, 0x36F), (0x370, 0x3FF), (0x2000, 0x206F), (0x2100, 0x214F), (0x2202, 0x2202)]
# math italic small alpha..omega, partial, epsilon, theta, kappa, phi, rho and pi symbols (U+1D6FC-1D71B)
ITALIC = dict(zip(list(range(0x3B1, 0x3CA)) + [0x2202, 0x3F5, 0x3D1, 0x3F0, 0x3D5, 0x3F1, 0x3D6], range(0x1D6FC, 0x1D71C)))


def wanted(cmap, exclude=()):
    # combining accents are kept even where Latin Modern Roman has them: an accent must come from the same file as
    # its letter (a tilde over kappa), or the browser draws the pair in a fallback font
    return sorted(c for a, b in RANGES for c in range(a, b + 1)
                  if c in cmap and (c not in exclude or 0x300 <= c <= 0x36F))


def css_range(codes):
    out, start, prev = [], None, None
    for c in codes + [None]:
        if c is not None and prev is not None and c == prev + 1:
            prev = c
            continue
        if start is not None:
            out.append(f"U+{start:X}" if start == prev else f"U+{start:X}-{prev:X}")
        start = prev = c
    return ",".join(out)


def build(src, dst, codes, name, features, remap=None):
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.hinting = False
    opts.layout_features = features
    opts.drop_tables += ["MATH", "FFTM"]
    opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14]
    font = TTFont(src, recalcTimestamp=False)
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=[remap.get(c, c) for c in codes] if remap else codes)
    sub.subset(font)
    if remap:  # point each character at its math italic glyph and drop the math italic code points themselves
        for table in font["cmap"].tables:
            for c, m in remap.items():
                if c in codes and m in table.cmap:
                    table.cmap[c] = table.cmap.pop(m)
    note = f"{name}: a subset of the GUST {os.path.basename(src)} for Qublitz (issue #158); not the original"
    table = font["name"]
    for rec in list(table.names):
        if rec.nameID in (1, 4, 16):
            rec.string = name
        elif rec.nameID == 6:
            rec.string = name.replace(" ", "")
        elif rec.nameID == 3:
            rec.string = note
    table.setName(note, 10, 3, 1, 0x409)
    font.flavor = "woff2"
    font.save(dst)
    return os.path.getsize(dst)


def main(srcdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    roman = set(TTFont(os.path.join(srcdir, "lmroman10-regular.otf")).getBestCmap())
    total = 0
    for style in TEXT:
        src = os.path.join(srcdir, f"lmroman10-{style}.otf")
        codes = wanted(TTFont(src).getBestCmap())
        dst = os.path.join(outdir, f"lmroman10-{style}-{SUFFIX}.woff2")
        size = build(src, dst, codes, f"QB LMRoman10 {style.capitalize()} Subset", ["kern", "liga", "mark", "mkmk", "ccmp"])
        total += size
        print(f"{os.path.basename(dst)}  {size} bytes  {len(codes)} chars\n  unicode-range:{css_range(codes)}")
    src = os.path.join(srcdir, "latinmodern-math.otf")
    codes = wanted(TTFont(src).getBestCmap(), exclude=roman)
    dst = os.path.join(outdir, f"latinmodern-math-{SUFFIX}.woff2")
    size = build(src, dst, codes, "QB LatinModernMath Subset", ["ccmp"])
    total += size
    print(f"{os.path.basename(dst)}  {size} bytes  {len(codes)} chars\n  unicode-range:{css_range(codes)}")
    italic = [c for c in codes if any(a <= c <= b for a, b in ITALIC_RANGES)]
    dst = os.path.join(outdir, f"latinmodern-math-italic-{SUFFIX}.woff2")
    size = build(src, dst, italic, "QB LatinModernMath Italic Subset", ["ccmp"], remap=ITALIC)
    total += size
    print(f"{os.path.basename(dst)}  {size} bytes  {len(italic)} chars\n  unicode-range:{css_range(italic)}")
    upright = [c for c in codes if c not in italic]
    print(f"  in italic text the rest stays upright, from {SUFFIX} math: unicode-range:{css_range(upright)}")
    print(f"total {total} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
