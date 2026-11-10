Self-hosted math face for the Qublitz pages (issue #158)
=======================================================

The pages set equations in 'Latin Modern Roman'. web/shared/retro.js declares that CSS family with the files
below, so equations look the same on every machine. They are served from this site; nothing is loaded from a
third-party server.

These are modified (subset, renamed) versions of fonts from the Latin Modern project, not the originals.
Originals: the Latin Modern fonts by Bogusław Jackowski and Janusz M. Nowacki (GUST e-foundry), based on
Donald E. Knuth's Computer Modern. Copyright 2003--2021 B. Jackowski and J.M. Nowacki (on behalf of TeX Users
Groups). Released under the GUST Font License (GUST-FONT-LICENSE.txt, an instance of the LaTeX Project Public
License 1.3c). Unmodified originals: https://ctan.org/pkg/lm and https://ctan.org/pkg/lm-math
(http://www.gust.org.pl/projects/e-foundry/latin-modern/).

Manifest (derived files and what they were made from)
  lmroman10-regular-qbsub.woff2   "QB LMRoman10 Regular Subset"  from lm 2.005  lmroman10-regular.otf
  lmroman10-italic-qbsub.woff2    "QB LMRoman10 Italic Subset"   from lm 2.005  lmroman10-italic.otf
  lmroman10-bold-qbsub.woff2      "QB LMRoman10 Bold Subset"     from lm 2.005  lmroman10-bold.otf
  latinmodern-math-qbsub.woff2    "QB LatinModernMath Subset"    from lm-math 1.959  latinmodern-math.otf
  latinmodern-math-italic-qbsub.woff2  "QB LatinModernMath Italic Subset"  from lm-math 1.959  latinmodern-math.otf

Changes from the originals: subset to Latin, Greek, punctuation and math characters (the Latin Modern Math file
holds only the characters Latin Modern Roman lacks, such as lowercase Greek, primes, angle brackets and most
math operators, plus the combining accents; the italic file maps lowercase Greek and the partial sign to the
font's own math italic letters, U+1D6FC-1D71B), hinting and the MATH table removed, fonts renamed, converted to
WOFF2. Made with scripts/make_math_fonts.py (fonttools), which lists the exact character ranges.
