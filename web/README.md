# Static demo site

Everything under `web/` is published to GitHub Pages by `.github/workflows/pages.yml`
(at https://mvwf.github.io/qublitz/ once Pages is enabled). No build step and no
server: each demo is a single self-contained HTML file.

```
web/
  index.html            hub page: cards, filters, search, in-page viewer
  courses/engs23/       ENGS 23 intuition helpers
  qublitz/              Qublitz pages ported from Streamlit
                        (arena.html is copied in from pages/_assets/ at build time)
  vendor/               third-party libraries served with the site (Plotly, MIT license)
```

Pages load their libraries from `vendor/`, not a CDN, so the site keeps working if a CDN
drops a version and the pages run offline. Plotly comes from the npm package
`plotly.js-dist-min@2.35.2` (`npm pack`, which checks the registry's integrity hash).

## Adding a demo

1. Put the HTML file in the right folder (e.g. `web/courses/engs53/my-demo.html`).
   Give it a back link to `../../index.html`.
2. Add an entry to `DEMOS` near the top of the script in `web/index.html`.
   For a new course, also add an entry to `GROUPS`.
3. Push to `main`. The workflow fails if a `DEMOS` href points at a missing file.

Preview locally with `python3 -m http.server -d web` and open http://localhost:8000.
Opening `web/index.html` directly also works. Only the Arena card needs the built
site, because its file is copied in at build time.

## Qubit Simulator engine

`qublitz/qubit-engine.js` is the browser port of `quantum_simulator.py` (Lindblad solve,
same Hamiltonian and conventions) plus the page's fit and Ramsey helpers. It is shared
by the ported pages and has no dependencies.

`qublitz/tests/qubit-engine.test.js` (run in CI) checks it against
`qublitz/tests/qubit-reference.json`, which `scripts/make_qubit_reference.py` writes by
running the Python originals. If you change the physics on either side, regenerate the
reference and make sure the test still passes:

```
python scripts/make_qubit_reference.py
node web/qublitz/tests/qubit-engine.test.js
```

Hidden qubits come from a shareable code (`?qubit=K7RX4P`). The same code gives the same
qubit everywhere. The value ranges are `QUBIT_RANGES` in the engine.

## Fridge Noise Explorer

`qublitz/fridge-engine.js` ports the attenuator-chain model from
`pages/Dilution_Refrigerator_Noise_Explorer.py`, and `qublitz/fridge-noise.html` is its page.
`qublitz/tests/fridge-engine.test.js` (run in CI) checks the port against
`qublitz/tests/fridge-reference.json`, written by `scripts/make_fridge_reference.py` from the
page's own Python functions. To regenerate:

```
python scripts/make_fridge_reference.py
node web/qublitz/tests/fridge-engine.test.js
```

## Offline download

The Qubit Simulator, the Fridge Noise Explorer and the three ENGS 23 demos each offer a single-file download
(`<page>-offline.html`, next to the page) that runs from disk in any browser with no network. The deploy workflow builds it with
`scripts/build_standalone.py`, which inlines the page's local scripts (Plotly and the engine)
(if any) and fails if anything is still loaded from the network. The Spectrogram Explorer's
archive.org recordings still need a connection; its bundled gong, uploads and microphone do not. To build it yourself:

```
python scripts/build_standalone.py web/qublitz/qubit-simulator.html qubit-simulator-offline.html
```

The download link 404s in a plain local preview of `web/`, because the file only exists in
the built site.

## Arena cross-links

The Arena links to other Qublitz pages by name (`Qubit_Simulator`, ...).
`web/qublitz/<Name>.html` forwards each name either to the ported page
(`Qubit_Simulator` → `qubit-simulator.html`,
`Dilution_Refrigerator_Noise_Explorer` → `fridge-noise.html`) or, until it is ported, to the Streamlit app.

## Embedding in Google Sites

Insert → Embed → By URL → `https://mvwf.github.io/qublitz/`, then drag the embed box
to full width and a tall height. To embed a single demo, use its own URL instead.
