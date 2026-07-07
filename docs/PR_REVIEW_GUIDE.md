# PR Review Guide: `quantum_chess.html`

This repo vendors `pages/_assets/quantum_chess.html` — the QuBlitz Arena game — as a
single ~4,000-line HTML/CSS/JS file. It is realistically unreviewable cold: there is
no build step and no module boundaries to lean on, so a reviewer scanning a diff has
no map of what region of the file they're looking at or why it matters. This doc is
that map. Read it once (~10 minutes), then jump to whichever anchors your PR touches
(~another ~10 minutes) — that should get you from clone to an informed
approve/request-changes in about 20 minutes total.

**Line-number caveat:** every line number below was read from the *canonical*
upstream source at `QuBlitz Project/quantum_chess.html` (4,322 lines as of this
writing), not from this repo's vendored copy at `pages/_assets/quantum_chess.html`
(currently 4,058 lines — it predates some canonical changes and will be refreshed in
a follow-up PR). Numbers will drift once that refresh lands. Anchor on the function
and variable **names** below (`QPhysics`, `lindbladStep`, `guardCritProb`, etc.) — all
of them are `grep`-stable — and treat line numbers as "was roughly here," not exact
addresses.

## 1. Orientation

QuBlitz Arena is a gate-level tactics game that teaches quantum mechanics by making
its game mechanics *be* the physics: every unit on the board is a qubit, every
action button is a quantum gate, and combat is resolved by literally measuring a
unit's state. A player charges a unit toward `|1⟩` (excited, "ready to strike") or
into superposition, closes to melee range before T1/T2 decoherence bleeds the charge
away, and attacks — which performs a Born-rule projective measurement on the
attacker. The GUARD stance measures a defender in the X-basis instead of the Z-basis,
which makes relative phase (Z/S/T gates) mechanically consequential rather than
cosmetic, and CNOT produces a real two-qubit Bell pair with correlated,
no-communication-respecting collapse.

It's one file by design, not by neglect: the game must stay a single self-contained
file that opens standalone in Chrome with no build step and no CDN dependencies (the
one external reference is a Google Fonts `@import` for display type — everything that
touches game logic is inline). That constraint is why you'll find CSS, markup, and
all of the JS — physics engine, bot AI, renderer, UI glue — concatenated into one
`<html>` document instead of split into modules.

It's embedded in this repo by `pages/QuBlitz_Arena.py`, which is a thin Streamlit
wrapper: `_load_game_html()` reads `pages/_assets/quantum_chess.html` as a raw string
(cached via `@st.cache_data`), and `_render_embed()` passes it to
`st.components.v1.html(..., height=920, scrolling=True)`, which renders it inside an
iframe. The Python side does exactly one piece of injection before handing the HTML
off: if a `QB_SAGE_PROXY_URL` is configured (env var or Streamlit secrets), it
prepends `<script>window.QB_SAGE_PROXY=...</script>` so the game can find its AI-Sage
proxy endpoint (see anchor 7 below). With no proxy configured, the script tag is
simply omitted and the game falls back to its offline heuristic Sage — no error, no
missing feature, just a different advice source.

## 2. The 10 anchor points

All line numbers below are from the canonical file at
`/Users/davidmukuruva/Desktop/Academics/Projects/QuBlitz Project/quantum_chess.html`.

### 2.1 The physics engine object — `QPhysics`
**`const QPhysics = (() => {...})();`, opens at line 1081, closes ~1282 (returns
at 1278-1280, IIFE closes 1281, self-test auto-runs 1282-1285).**

This is the whole quantum-state math layer: Bloch-vector conversions
(`blochFromPure`/`pureFromBloch`), `charge`, `guardCritProb`, the Lindblad
decoherence step, the density-matrix gate-application layer (`rhoFromBloch`,
`gateBloch`, `gateMatrix`), the `Bell` entanglement sub-object, Z-basis `measure`,
and the `diagnose`/`selfTest` regression harness. It is written with **no DOM
dependency** — it never touches `document` or `GAME` — specifically so it can be
extracted verbatim into a Node test file (see anchor 3 and section 3). **Check on
change:** does the diff preserve that DOM-independence? Any new function here that
reaches for `document.*` or a global game-state object breaks the extraction trick
`tests/qphysics.test.js` relies on, and silently stops the physics regression suite
from testing the real code.

### 2.2 Born-rule measurement / combat resolution
**Three sites, by design (see anchor 6 for why there are two attack-resolution
paths):**
- Line 1204: `QPhysics.measure` — "Born-rule projective measurement in the Z basis,"
  `Math.random() < charge(r)`, collapses to `{x:0,y:0,z:±1}`.
- Line 1420-1439: `BotAI.resolveAttack(attacker, target, rng)` — the pure,
  headlessly-testable port of the in-browser `attack()` function's combat math: a
  Born-rule "fire roll" (`rng() < pAtt`), then the GUARD X-basis measurement
  (`GATES.H(target.state)` then a Z-basis draw), crit detection, and damage.
- The live in-browser call site is `attack()` around line 2010-2043, which as of the
  "GD-1" comment there now delegates to `BotAI.resolveAttack` for the actual math
  rather than re-implementing it, so the browser and the test harness provably run
  the same formula.

**Check on change:** if someone edits the fire-roll or crit formula in one of these
three places, did they edit (or at least re-verify) the other two? A drift between
`BotAI.resolveAttack` and the in-browser caller would mean the tests are no longer
testing what players actually experience.

### 2.3 The Lindblad decoherence step — `lindbladStep`
**`function lindbladStep(r, { T1, T2, dt = 1 })`, line 1113-1118.**

This is the exact analytical solution to the open-quantum-system (Lindblad
master-equation) decay for amplitude damping (T1) and dephasing (T2) applied to a
Bloch vector: `x,y` decay by `e^(-dt/T2)`, `z` relaxes toward `+1` (ground state) by
`e^(-dt/T1)`, with `T2` clamped to `≤ 2·T1` (the physical constraint). This is the
single most physics-correctness-critical function in the file — it is what the
falsifiability harness (`diagnose`, anchor 3 / section 3) exists to validate, and
it's the function every "charge bleeds away over time" mechanic in the game
ultimately calls. **Check on change:** any edit here should come with an updated
`node tests/qphysics.test.js` run showing R²(exponential) still > 0.99 and the fitted
T1 still recovering the input T1 — a hand-wavy "decay felt better this way" edit
here would break the game's core physics claim.

### 2.4 The GUARD crit rule — `guardCritProb`
**`const guardCritProb = pureOrBloch => {...}`, line 1104-1107.**

`guardCritProb = (1 − x) / 2`, where `x` is the Bloch x-component. The comment above
it (lines 1099-1103) explains the reasoning: a GUARDing unit is measured in the
X-basis rather than the Z-basis, so its chance of being caught in the "excited"
outcome of *that* basis is `P(|−⟩)`. This makes relative phase — otherwise invisible
to the Z-basis `charge` a player watches all game — mechanically real: `|+⟩` (x=+1)
is crit-immune under GUARD, `|−⟩` (x=−1) is fully exposed, and `|0⟩`/`|1⟩` (x=0, no
defined phase) are a coin flip. This is called out in the project's own docs as a
genuine pedagogical invention (not a textbook mechanic) precisely because it's the
one place phase management has a stakes-bearing payoff — a reviewer should understand
*why* the formula is `(1-x)/2` and not just accept that it is, since it's the
justification for GUARD existing as a mechanic at all. **Check on change:** does an
edit preserve the property that pure `|+⟩`/`|−⟩` states hit the 0%/100% extremes, and
does the in-game copy (Physics Lab tooltip text, GUARD button explainer text) stay in
sync with the formula if it changes?

### 2.5 The Bell-pair / entanglement state — `Bell`
**`const Bell = (() => {...})();`, line 1133-1181, nested inside `QPhysics`.**

Models a shared 4-amplitude two-qubit state `[c00,c01,c10,c11]`. `phiPlus()`
constructs `|Φ+⟩ = (|00⟩+|11⟩)/√2` (what CNOT produces from `|+⟩⊗|0⟩`);
`marginalP1(a, which)` computes a single qubit's marginal `P(|1⟩)` by summing over
the partner; `applyLocal(a, which, U)` applies a 2×2 unitary to *only one* qubit's
half of the joint state (tensored with identity on the other); `measure(a, which,
rand)` performs a correlated projective measurement and returns the renormalized
post-collapse state plus both outcomes. The comment at lines 1130-1132 flags that
this replaces older code that applied a gate directly to the entangled partner — "the
#1 entanglement misconception" — so this correctness is a deliberate fix, not
incidental. Use-sites reach it via `bell.amps` (the stored joint-state array on an
entangled unit pair) and `Bell.applyLocal`/`Bell.measure`. **Check on change:** does a
local gate on one half of a pair still leave the partner's `marginalP1` provably
unchanged (no-communication)? Does measuring one half still force a fully correlated
outcome on the other? `tests/qphysics.test.js` asserts both — re-run it after any
`Bell` edit.

### 2.6 The bot AI module — `BotAI`
**`const BotAI = (() => {...})();`, opens line 1297, closes 1441-1442.**

A newer, pure/DOM-free module (`scoreMoveSpots`, `pickTarget`, `chooseAction`,
`resolveAttack`) extracted from the logic that used to live only inline inside the
browser's `doBotMove()`/`attack()` functions — specifically so the bot's decision
policy and combat math can be exercised headlessly in a test harness
(`tests/bot_guard_harness.test.js`, see section 4) instead of only being verifiable
by clicking through games in a browser. This is the same extraction pattern
`QPhysics` itself already used (anchor 1): keep the logic that needs to be correct
free of `document`/DOM reads so it can be lifted out of the HTML file verbatim by a
Node test via source-text extraction (`findBlock()` regex matching, not a copy —
so the test always runs the *actual* shipped code). `chooseAction` in particular
encodes a fairly detailed decision cascade (numbered priority: kill shot → scheduled
guard → defensive measure → CNOT setup → guard-preferred → charge-up → take-the-shot)
with inline comments explaining each branch's rationale (e.g. "GD-1" comments
throughout, about making the bot actually use GUARD against X-only opponents).
**Check on change:** does a change to bot behavior stay inside this pure module
(testable) or leak DOM/`GAME`-global reads back into it (untestable, and liable to
silently diverge from what `attack()`'s live call site does — see anchor 2)?

### 2.7 The Sage proxy boundary — `SAGE_PROXY_URL`
**`const SAGE_PROXY_URL = ...`, line 3955-3957; used at the `fetch()` call ~line
3994.**

Security model: the client HTML never holds an API key. It only ever holds a proxy
*URL*, read from `window.QB_SAGE_PROXY` (a global the embedding page — see
`pages/QuBlitz_Arena.py`'s `_render_embed()` — injects via a `<script>` tag only when
a proxy is actually configured). If that global isn't set, `SAGE_PROXY_URL` is `''`,
`_sageEnabled()` returns `false`, and the game silently falls back to its offline
heuristic Sage — no key prompt, no broken feature, no exfiltration risk. The actual
Anthropic API key lives server-side in `sage_proxy.py`, which holds the key and
issues the real Claude calls behind a rate limiter. That file is **not** part of this
fork — it lives in the canonical game repo at
`/Users/davidmukuruva/Desktop/Academics/Projects/QuBlitz Project/sage_proxy.py` — so
if a PR touches Sage behavior, the proxy-side code that actually needs a security
review isn't in this diff at all; go read it there. **Check on change:** does any
new code path in this file read, log, or forward anything that looks like a
credential? The invariant to hold is "this file only ever sees a URL."

### 2.8 Board / turn state — `GAME`
**`const GAME = {...};`, line 1696 (through ~1728, initial literal; mutated
throughout the file after).**

The central mutable game-state object: `pieces` (the qubit units), `turn`/`phase`
(whose move, and whether the UI is mid-menu), `selected`/`reachable` (current
selection + cached legal moves), `turnCount`/`turnCap` (the 60-turn cap that decides
the game if no one is eliminated), `mode`/`botColor`/`difficulty`, `decoRate` (feeds
`QPhysics.timesFromRate`), `cnotMode`/`attackMode` (two-step target-picking UI
state), plus a grab-bag of presentation state (`particles`, `deaths`, `fx`, `shake`,
`floats`, `beamFlash`) and campaign/vote-mode fields. **Check on change:** this
object is read and mutated from a lot of places in the file — a new field added here
should have a comment explaining what owns it, and a reviewer should check whether
`resetGame()` (just below it) was updated to reset the new field between games.

### 2.9 The renderer — canvas draw functions
**`grep -n "^function draw" quantum_chess.html`** turns up the main entry points:
`drawSprite` (1468), `drawCharFrame` (1677), `drawSageNPC` (2260), `drawBoard`
(2301), `drawHighlights` (2372), `drawEntanglementBeams` (2428), `drawPiece` (2467),
`drawDeaths` (2572), `drawFx` (2575), `drawParticles` (2585), `drawGateMenu` (2659),
`drawTurnBanner` (2786), `drawOverlay` (2820), `drawBloch` (2873, the live
Bloch-sphere panel), `drawFloats` (3331, floating damage/crit numbers), and
`drawDecoherenceDiagram` (4132, the Physics Lab's decay-curve visualization). **Check
on change:** canvas drawing is inherently hard to review from a text diff — a
one-character coordinate or color change can silently break layout. This is exactly
what the screenshot set (section 5) exists to catch; if a PR touches any `draw*`
function, expect (and ask for) a fresh screenshot, not just code review.

### 2.10 UI panels / DOM rendering
**Side-panel population, keyed on the DOM ids `sel-eq` (line 323 CSS, 772 markup,
populated ~2953-2975) and `sel-xbasis` (778 markup, populated ~2957-2975).** These
elements show the selected unit's live `|ψ⟩` equation and its X-basis readout; the
populating code sits in the same function that also computes and displays
`guardCritProb` for the current selection (anchor 4) — so a phase-formula change and
its UI readout are adjacent and should be reviewed together. The Sage panel's own
rendering (advice text, badge state via `_updateClaudeBadge()`) lives near the Sage
proxy code (anchor 7, ~3960-3980). **Check on change:** does the displayed equation
text / X-basis readout actually match the underlying Bloch vector after a physics
change, or is it reading stale/mismatched state?

## 3. The console self-tests

`QPhysics` ships two verification entry points, both exposed on `window` at the
bottom of the IIFE (line ~1282-1285) and auto-run once at load (`QPhysics.selfTest()`
is called immediately, logging to console):

- **`QPhysics.selfTest()`** (defined 1253-1276) — a fast, synchronous set of 8 exact
  analytic-value checks: `H|0⟩ → |+⟩`, `X|0⟩ → |1⟩` (z=-1), `Z` preserves z, `charge(|1⟩)=1`,
  `charge(|+⟩)=0.5`, one Lindblad step matches `e^(-1/T1)`/`e^(-1/T2)` exactly, and
  `T2` clamps to `≤2·T1`. Returns `{ pass: boolean, tests: [[name, bool], ...] }` and
  also logs a one-line summary (`console.log`/`console.error`) — `"QPhysics
  self-test: 8/8 passed ✓"` on success, or lists failing test names.
- **`window.runDecoherenceDiagnostic(opts)`** — a thin wrapper (line 1284) around
  **`QPhysics.diagnose({ decoRate, turns })`** (defined 1229-1252): starts a unit
  fully excited (`z=-1`), steps it through `lindbladStep` for `turns` steps, fits the
  resulting charge-vs-turn curve to both an exponential and a straight line, and
  reports R² for each plus the recovered T1. A passing/healthy result is
  R²(exponential) close to 1.000 and clearly greater than R²(linear), with the
  recovered T1 close to the input T1 — this is the falsifiability check that a real
  physical decay model, not an arbitrary curve, is driving the game.

**How a reviewer runs them:** open `quantum_chess.html` directly in Chrome (or the
Streamlit-embedded version), open devtools, and in the console run
`QPhysics.selfTest()` and `runDecoherenceDiagnostic()`. The in-game Physics Lab panel
also surfaces this — it calls `QPhysics.diagnose({decoRate, turns:24})` (line 4135)
and renders it as `drawDecoherenceDiagram`, and the panel's own footer text (lines
4123, 4275) tells the player to run the same two console calls themselves.

## 4. How to run the verification gates

**In this repo:** `bash scripts/verify.sh`. Reading it (`scripts/verify.sh`, 40
lines), it runs three checks in sequence and prints a single PASS/FAIL:
1. `ruff check .` — lint.
2. `pytest -q` — the Python smoke test suite.
3. **Conditionally**, `node pages/_assets/tests/qphysics.test.js
   pages/_assets/quantum_chess.html` — but only if
   `pages/_assets/tests/qphysics.test.js` exists (i.e. the game's test file has been
   vendored alongside the HTML) **and** `node` is on `PATH`; otherwise it prints
   "SKIPPED: node not installed" and doesn't fail the gate. As of this writing that
   vendored test file does exist in this repo (`pages/_assets/tests/qphysics.test.js`,
   byte-identical to the canonical copy) and is exercised by `verify.sh` whenever
   Node is available.

**In the canonical game repo**, there are two Node test files worth knowing about
(found by listing its `tests/` directory directly — don't assume just one exists):
- `node tests/qphysics.test.js` — the physics regression suite described in section
  3; this is the one currently vendored into this fork.
- `node tests/bot_guard_harness.test.js` — a newer, **not yet vendored** headless
  bot-vs-bot harness for the "GD-1" bot-guard behavior work (anchor 6). It uses the
  same source-extraction technique (regex-matched block extraction straight out of
  `quantum_chess.html`, no copy-paste) to pull out `C`, `GATES`, `QPhysics`, and now
  `BotAI` as well, then drives a headless simulated game around
  `BotAI.chooseAction`/`BotAI.resolveAttack` to check that a Medium-difficulty bot
  actually uses GUARD to punish an "X-only" opponent. Since it isn't vendored into
  `pages/_assets/tests/` yet, `scripts/verify.sh` does not currently run it — a
  reviewer who wants to check `BotAI` changes against it needs to run it directly
  against the canonical file, or wait for a follow-up PR that vendors it the same way
  `qphysics.test.js` was vendored.

## 5. The screenshot set

Visual regression reference lives outside both repos, at
`/Users/davidmukuruva/Desktop/Academics/Projects/_arena_screenshots/`. As of this
writing it contains: `00_arena_streamlit_full.png`, `01_menu.png`, `02_board_play.png`,
`03_selected_plus.png`, `04_guard_minus.png`, `05_bell_pair.png`,
`06_sage_expanded.png`, `07_physics_lab.png`, plus a second pass (`v2_10_arena_top.png`,
`v2_11_arena_full.png`, `v2_12_game_menu.png`, `v2_13_game_fit_768.png` — the last
being a 768px tablet-width capture) and a third-pass pair
(`v3_arena_top.png`, `v3_concept_map.png`).

These should be re-captured after any UI-affecting change (anything touching a
`draw*` function, panel layout, or CSS) and eyeballed against the existing set —
not just diffed as text. A prior LLM-council review of this project found five
defects in these screenshots that a code-only review would not have caught (layout
clipping, contrast/legibility issues, and similar visual-only problems). Treat the
screenshot set the way you'd treat a snapshot test: regenerate, look at it, and
specifically check the tablet-width (768px) capture and the Sage/Physics-Lab panels,
since those are the ones with the most dynamic content and the most history of
clipping.
