# Perf measurement methodology (UP-4)

`scripts/measure_perf.py` measures two numbers per page, using Streamlit's own
`AppTest` framework (already used by `tests/test_pages_smoke.py` — no new
dependency):

- **time-to-interactive**: wall-clock time for `AppTest.from_file(path).run()`
  to complete a script's first run — imports, top-level compute, first render.
- **recompute-after-widget-change**: wall-clock time for a second `at.run()`
  after nudging one real input widget (not a display-only checkbox — see the
  Dilution note below) to a different value, simulating the actual common
  case: a student drags one slider and the whole script reruns.

Run: `python3 scripts/measure_perf.py` from the repo root.

## What this does NOT measure

`AppTest` runs the script's Python top-to-bottom and builds Streamlit's
element tree — it does **not** launch a browser, so it excludes real-browser
cost: Plotly chart rendering/re-layout, network round-trips, and actual
cold-start (process boot + dependency import on a fresh deploy). Streamlit's
own "the page froze" user complaint is the SUM of the Python recompute time
this script measures *plus* that browser-side cost — a page can measure fast
here and still feel slow in a real browser if its charts are heavy to
re-render. Treat these numbers as the Python-side floor, not the whole story;
a full before/after "user-perceived" number would need a real Playwright-
driven measurement against a live `streamlit run`, which is a heavier
follow-up, not done here.

## Before (2026-07-09, pre-UP-3)

| Page | Time-to-interactive | Recompute after 1 widget change |
|---|---|---|
| Dilution_Refrigerator_Noise_Explorer | 1.74s | 0.06s |
| Laser_Heating_Calculator | 0.04s | 0.03s |
| Quantum_Measurement_Tutorial | 1.00s | 0.90s |
| EP_TPD_exploration | 0.47s | 0.46s |

**Honest reading, not the reading I expected going in:** the 4 pages UP-3
names as "ungated" are NOT uniformly slow. Quantum_Measurement_Tutorial's
recompute cost (0.90s) is the real standout — consistent with UP-3's own
done-when singling it out ("slider-drag on the Tutorial no longer freezes the
page"). Dilution's recompute (0.06s) is fast even *before* any gating — its
own noise-pipeline math isn't expensive at this problem size; what UP-3 fixes
there is wasted reruns on unrelated widget changes, not a slow computation.
Laser_Heating_Calculator is fast on both metrics already — "zero gating" in
the original audit did not mean "slow," it meant "reruns needlessly," and the
needless-rerun cost here is small in absolute terms. This changes the
priority order: Tutorial first, Dilution/EP_TPD next (moderate, real but
smaller), Laser Heating last (correctness of the pattern, not a felt-latency
fix).

**First attempt for Dilution used a display-only checkbox** ("Show plot:
thermal photon number vs frequency") instead of a temperature `number_input`
— measured a nearly-identical 0.06s either way once corrected, but the first
draft would have been driving the wrong kind of widget for what this number
is supposed to represent (a change that touches the real compute path, not
just a plot toggle) even though it happened not to change the answer here.

## After (2026-07-09, post-UP-3)

| Page | Time-to-interactive | Recompute after 1 widget change |
|---|---|---|
| Dilution_Refrigerator_Noise_Explorer | 1.03s | 0.04s |
| Laser_Heating_Calculator | 0.02s | 0.02s |
| Quantum_Measurement_Tutorial | 0.64s | 0.54s |
| EP_TPD_exploration | 0.29s | 0.28s |

**A second, more important limitation than the browser one above — read
before citing these numbers in a PR body.** Every page's number dropped, but
I do NOT trust this table as clean proof of `st.form`/`st.fragment` working —
here's why, worked through rather than assumed:

`st.fragment`'s and `st.form`'s whole point is that a live *session* defers a
rerun to just the fragment, or until submit. `AppTest.run()` has no notion of
a multi-turn session with that kind of deferral — it re-executes the script
fresh, top-to-bottom, every single call, using whatever widget values are
current at that instant. Checked this directly: EP_TPD_exploration's
`kappa_tilde_c` slider is now inside an `st.form`, but `at.slider[0].set_value(1.2); at.run()`
still re-executes the entire script immediately with the new value — nothing
about the measured call waited for a submit. So the recompute-after-change
number, for a page whose only change was wrapping inputs in a form (EP_TPD,
Dilution) or a fragment (Tutorial), *should* measure the same code path
running the same way, before and after — and yet all four dropped 30-45%. The
honest explanation is almost certainly **measurement noise, not the fix**:
these two table runs are separate process invocations several minutes apart,
Python's per-file bytecode cache (`__pycache__`) was cold for the "before"
run and warm by the "after" run (both tables' pages had been imported dozens
of times by other tests/py_compile calls in between), and general system
load varies run to run. A controlled A/B (same process, alternating
before/after code via git worktrees, many repeated trials, discarding the
bytecode-cache confound) would be needed to actually attribute a number to
these specific changes — not done here; flagging the gap rather than
presenting noise as a result.

**What IS actually verified, and how — read this as the real evidence, not
the table above:** (a) correctness — `tests/test_physics_regression.py`
calls the real compute functions with 3 fixed parameter sets each and
asserts their output against a captured-before-refactor reference, so the
form/fragment/cache changes provably did not change any answer; (b) the
mechanism itself — `st.fragment`'s and `st.form`'s deferred-rerun behavior is
Streamlit's own documented contract, not something this session invented,
and was checked functionally against a live `streamlit run` (not AppTest):
loaded Quantum_Measurement_Tutorial.py in a real browser, dragged a
Resonator-section slider via Playwright, and confirmed it updates correctly
with no errors and no unrelated section resetting. That confirms the
mechanism *works*, not precisely *by how much* — a real before/after
timing number for the live-session case would need a Playwright-driven
measurement against `streamlit run`, timing an actual widget drag in a real
browser tab. That's a heavier follow-up (comparable in scope to FE-14d's own
still-deferred "capture inside a live app" infrastructure on the game-repo
side), not done in this pass.
