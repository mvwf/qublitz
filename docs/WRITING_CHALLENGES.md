# Writing a challenge

Every explore page on the site can carry **challenges**: short goals that tick themselves off when the page reaches the right state.
Writing one is a good assignment for students at the Engineer or Researcher depth. You only get a challenge right once you understand
the physics well enough to say exactly what "done" looks like, and to explain it to someone a level below you.

This guide is for students (and anyone else) who want to add one.

## What a challenge is

A challenge is one entry in the list a page passes to `QL.challenges` (in `web/shared/outreach.js`):

```js
{ id: 'absorb',
  text: 'Switch to <b>Reflection</b> and make it <b>critically coupled</b> (Q<sub>i</sub> = Q<sub>c</sub>). How much is reflected on resonance?',
  hint: 'Pick Reflection, then the <b>Critically coupled</b> preset. On resonance S11 = 1 − 2Q<sub>l</sub>/Q<sub>c</sub> = 0 when Q<sub>i</sub> = Q<sub>c</sub>: every photon is absorbed.',
  check: () => ex.g === 'reflection' && Math.abs(Math.log10(Qi / Qc)) < 0.05,
  depth: 'student+' }
```

| Field | What it is |
|---|---|
| `id` | Lowercase letters, digits and hyphens, at most 32 characters. Never rename it later, because usage counts are grouped by it. |
| `text` | The goal, ending in a question that makes the player look at the result. One or two sentences. Basic HTML (`<b>`, `<sub>`) is fine. |
| `hint` | Shown after 40 seconds stuck. It says how to do it **and why the result comes out as it does**, with the equation that explains it. |
| `check()` | Returns `true` once the goal is met. It reads the page's own state (variables, sliders, results), never the screen. |
| `depth` | Optional: who sees it. `'curious'`, `'student+'`, `'engineer+'`, `'-student'` and so on (see the Depth section of `outreach.js`). Leave it out for everyone. |

Challenges tick off in order, and the page remembers finished ones in the player's browser only.

## The rules

1. **The default state ticks nothing.** Load the page fresh and no challenge may already be done.
2. **Check the physics with the page's own model.** Every number in `text` and `hint` must come from the simulation the page runs. Compute
   it with the page's engine (most pages have one in `web/.../*-engine.js` that Node can `require`) and say how in your pull request.
3. **One idea per challenge.** If you need "and" twice, it is two challenges.
4. **Ask a question the result answers.** "Make it critically coupled" is a task; "How much is reflected on resonance?" makes them think.
5. **The hint explains, not just instructs.** "Click X" alone teaches nothing. Add the one-line reason.
6. **Say what the model leaves out** if your challenge depends on it. For example, the Virtual Qubit Lab's T₁ is far shorter than a real transmon's.
7. **Nothing typed, nothing personal.** Challenges never collect text or anything about the player (see `web/privacy.html`).

## How to submit one

1. Fork <https://github.com/mvwf/qublitz> and find the page's `QL.challenges([...])` list (search the page's `.html` file).
2. Add your challenge where it fits the order, from easiest to hardest.
3. Run the page locally (`cd web && python3 -m http.server`), load it fresh, and check that your challenge starts unticked, ticks when it
   should, and shows its hint.
4. Run the tests (`node web/shared/tests/outreach.test.js`, plus the page's engine test if it has one).
5. Open a pull request against `main`. Include: what the challenge teaches, the depth it is for, and how you checked every number.

A maintainer checks the physics before merging.

## Ideas to get started

* **Virtual Qubit Lab:** "Lower T₂ in free play until the Ramsey fringes die out before 100 ns. What is the pure dephasing rate?"
* **Fridge Noise Explorer:** "Find the cheapest attenuation split that keeps the MXC below 10⁻³ thermal photons."
* **ErrorSweeper:** "Build a code in the lab with k = 2 and d = 3 on at most 13 qubits."
* **Spectrogram (Doppler):** "Find a plane whose pitch drops by more than 30 % as it passes. How fast is it?"
