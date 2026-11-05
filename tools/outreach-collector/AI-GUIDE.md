# Using the outreach data with an AI

The collector is deliberately loose: games can log new events and their own fields at any time, and every record
says which version of the game sent it. That makes the export a small, semi-structured dataset an AI can read
directly and reason about, to find where games lose people, which ideas they get wrong, and what to change next.

## Get the data
In the Sheet: **Qublitz outreach → Export for AI**. Two files appear in My Drive:
- `qublitz-outreach-YYYY-MM-DD.jsonl`: one JSON object per line, from the `events` and `quiz` tabs;
- `qublitz-outreach-YYYY-MM-DD-README.md`: the field dictionary below, for the AI.

Give the AI those two files plus, from this repository, `web/learning/*.js` (each game's learning objectives, how
to play, goals and question text) and, for a deeper look at one game, its page (e.g. `web/qublitz/beamline.html`).

## What a record looks like
```json
{"table":"events","day":"2026-11-12","game":"beamline","event":"level_complete","level":"half","stars":3,"device":"desktop","gv":"2026-11-05+addfe255"}
{"table":"events","day":"2026-11-12","game":"beamline","event":"fire","level":"half","value":4,"device":"desktop","gv":"2026-11-05+addfe255","extra":{"theta_pi":0.5,"beam":"x"}}
{"table":"events","day":"2026-11-12","game":"rider","event":"leave","level":"photon|playing","value":2,"dur_s":312,"code":"SCIFAIR26","device":"touch","gv":"2026-11-05+7a33eb94"}
{"table":"quiz","day":"2026-11-12","game":"copter","device":"desktop","score":1,"n":1,"answers":[{"q":"y90","obj":"pulse","pick":2,"ok":true}],"source":"ingame","gv":"2026-11-05+6930b5ef"}
```

| Field | Meaning |
|---|---|
| `table` | `events` (something happened) or `quiz` (questions answered, in the game or in the end-of-activity debrief) |
| `day` | the visitor's local date, `YYYY-MM-DD`; there is never a time |
| `game` | page id: `qubit-lab`, `copter`, `rider`, `beamline`, `arena`, `fridge`, `resonator`, `ep-tpd`, `trio`, `string`, `drum`, `spectrogram`, `sonify`, `hub` (see `web/learning/manifest.js`) |
| `event` | what happened, lower_snake. Common ones: `open`, `start`, `mode`, `level_start`, `level_complete`, `win`, `lose`, `hint`, `reset`, `run`, `quiz_open`, `to_lab` (clicked into the Virtual Qubit Lab), `leave` (the visit ended). Games add their own |
| `level` | the level, mission, lesson, guided step or place; `a\|b` means level a, step b. On `open` it may be `from:<game>` (arrived from that game) |
| `value` | a number whose meaning depends on the event: score, stars, P at measure, fire count; on `leave`, levels completed in that visit |
| `stars` | 0–3 on `level_complete` |
| `dur_s` | active seconds on the page, on `leave` (time with the tab hidden is not counted) |
| `code` | outreach-session label from `?event=` links |
| `device` | `desktop` or `touch` |
| `gv` | page version: date and commit of the last change to that page's files. Records with different `gv` came from different versions of the game; `git show <commit>` says what changed |
| `extra` | the game's own fields (numbers, true/false, one-word labels) |
| `score`, `n` | questions right / answered in that record |
| `fun`, `learned` | 1–5 ratings from the debrief (may be missing) |
| `answers` | `{q: question id, obj: objective id, pick: choice index, ok: right?}`; question and choice text are in `web/learning/<game>.js` |
| `source` | `ingame` (answered inside the game: a ring, a target, a prediction) or `debrief` |

There are no identifiers: records cannot be linked to a person, and records from one visit are not linked to
each other except by sharing a day, game and version. Count visits by `open` events and visit length by `leave`.

## A prompt to start from
> You are helping improve free educational physics games. Attached: a JSONL export of anonymous usage records,
> its field dictionary, and each game's learning file (objectives, how to play, questions with choices in index
> order). For each game: (1) where do visits end (the `level` of `leave` events) compared with where people start,
> and which places lose an unusually large share; (2) which learning objectives have the lowest share of right
> answers, and which wrong choice is picked most (a likely misconception); (3) how fun and learned ratings compare;
> (4) for games with more than one `gv`, whether the numbers changed between versions, with counts on both sides
> and a note when there are too few to tell. Then propose the three changes most likely to help people progress
> and learn, each with the evidence, the expected effect and how to check it in the next export. Don't speculate
> about individuals; there is no information about them.

## Logging more from a game
Games can send anything useful with `QL.event(name, { level, value, ...yourFields })`: for example
`QL.event('fire', { level: 'half', theta_pi: 0.5, beam: 'x' })`. Rules, enforced in the browser and again by the
collector: up to 12 extra fields, names in lower_snake, values a number, true/false or a single word (letters,
digits, `_ . : | + -`, no spaces, at most 40 characters). Never log anything a visitor typed or chose to name.
No collector change or redeploy is needed.
