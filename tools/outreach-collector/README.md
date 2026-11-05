# Outreach collector

A Google Apps Script web app, bound to a Google Sheet you own, that receives the anonymous usage counts and quiz
answers sent by `web/shared/outreach.js` and serves aggregate counts to `web/outreach/dashboard.html`.

## What it stores

| Tab | One row per | Columns |
|---|---|---|
| `events` | page event (open, level solved, left after N s, ...) | day, game, event, level, value, stars, dur_s, code, device |
| `quiz` | submitted quiz | day, game, code, device, score, n, fun (1–5), learned (1–5), answers (question id, objective, choice, right/wrong) |
| `feedback` | quiz with free text | day, game, code, "what did you learn", "what was confusing" |

No cookies, no identifiers, no IP addresses (Apps Script never sees them), and nothing finer than the day.
Free text is scrubbed of email addresses, phone numbers and links in the browser *and* again here, and it lives
only in the `feedback` tab: the summary endpoint returns counts, never the text. `code` is an optional
outreach-session label from links like `…/beamline.html?event=SCIFAIR26`.

## Set it up (about 10 minutes)

1. Create a Google Sheet (a Dartmouth account is fine). Name it e.g. *Qublitz outreach*.
2. **Extensions → Apps Script.** Replace the contents of `Code.gs` with this folder's `Code.gs`. Save.
3. **Deploy → New deployment → Web app.** Execute as: *Me*. Who has access: *Anyone*. Deploy, authorise,
   and copy the **Web app URL** (ends in `/exec`).
4. Paste that URL into `web/shared/outreach-config.js` as `endpoint`, commit, and let Pages deploy.
5. Open `https://mvwf.github.io/qublitz/outreach/dashboard.html`. It reads the same URL's `?summary=1`.

The tabs and their header rows are created on the first request. To change the code later, edit it in Apps Script
and use **Deploy → Manage deployments → Edit → New version** so the URL stays the same.

## Things to know

- Anyone can POST to a public web app. The collector validates every field and caps sizes, but counts could be
  inflated by someone determined; treat them as indicative, and use `?event=` codes for sessions you run.
- If you plan to publish anything from the quiz answers, ask Dartmouth's CPHS (IRB) for a determination first.
  The data are anonymous, but outreach audiences can include minors.
- To stop collecting, empty `endpoint` in `outreach-config.js` (pages then send nothing) and/or archive the
  deployment in Apps Script.
- The pure functions (`cleanEventRow`, `cleanQuizRows`, `summarize`) are tested in Node by
  `web/shared/tests/outreach.test.js`.
