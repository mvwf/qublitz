# Outreach collector

> **Start here: [SETUP.md](SETUP.md)**, a step-by-step guide (about 15 minutes). To analyse the data with an
> AI, see [AI-GUIDE.md](AI-GUIDE.md).
> Collection is off until `endpoint` is set in `web/shared/outreach-config.js`.

A Google Apps Script web app, bound to a Google Sheet you own, that receives the anonymous usage counts and quiz
answers sent by `web/shared/outreach.js` and serves aggregate counts to `web/outreach/dashboard.html`.

## What it stores

| Tab | One row per | Columns |
|---|---|---|
| `events` | page event (open, level solved, left after N s, ...) | day, game, event, level, value, stars, dur_s, code, device, gv (page version), extra (the game's own fields, JSON) |
| `quiz` | question answered in a game, or a debrief | day, game, code, device, score, n, fun (1–5), learned (1–5), answers (question id, objective, choice, right/wrong), source (in-game or debrief), gv |

No cookies, no identifiers, no IP addresses stored (Apps Script never sees them; Google's servers receive them as
part of any web request, under Dartmouth's Google agreement), and nothing finer than the day. **No free text**: the
pages have no text boxes, and the collector drops any text field a client might send, so nothing a visitor types,
such as a name, school or email, can end up in the Sheet. `code` is an optional outreach-session label from links
like `…/beamline.html?event=SCIFAIR26`. The privacy notice for visitors is [`web/privacy.html`](../../web/privacy.html).

## Set it up (about 10 minutes)

1. Create a Google Sheet (a Dartmouth account is fine). Name it e.g. *Qublitz outreach*.
2. **Extensions → Apps Script.** Replace the contents of `Code.gs` with this folder's `Code.gs`. Save.
3. **Deploy → New deployment → Web app.** Execute as: *Me*. Who has access: *Anyone*. Deploy, authorise,
   and copy the **Web app URL** (ends in `/exec`).
4. Paste that URL into `web/shared/outreach-config.js` as `endpoint`, commit, and let Pages deploy.
5. Open the dashboard, [`https://mvwf.github.io/qublitz/outreach/dashboard.html`](https://mvwf.github.io/qublitz/outreach/dashboard.html)
   (source: [`web/outreach/dashboard.html`](../../web/outreach/dashboard.html)). It reads the same URL's `?summary=1`.
6. Reload the Sheet and choose **Qublitz outreach → Install daily retention clean-up** once. It empties any
   leftover `feedback` rows (from an older version of this script) after 365 days; the current version writes none.
   **Share the Sheet with no one else** unless they need it, and keep it in a Dartmouth account.

The tabs and their header rows are created on the first request. The schema is open: new games, event names and
`extra` fields are stored without any change here, and new columns are only ever added at the end of a tab. To change the code later, edit it in Apps Script
and use **Deploy → Manage deployments → Edit → New version** so the URL stays the same.

## The dashboard

[`web/outreach/dashboard.html`](../../web/outreach/dashboard.html) is public but not linked from the site: it is reached
from the repository's README, and it is not indexed by search engines. It shows only what `?summary=1` returns: counts
and averages.
[`web/outreach/dashboard.html`](../../web/outreach/dashboard.html) is public but deliberately out of sight: it is reached
from the repository's [README](../../README.md#outreach-usage-dashboard), not from any page of the site, and search
engines are asked not to index it. It shows only what `?summary=1` returns: aggregate counts and averages.

- `?demo=1` shows realistic synthetic data, labelled **DEMO DATA**, so you can see it before collection starts.
- `?days=7|30|90|365` picks the period; `?since=YYYY-MM-DD` shows everything since a date (each entry in
  "You said, we did" has a button for this); `?endpoint=<url>` reads another collector deployment.
- With `endpoint` empty in `outreach-config.js` it shows a "collection is off" notice with the setup steps above.
- The **Insights** at the top are computed in the browser by [`web/outreach/insights.js`](../../web/outreach/insights.js),
  whose thresholds are constants with comments there (and listed on the page). They flag questions under 50% right,
  levels people rarely finish, places where many visits stop, ratings below 3 of 5, debriefs opened but not sent, and
  very short visits, each with its evidence and a suggested kind of fix, ranked by people measured × gap.
- The **Report** tab is for proposals and grant reports: a date range and optionally one event code, headline numbers
  (page visits, which are not unique people; activities used; active hours; levels completed; in-game answers and
  debriefs; share right by learning objective; ratings; lab arrivals; growth by month; one row per outreach event),
  "Copy as text" (a paragraph and bullets with the range and caveats), CSV downloads (games by month, events), the raw
  JSON, and a print stylesheet for "Print / Save as PDF".
- The **Redesign & compare** tab builds a markdown redesign brief for one game (stop places, weakest objectives with
  the most-picked wrong answer, ratings, minutes, flags and suggested experiments) and a before/after comparison for
  any `improvements.json` entry or date. `improvements.json` entries may add `"level"` or `"objective"` (ids) to
  name what the change targets.
- Question and objective text come from each page's `web/learning/<id>.js`; a missing file only means the dashboard
  shows question ids instead of text.

## Closing the loop

The point of the counts is to make the games more fun and more educational. The cycle:

1. **Review** the dashboard's Insights, weekly during term and after each outreach event (filter the Report by its
   `?event=` code).
2. **File a GitHub issue** labelled `feedback` for each thing worth changing, with the game, the evidence exactly as
   the dashboard states it (e.g. "question q3: 38% correct of 52 answers; possible misconception: …", or
   "level crosstalk: 9 completions of 31 starts"), and the proposed change. The **Redesign & compare** tab builds this
   as a brief for one game, with a button that opens the issue prefilled.
3. **Fix** it in a pull request that links the issue.
4. **Log the change** in [`web/outreach/improvements.json`](../../web/outreach/improvements.json), which the dashboard
   shows as "You said, we did":
   ```json
   [{ "date": "2026-11-20", "game": "beamline", "change": "Hint added to the crosstalk level", "because": "level crosstalk: 29% of starts completed" }]
   ```
   `date` is the day the change went live, `game` a manifest id, `because` the evidence type and number.
5. **Compare** the same metric afterwards: the entry's "Compare before / after" button shows the level, objective and
   ratings in equal windows before and after that date, with n on each side (greyed out until both sides have enough
   answers); "Show the numbers since …" sets `?since=` to that date. Give it enough visits to clear the rule's minimum
   count before calling it fixed, and note the result on the issue.

## Things to know

- Anyone can POST to a public web app. The collector validates every field and caps sizes, but counts could be
  inflated by someone determined; treat them as indicative, and use `?event=` codes for sessions you run.
- To stop collecting, empty `endpoint` in `outreach-config.js` (pages then send nothing) and/or archive the
  deployment in Apps Script.
- The pure functions (`cleanEventRow`, `cleanQuizRows`, `summarize`) and the dashboard's
  insight rules are tested in Node by `web/shared/tests/outreach.test.js`.
- Sheets turns a cell like `2026-11-05` into a date; `summarize()` reads such cells back as the same day.
