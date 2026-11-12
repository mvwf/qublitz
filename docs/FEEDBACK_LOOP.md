# The feedback loop: from what players say to a merged change, and back

Qublitz and [Coherence Vault](https://github.com/mvwf/MARQUIS_Q_VAULT/blob/main/docs/FEEDBACK_LOOP.md) share one feedback
loop, so one weekly routine covers both sites. Feedback is collected on the page where it happens, ranked once a week,
turned into small pull requests, and checked afterwards against the numbers that prompted it.

**Every change made because of feedback names the evidence behind it, and the dashboard shows whether it helped.**

## 1. Collect, with context

The outreach kit (`web/shared/outreach.js`) adds "Was this page useful?" to the footer of every page.

* **Yes or No** is an anonymous event of that page (`helpful_yes`, `helpful_no`), counted like every other event and
  listed on the [privacy page](https://mvwf.github.io/qublitz/privacy.html). Nothing typed is sent.
* **Suggest a change** opens the public Feedback issue form (`.github/ISSUE_TEMPLATE/feedback.yml`) with the page's
  link, including its settings, and the page version filled in. Email is offered beside it.
* **Script errors** count once per page load as `js_error`. The message never leaves the browser.

Questions, levels, ratings and stops keep flowing to the [outreach dashboard](https://mvwf.github.io/qublitz/outreach/dashboard.html) as before.

## 2. Rank, once a week

`scripts/feedback_digest.js` reads the collector's summary, the open issues and `web/outreach/improvements.json`.
It flags counts with the dashboard's own rules (`web/outreach/insights.js`), adds script errors and pages marked not
useful, and ranks everything by impact, with open issues first. The [feedback-digest workflow](../.github/workflows/feedback-digest.yml)
runs it every Monday and rewrites one open issue labelled `feedback-digest`, so digests never pile up.

```bash
node scripts/feedback_digest.js --days 30                 # markdown to the terminal
node web/shared/tests/feedback-digest.test.js             # the ranking, checked on made-up data
```

## 3. Act, in small pull requests

In Claude Code, `/improve` reads this week's digests for both sites, proposes the few changes most likely to help, and
after a yes makes each one as its own pull request whose body starts with `Feedback:` and the issue or flag it answers.
The PI reviews and merges.

## 4. Close the loop

Each merged change gets one entry in `web/outreach/improvements.json` (date, game, change, because, pr). The dashboard
lists it and compares the numbers before and after, and the digest shows it in the following weeks. The person who
opened the issue gets a reply linking the merged pull request.

**A change is finished when the number that prompted it has moved, or when the person who asked has their answer.**
