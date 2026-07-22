# Page contract

Every page under `pages/` starts with `utils.ui.page_header(title, icon, blurb, group)`
as its **first** Streamlit call — same rule `st.set_page_config` itself has, since
`page_header()` calls that internally. It sets the tab title/icon and renders a
consistent header row (logo + page name + a one-sentence blurb naming the actual
physics on that page), so every page reads as the same site instead of ten
independent scripts.

Why this exists: before it, 4 of 9 pages called `st.set_page_config` ad hoc and the
other 5 didn't call it at all — no shared header, inconsistent tab titles, at least
one page whose internal title was copy-pasted from a different page and never
updated (`Qubit_Simulator.py` titled itself "Custom Qubit Query"). An unenforced
convention decays back to this the moment the person who wrote it graduates —
`page_header()` is the cheap fix; a real page-contract test (asserting every page
under `pages/` calls it) is the natural follow-up once this file exists to test
against.

Adding a new page: call `page_header()` first, pick the most fitting `group`
(`"Learn"`, `"Play"`, or `"Research tools"` — matches the landing page's own
proposed grouping, HANDOFF.md's UP-2), and write a blurb that names the specific
physics/action on the page, not a generic description.
