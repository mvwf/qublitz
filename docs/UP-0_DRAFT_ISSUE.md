# UP-0 — draft GitHub issue for `mvwf/qublitz` (NOT POSTED)

*Status: drafted 2026-07-11, held for explicit user approval before posting. Per this project's
own hard constraints, no outward-facing action on a repo we don't own happens without an
explicit go-ahead. Nothing below has been sent anywhere — this file is the review artifact.*

Once approved, this becomes the body of a new issue on `mvwf/qublitz`, titled below.

---

**Issue title:** Proposal: a front door for the lab's tools (landing rebuild + a few small identity questions)

**Body:**

Hi — I've been working on the QuBlitz Arena page and, while auditing the rest of the site for
the same latency/visualization pass, noticed the other nine tools don't have a shared entry
point. `home.py` today is two logos, a link to the FitzLab site, and ten undescribed page links
— a first-time visitor has to guess which tool does what, and the VPN-only Custom Qubit Query
tool has no warning before it hangs for anyone off-campus.

I'd like your sign-off on a few things before touching anything identity-facing (page contract /
perf work on the tool pages themselves is already underway on its own branch and doesn't need
this issue — this is specifically about the *front door* and the lab's public-facing choices).

**Proposed, low-risk (page contract — already built, screenshots below):**
- Every page gets a consistent header: page title (so browser tabs are readable), an icon, a
  one-line "what is this" blurb, and a group label (Learn / Play / Research tools).
- `Custom_Qubit_Query.py` gets a warning *before* the VPN check, not just after it fails.

**Proposed, needs your call before I build it:**
1. **Landing page rebuild (`home.py`).** A real hero line ("Simulate real quantum hardware in
   your browser — the same open-quantum-systems physics our lab runs, no install"), the nine
   tools grouped by what a student would actually be trying to do (Learn / Play / Research), one
   real-physics sentence per tool (not generic copy), a "New here? Start with the Qubit
   Simulator" pointer, and the FitzLab link moved to a normal footer line instead of the current
   giant top-of-page link *away* from the site. No page renames, no URL changes, no
   `st.navigation` restructure — existing links/bookmarks keep working.
2. **A platform theme.** Right now the site is stock light-mode Streamlit hosting one page
   (the Arena) with its own dark neon identity. Happy to propose a light color pass reconciling
   the two, entirely revertable if it's not to your taste — I won't ship this without seeing it
   approved here first.
3. **The Custom Qubit Query naming.** Its current label is "Custom Qubit Query (Local/VPN)" —
   its URL may already be in a syllabus somewhere, so any rename/badge/gating call is yours, not
   mine to make unilaterally.
4. **A couple of long page titles/labels** (e.g. the EP/TPD page's link label) could stand to be
   shortened for the browser tab / sidebar — happy to do this as part of the page-contract PR if
   it's welcome.
5. **Dependency changes:** none needed beyond what's already in `requirements.txt` — turned out
   to be a non-issue once I dug in (see the linked PR for specifics), so nothing to approve here.

No pressure on timeline — a 👍/👎/comment on any of the numbered items above is enough for me to
know what to build next. Happy to open a small PR with mock screenshots of the landing rebuild
first if that's easier to react to than prose.

---

## Reviewer notes (not part of the issue body)

- Item 5 in the original task spec ("any dependency changes beyond lazy imports") is now N/A:
  UP-5 turned out to need zero new dependencies — `librosa`/`soundfile` were already declared in
  `requirements.txt`, just never imported anywhere; wiring them up used what was already there.
- Mock screenshots for item 1 (the landing rebuild) are **not** included in this draft — building
  even a mockup starts making the same identity calls this issue exists to ask permission for
  first. Recommended next step if this issue is approved: build UP-2 for real on its own branch
  (`platform-home`, per `HANDOFF.md`'s PR-E), then attach real before/after screenshots to the
  issue or the PR, whichever the maintainer prefers.
- The page-contract work (UP-1) is already built, tested, and committed on `platform-page-contract`
  — genuinely doesn't need this issue's answer, since it's inside the already-agreed
  "visualization improvements" mandate and touches no identity/copy decisions. Listed above only
  for context, not as something gated on this issue.