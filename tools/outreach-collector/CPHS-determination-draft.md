# Draft: request for a CPHS determination

*A starting point for Dartmouth's Committee for the Protection of Human Subjects. Edit the bracketed parts, and
attach `web/privacy.html` and screenshots of a debrief and an in-game question.*

**Title.** Anonymous usage and learning-check counts for the Qublitz interactive physics demos.

**Investigator.** Mattias Fitzpatrick, [title, department], Dartmouth. [Co-investigators, if any.]

**Purpose.** The Qublitz demos (https://mvwf.github.io/qublitz/) are free browser activities on qubits and waves,
used in Dartmouth courses (ENGS 23, ENGS 53) and at public outreach events. We want to know which activities are used,
where people stop, which ideas they get wrong, and whether they find the activities fun and useful, so we can improve
them. We would also like to report aggregate use (visits, time spent, share of correct answers) in grant proposals and
reports. [State whether you might publish learning outcomes in a research article; if so, say so here.]

**Procedures.** No change to how anyone uses the demos. While the collection is on, a page sends anonymous records to a
Google Sheet owned by the lab:
1. activity counts: the demo's name, the event (opened, level started or solved, left after N minutes and where),
   the level or step, a number such as stars, the day (no time of day), and touch screen or not;
2. optional multiple-choice questions shown inside the activities and in an end-of-activity "debrief": the question
   id, the choice picked, whether it was right; and two optional 1–5 ratings ("how fun", "how much did you learn").

There is no free text, no account, no cookie, no persistent or device identifier, and no IP address is stored (the
Apps Script does not receive it). Records cannot be linked to each other or to a person. An optional session label
(`?event=CODE`) chosen by an event organiser groups records from one session. Visitors can opt out on any page; Do Not
Track and Global Privacy Control are honoured; offline copies never send anything. Participation in the questions and
ratings is voluntary and has no effect on anything else.

**Participants.** Anyone who opens the demos: Dartmouth students, members of the public, and school groups at outreach
events, so some users may be **minors**, including children under 13. No one is recruited and no one is identifiable.

**Risks.** Minimal. No identifiable information is collected or stored, so a data breach could not reveal anything
about a person. The questions are about physics concepts, not about the person.

**Determination requested.** We believe this is program evaluation that does not involve identifiable private
information, and ask CPHS to confirm it is **not human subjects research**. If CPHS considers it research (for example
because we may publish learning outcomes), we ask which exemption applies; note that the multiple-choice items are
educational tests, while the two ratings are survey items, and that some users may be minors.

**Data handling.** Sheet in a Dartmouth Google account, accessible only to the investigator(s). Only aggregate totals
and averages are shown publicly or reported. Code: `web/shared/outreach.js` and
`tools/outreach-collector/Code.gs` in https://github.com/mvwf/qublitz.
