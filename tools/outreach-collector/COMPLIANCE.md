# Before switching collection on

A checklist for the lab. This is practical guidance written alongside the code, **not legal advice**: Dartmouth's
CPHS (IRB) and Office of General Counsel / privacy office have the final word. Collection stays off until
`endpoint` is set in `web/shared/outreach-config.js`; tick every box first.

- [ ] **CPHS determination.** Submit [CPHS-determination-draft.md](CPHS-determination-draft.md) (edit it first) and
      keep the determination letter. Needed before anything from the data is used in research or a publication;
      advisable even for program evaluation, because audiences include minors.
- [ ] **Privacy notice reviewed.** Ask Dartmouth's privacy office or Office of General Counsel to review
      [`web/privacy.html`](../../web/privacy.html). Update its "last updated" date with any change.
- [ ] **Storage.** The Sheet and the Apps Script live in a Dartmouth Google account, shared with no one who doesn't
      need them. The web app runs as *Me* with access *Anyone* (needed for anonymous posts).
- [ ] **Retention trigger.** Run `setupRetention` once in the Apps Script editor (see the README).
- [ ] **Event organisers informed.** For school groups and outreach sessions, send the organiser the note in
      [organizer-note.md](organizer-note.md) beforehand. Use neutral session codes (`SCIFAIR26`, `OUTREACH-03`), never a
      person's name; avoid naming a small group that could identify individuals.
- [ ] **Turn it on.** Paste the `/exec` URL into `outreach-config.js`, and write the date here: ____________.

## Why the design is low risk

| Framework | What it covers | How the design stays clear of it |
|---|---|---|
| COPPA (US) | Personal information collected online from children under 13 | No personal information at all: no names, emails, accounts, cookies, device or persistent identifiers, IP storage or free text. (Dartmouth, a nonprofit, may fall outside COPPA itself, but schools expect this level of care.) |
| FERPA and state student-privacy laws | Education records about identifiable students | Nothing identifies a student. The Virtual Qubit Lab's per-student qubit codes are never sent (checked in `qubit-simulator.html`: events carry only generic labels such as "code entered"). |
| GDPR / UK GDPR | Personal data of people in the EU/UK | Records are anonymous and stand alone (no identifier links them), with no free text. Google receives the IP address in transit as for any web request; the script never sees or stores it. |
| Human-subjects research (45 CFR 46) | Research involving living people | Program evaluation to improve the demos is the stated purpose. Publishing learning outcomes needs a CPHS determination first. |

## Rules for future changes

Repeat this checklist, and update `web/privacy.html` first, before adding any of: a text box that is sent, a cookie or
any persistent identifier, a timestamp finer than the day, location, a third-party analytics service, or a link from
a record to a particular visitor. Those are what would turn anonymous counts into personal data.
