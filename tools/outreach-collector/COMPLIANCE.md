# Privacy notes

Short version: the collector stores anonymous usage statistics only, and needs no special sign-off for improving
the games or quoting usage numbers in proposals. One case needs a quick check with Dartmouth's IRB: publishing
research findings from the data. (My reading of the published policies, not legal advice.)

## Why no sign-off is needed for normal use

- **It is less than Dartmouth's own websites collect.** The
  [Privacy Policy for Dartmouth Websites](https://policies.dartmouth.edu/policy/privacy-policy-dartmouth-websites)
  says Dartmouth sites log visitors' IP addresses for usage statistics, may use Google Analytics with cookies, and
  do not respond to Do Not Track. This collector stores no IP addresses, no cookies or identifiers, no text and no
  time finer than the day, and it honours Do Not Track and Global Privacy Control.
- **Program evaluation is not human-subjects research.** CPHS's
  [protocol guide](https://www.dartmouth.edu/cphs/docs/cphsprotocolguide.pdf) defines it as work involving
  interaction with people or identifiable private information *and* designed to contribute to generalizable
  knowledge. Improving the games, the dashboard, and usage numbers in proposals and reports are internal program
  evaluation, which Dartmouth's IRB overview lists as not generalizable.
- **Children and students:** nothing collected can identify anyone, a visitor cannot type anything that is sent,
  and the Virtual Qubit Lab never sends ENGS 53 students' qubit codes.

## When to check with CPHS

If you plan to **publish research** from the data (for example a paper on what players learned), contact CPHS
before collecting data for it and file their short
[Not Human Subjects Research application](https://www.dartmouth.edu/cphs/tosubmit/forms/).
[CPHS-determination-draft.md](CPHS-determination-draft.md) is a ready-made description to adapt.

## Good habits (optional)

- Keep the Sheet in a Dartmouth Google account and don't share it widely.
- Use neutral `?event=` codes (`SCIFAIR26`), never a person's name.
- For school visits, [organizer-note.md](organizer-note.md) is a ready-made heads-up for the teacher.
- The public [privacy page](../../web/privacy.html) describes all of this to visitors; update its date if the
  collection changes.

## Keep it anonymous

These rules are what keep the data anonymous; they are also enforced in code. Revisit this page before adding
any of: a text box that is sent, a cookie or persistent identifier, a timestamp finer than the day, location, a
third-party analytics service, or a way to link records to a particular visitor.
