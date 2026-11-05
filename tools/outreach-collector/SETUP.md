# Set up the outreach collector (about 15 minutes)

You need a Dartmouth Google account and push access to this repository. Until step 6, nothing is collected.

## 1. Make the Sheet
Go to [sheets.new](https://sheets.new) while signed in to your Dartmouth account. Name it **Qublitz outreach**.
Don't share it with anyone.

## 2. Add the script
In the Sheet: **Extensions → Apps Script**. Delete everything in `Code.gs`, paste in the whole of
[`Code.gs`](Code.gs) from this folder, and press **Save** (the disk icon).

## 3. Publish it as a web app
**Deploy → New deployment →** the gear next to "Select type" **→ Web app**.
- Description: `collector`
- Execute as: **Me**
- Who has access: **Anyone** (pages post anonymously, without signing in)

Press **Deploy**, then **Authorize access**, choose your account, and allow it. (Google shows an "unverified
app" warning for your own scripts: **Advanced → Go to … (unsafe)** is expected here, since you wrote it.)
Copy the **Web app URL**: it ends in `/exec`.

## 4. Turn on the clean-up
Back in the Sheet, reload the page. A **Qublitz outreach** menu appears. Choose **Install daily retention
clean-up** once (allow the permission it asks for).

## 5. Check the URL works
Open the `/exec` URL in a browser: it should say `outreach collector: ok`. Add `?summary=1` to the end:
you should see a short JSON summary with empty counts.

## 6. Switch collection on
Paste the URL into [`web/shared/outreach-config.js`](../../web/shared/outreach-config.js) as `endpoint`, in a pull
request (or send it to Claude to do). Once it's merged and the site has deployed (a minute or two):
- the bottom of every demo says it sends anonymous counts, with **Opt out** and **Privacy** links;
- [the privacy page](https://mvwf.github.io/qublitz/privacy.html) says "Usage counting is on";
- the [dashboard](https://mvwf.github.io/qublitz/outreach/dashboard.html) starts filling in (refresh after
  playing a level).

The Sheet fills in by itself: tabs `events` and `quiz` appear with the first data.

## Day to day
- **Outreach sessions:** hand out links with `?event=SOMECODE` on the end (for example
  `https://mvwf.github.io/qublitz/?event=SCIFAIR26`) to see each session separately. Use neutral codes, never
  a person's name.
- **Look at the numbers:** the dashboard (Report tab for proposals; Redesign & compare for improving a game).
- **Ask an AI about it:** in the Sheet, **Qublitz outreach → Export for AI**. Two files appear in your Drive:
  see [AI-GUIDE.md](AI-GUIDE.md).
- **Updating the games** needs nothing here: new games, new events and new fields are stored automatically,
  and every record is stamped with its page's version.
- **Updating the collector** (when `Code.gs` changes in the repo): paste the new code in Apps Script, then
  **Deploy → Manage deployments → ✏ Edit → Version: New version → Deploy**. The URL stays the same.
- **To stop:** empty `endpoint` again (pages send nothing), and/or archive the deployment.
