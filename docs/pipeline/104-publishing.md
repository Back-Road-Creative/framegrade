### Publishing (`console/pubqueue.py`, `GET /publishing`, `POST /publishing/enqueue`,
`POST /publishing/drain`)

Metricool-exit control surface fronting the delivery queue (`publishing/queue.py`) -- never
re-implements enqueue/drain. `GET /publishing` reads every `deliveries` row read-only
(`config.seo.storage_path`, the CLI's own path) and groups them **attention-first** (`
awaiting_manual_publish` -> `queued` -> `submitting` -> `uploaded` -> `published` ->
`failed_permanent`, then anything else); a missing DB file is the same empty state as an empty
queue and is never created by viewing the tab. The two POST routes build argv for the real
`enqueue-plan`/`dispatch-queue` subcommands, field-for-field against `cli/parser.py`, and launch
through `console/jobs.py`'s allowlisted runner.

**The paths on both POST routes are contained the same way every other path-taking console
route is.** `state_path`, `output_dir` (enqueue) and `sidecar_dir` (drain) arrive as free text in a
form field and used to reach the launched argv untouched, so a console operator could enqueue a plan
file belonging to a different deployment just by typing where it lives -- the last unguarded routes
after the drive-route checks. Directories go through the same `drives.resolve_allowed_output_dir()` those routes
use. `state_path` is a FILE at a depth the console cannot predict (`<out>/pipeline_state.json` and
`<out>/<project>/pipeline_state.json` both occur), so it goes through a sibling,
`drives.resolve_allowed_run_file()`, which asks whether the resolved path is *inside* an allowed
output dir rather than equal to one; resolving both sides first is what stops a `..` segment or a
symlink walking out of an allowed run. Each route launches the **resolved** path, not the submitted
string, and a non-match is a 404 whose body never echoes what was submitted, so the response cannot be
used to probe which paths exist. An empty `output_dir`/`sidecar_dir` stays empty and is not a refusal: the CLI then omits the
flag and uses its own default, so there is nothing from the request to contain. The run file itself
is never stat-ed -- a missing plan file fails honestly in the job that opens it, and pre-checking
here would leak which paths exist.

**Ordering: `awaiting_manual_publish` sorts first, `failed_permanent` stays last.** It is
the only status a drain can never advance on its own, so it is the only row that strictly needs a
person. `failed_permanent` is NOT promoted alongside it: it is terminal and its error is already
inline in the row, unlike `awaiting_manual_publish`, which sits open until someone clicks
"mark published" right there. An unlisted status still sorts deterministically, alphabetically,
last.

**Bounded read.** The tab reads ONE page, `pubqueue.queue_page`, defaulting to the same
per-page size as the Media library. Both the `LIMIT`/`OFFSET` and the lifecycle `ORDER BY` are in
the SQL: fetching every row and slicing in Python would cost exactly what the unbounded read cost,
and ordering after the fetch would bury the state above -- `ORDER BY status` is alphabetical, so
`awaiting_manual_publish` would land behind a page of `cancelled`. The SQL sort key is GENERATED
from the same lifecycle tuple the grouping uses, so the two cannot drift, and an unlisted status
sorts last there as well. A page out of range lands on the nearest real page, because an operator
reads an empty table as an empty queue. The per-state counts in the headings are per-state within
the page; the pager line carries the queue-wide total and shows even on a single page, so "this is
the whole queue" is never confused with "this is the first page of it".

`pubqueue.queue_rows` stays unbounded and is no longer what the tab reads. Its remaining callers
are the observers in `console/notify.py`, whose job is to look at every row; a `LIMIT` there would
stop the failure webhook at the page size and call the rest healthy.

**`awaiting_manual_publish` (TikTok pack and inbox modes) is surfaced here.**
`handlers.make_tiktok_handler` defaults to the TikTok Studio pack
(`$GMS_TIKTOK_MODE`, default `"pack"` -- see `docs/social-publishing-tiktok.md` § Pack mode; `"inbox"`
is the sandbox upload), which never auto-publishes: once the file is in the pack (or, in inbox
mode, once the chunked upload completes), the delivery lands `awaiting_manual_publish`
instead of `published`, parking its own `next_attempt_at` far in the future so
`dispatch._select_due` never reselects it. The tab renders that group with a mode-dependent hint (pack mode: "Post the file from <pack directory>
(captions.txt), then confirm here.", where the directory is `$GMS_TIKTOK_PACK_DIR`, default `tiktok-pack` under the data home; inbox mode: "Publish from the TikTok app inbox, then confirm
here."; an unreadable `$GMS_TIKTOK_MODE`: a neutral line) and a mark-published button per row; `POST
/publishing/mark-published/{delivery_id}` calls `pubqueue.mark_published`, which transitions the
row to `"published"` via `queue.transition` -- 409 if the row isn't currently
`awaiting_manual_publish`, 404 for an unknown id.

**Per-platform pause is wired.** The tab used to render a static "not available yet" note
citing a missing platform filter; that note outlived the thing it described. `queue.pause_platform`
/ `resume_platform` exist and the drain's own `SELECT` excludes a paused platform, so the rule was
already enforced for every drain path -- the CLI, this page's own drain button, a future worker --
and the only thing missing was a way to say so. `POST /publishing/pause` takes a platform name and
a pause/resume action and calls `console/pubqueue.py`'s `set_platform_pause`, which turns the
queue's bare `False` (an unknown platform name has nothing to pause; a platform that was never
paused has nothing to resume) into a sentence rendered on this page at 400, the same refusal shape
every other console form now uses. The GET builds its context from `known_platforms()` and
`paused_platforms()`, so the form offers exactly the vocabulary the queue accepts -- a control the
queue would reject is never drawn -- and the refusal path re-READS that context rather than reusing
the one the submission was built against, because a refusal is exactly when the operator needs the
current pause list. Nothing here touches a `deliveries` row: a pause governs the next pickup, and
the platform call an in-flight row is mid-way through cannot be un-sent.

**Title/routes come from each row's own sidecar; the effective schedule is the row's own
`scheduled_at` column, never the sidecar.** `pubqueue.enrich_rows_with_sidecar` reads the
row's `sidecar_path` (`publishing/plan_io.load_publish_plan`, per-call cache keyed by path),
matches the short by `(project_name, short_index)`, and adds `scheduled_at_local`/`_utc`,
`scheduled_platforms`, and `title`. A missing `sidecar_path`/file/match leaves those `None`; a
malformed one additionally sets `sidecar_read_error = True` so the template says
"could not read sidecar" instead of implying there is no schedule. Never raises.

`scheduled_at_effective_local` is what the template shows: the row's own `scheduled_at`
(`publishing/queue.py`'s "the plan's publish time" column), plain UTC -- losing/moving the sidecar
can never render a correctly-scheduled delivery as unscheduled (the failure this prevents). No row
`scheduled_at` but a sidecar time -> show the sidecar's, nothing authoritative to prefer. Both
present and disagreeing -> `schedule_discrepancy` carries the sidecar's time separately, never
silently resolved. Neither present and no `sidecar_read_error` -> "no schedule recorded", distinct
from "could not read sidecar".

**Calendar reading (`console/pubqueue_calendar.py`, `GET /publishing/calendar`).** The same
queue rows as a Mon-Sun week grid instead of a status-grouped list, linked from the top of the list
view and linking back. `week_calendar` flattens `pubqueue.rows_by_state`'s groups back into one list
rather than re-enriching or re-reading the queue, so the calendar can never disagree with the list
about a row's effective schedule or publish-confirmation label. Each row's day bucket is read
verbatim from `scheduled_at_effective_local` -- the exact field the list view tests before printing
"no schedule recorded" -- so a delivery the list can't claim a time for is never placed on a day
here either; it lands in the "Not scheduled" group instead. The week is always all 7 days, each
rendered even when empty, so the grid never implies a day doesn't exist; `?week=YYYY-MM-DD` (a
Monday, from `prev_week`/`next_week`) pages to another week. Server-rendered, no new dependency; the
grid is a plain `<table>` so it picks up `_base.html`'s existing horizontal-scroll wrapping
without any per-page markup.
