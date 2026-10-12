### Overview screen (`console/overview.py`, `GET /`)

`GET /` used to be a bare `<ul>` of the eleven nav entries -- where things live, not what
needs an operator. `overview.build_overview()` (called by the `index` route, never
duplicated in the template) reads three existing sources and renders three sections:
**Needs attention** (jobs `failed`/`interrupted`, via `jobs.list_jobs`; deliveries
`failed_permanent`/`awaiting_manual_publish`, via `pubqueue.queue_rows` -- the same
vocabulary `notify.check_queue_regressions` watches, read directly rather than through that
function, which POSTs a webhook and a GET must never trigger), **Jobs in progress** (jobs
`queued`/`running`), and **Deliveries coming up** (the next `UPCOMING_LIMIT` non-terminal deliveries
with a `scheduled_at`, soonest first). Each section is `{"rows": [...], "unreadable":
[...]}`: an empty `rows` with an empty `unreadable` renders as a plain "nothing here" -- the
template never infers that state, it reads it. A source that raises `sqlite3.Error` is
named in `unreadable` instead, so a read failure can never look like an empty, healthy
section. The nav list itself still renders below the three sections, unchanged.
