### Failure webhooks (`console/notify.py`, `GMS_CONSOLE_NOTIFY_URL`)

**URL source and phone mode.** The URL is `GMS_CONSOLE_NOTIFY_URL`, else the first line of
`notify_url` under the config home (`$GMS_CONFIG_HOME`, else `$XDG_CONFIG_HOME/gomoveshift`, else
`~/.config/gomoveshift`); neither set is a silent no-op. When its host is `NTFY_HOST`
(`ntfy.sh`, on the egress allowlist, `docs/pipeline/84-network-allowlist-credential-stores.md`) the event becomes a phone push: JSON to
`https://ntfy.sh/`, `topic` = the URL's first path segment, plain-words title and body in
America/New_York. Any other URL keeps the JSON webhook below unchanged. The URL and topic are a
password, never logged (a failed push logs only the exception type). Messages: a TikTok inbox
draft (slot, the Inbox steps, the caption from the row's plan sidecar -- `payloads.tiktok`
caption, else title; priority 4); a failed post (error text redacted of `access_token=`/`Bearer`/
`OAuth` strings, cut near 300 characters); a missed slot; job, gate and console events too. Setup:
`python3 tools/phone_alerts_setup.py` (stdlib only; `--test`, `--rotate`).

Failures used to be discovered (someone opens a tab), never pushed. `notify.notify(event,
identity, status)` POSTs `{event, **identity, status, timestamp, console_url}` as JSON to
the notify URL with a 5s timeout; unset is a silent no-op, and a notify failure
(network, timeout, bad status) is logged and swallowed -- it must never fail a job or a queue
scan. `GMS_CONSOLE_URL` (optional) is echoed back as `console_url` so a webhook receiver can
link straight to the instance. Two hooks:

- **Job failure** -- `jobs._reap()` calls `notify.notify_job_failed()` once, right when a job's
  `rc` is first observed nonzero (never on a later reap of the same row). `jobs.reap_finished()`
  is the shared entry point `get_job`/`list_jobs` (pull) and the `Ticker` (push, each tick) both
  call to trigger it.
- **Job interrupted** -- `jobs.sweep_orphaned_jobs()` (startup, before anything else reads the
  jobs table) calls `notify.notify_job_interrupted()` for each DB row it finds stuck `running`
  with no live in-process handle -- a console restart mid-job, otherwise silent until someone
  opens the tab.
- **Console errors** -- the app's catch-all `Exception` handler now logs (`logger.exception`) and
  calls `notify.notify("console_error", {path, exc_type}, str(exc))` before rendering the usual
  500 page; previously an unhandled exception was swallowed with no log line and no notify.
- **Queue regressions** -- `notify.check_queue_regressions()` reads the delivery queue read-only
  via `pubqueue.queue_rows()` (never touches `publishing/**` directly) for rows newly in
  `failed_permanent` or `awaiting_manual_publish`, tracking last-seen `(id -> status)` in
  `<GMS_CONSOLE_DATA_DIR>/notify_seen_queue.json` so a repeat sighting never re-notifies. The
  `Ticker` above calls it once per tick, right after `run_due`, and `dispatch-queue --apply`
  (however it is scheduled) calls it after the drain as an observer: it never fails the run, changes its
  exit code or displaces the summary line. **First use seeds silently:** with no seen file the scan
  records rows already in a watched status without notifying, so enabling alerts is no flood. A row
  is recorded only once its push went out, and forgotten when it leaves a watched status (a requeued
  row that fails again alerts again).
- **Pending review gates** -- `notify.check_pending_gates(output_dirs)` reads
  `drives.list_pending_gates()` and notifies `gate_pending` once per question file, plus
  `gate_stale` once a question has waited longer than `GMS_GATE_STALE_MIN` minutes (default 30).
  Per-gate `pending`/`stale` seen state (keyed by `output_dir`/`gate_id`) lives in
  `<GMS_CONSOLE_DATA_DIR>/notify_seen_gates.json`, same pattern as queue regressions above. The
  `Ticker` calls it once per tick with output dirs from the job store's own argvs.
- **Missed slots** -- `notify.check_missed_slots()` notifies `slot_missed` once for a
  delivery whose `scheduled_at` is more than `GMS_SLOT_MISSED_MIN` minutes past (default 30)
  while its status is still non-terminal. Every other observer here fires on something that
  HAPPENED; this one fires on the absence of it -- a drain that stopped, a ticker never started,
  a platform paused and forgotten -- which is why nothing caught it before. Reads the queue
  read-only through `pubqueue.queue_rows()`; a missing DB file is an empty queue and is never
  created by observing. A NULL `scheduled_at` (legacy rows) cannot have missed a slot and is
  skipped rather than read as the epoch. Seen state is its OWN file,
  `<GMS_CONSOLE_DATA_DIR>/notify_seen_missed_slots.json`: "this row regressed" and "this row is
  late" are independent facts about the same delivery, and sharing `notify_seen_queue.json`
  would let either one clear the other's memory.

  **A paused platform is never late.** Its rows sit past their slot on purpose (per-platform
  pause, `docs/pipeline/119-pause-a-platform-eligible-retry.md`), so the check skips them -- and skips them WITHOUT recording seen state. The
  pause suppresses the notification; it does not consume it. A resume therefore surfaces
  whatever is genuinely late now, which is what the operator who resumed needs to be told, and
  matches the queue layer's own refusal to backfill or throttle a resumed platform.

  **This used to fire only from the console's in-process `Ticker`** -- a deployment that does not
  run the console server (no `gomoveshift server` process, no service unit) never ran the observer
  on the path that actually publishes. That path is a scheduled `gomoveshift dispatch-queue
  --apply`, so
  `cli/handlers/dispatch_queue.py:cmd_dispatch_queue` now calls `notify.check_missed_slots(config=config)`
  once per `--apply` pass, right after `drain_queue()` returns (never on a dry run). Same
  independently-guarded shape as every observer here: an exception is logged and swallowed, never
  allowed to turn a healthy publish pass into a failed one. The cadence of an external schedule is not tracked code; a tighter interval
  than hourly gives the `GMS_SLOT_MISSED_MIN` grace window above finer resolution.

**Slot precision lives in the delivery queue, not the schedule interval.** A schedule only needs
to drain often enough to notice a due delivery -- `dispatch.py`'s `_select_due` (via
`next_attempt_at`) decides which rows are actually due for THIS drain pass, and
`compute_next_attempt_at` is what times a delivery's slot. The canonical schedule to create once
this ticker is running in an instance is a `dispatch-queue` row draining every 5 minutes:
`POST /jobs/schedules` with `name=drain publishing queue`, `subcommand=dispatch-queue`,
`interval_minutes=5` -- nothing seeds this row automatically, an operator (or a follow-up) creates
it once per console instance.

An external cron that already runs `dispatch-queue --apply` may keep running alongside the console schedule.
