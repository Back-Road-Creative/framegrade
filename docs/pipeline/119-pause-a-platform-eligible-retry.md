### Pause a platform / eligible retry (`publishing/queue.py::pause_platform`, `::resume_platform`,
`::is_platform_paused`, `::retry_delivery`)

Wired to the console (`POST /publishing/pause`, `docs/pipeline/104-publishing.md`) — this section stays the
authority on the rule; the route only renders its refusals. Pause state is a `platform_pauses` table in the
**queue database**, not a console setting: `dispatch-queue` runs as a launched job in a separate
process from the console, so any state a pause needed to affect the drain has to live somewhere
both processes open — the queue file is the only such place. `platform_pauses` is added via
`CREATE TABLE IF NOT EXISTS` inside `_SCHEMA` itself; unlike a column added to the existing
`deliveries` table (which needs the `ALTER TABLE` idempotent-migration pattern in `publishing/queue.py`) or the
active-delivery index (whose predicate `CREATE UNIQUE INDEX IF NOT EXISTS` would silently keep
stale), a wholly new table has no prior definition to go stale — simply reopening an
already-deployed queue file creates it.

`pause_platform(conn, platform, now=None) -> bool` upserts a pause row; refuses (`False`, never
raises) a `platform` not in `publishing.models.KNOWN_PLATFORMS` rather than creating a row nothing
will ever match. `resume_platform(conn, platform) -> bool` deletes it (`False` if it wasn't
paused). `is_platform_paused(conn, platform) -> bool` is the read.

The pause is enforced inside `dispatch._select_due`'s `WHERE` clause (`platform NOT IN (SELECT
platform FROM platform_pauses)`) rather than in the drain loop or a console route — the same reason
`TERMINAL_STATUSES` and `shadow` are gated there: it is the one query every caller (CLI, cron,
future worker) funnels through, so no other path can publish a paused platform's row. A paused
row is excluded, never mutated — no status change, no attempt bump, no error written — because a
pause means "leave this alone", not "fail it". It governs only what is picked up NEXT: a row
already leased when the pause took effect is untouched, since the platform API call it may be
mid-flight on cannot be un-sent. Resuming restores selection with no backfill or catch-up
throttle — if rows piled up while paused, they all become due on the next drain pass, a real and
visible consequence of the pause's duration rather than something to mask with an invented
throttle.

`retry_delivery(conn, delivery_id, now=None) -> bool` clears a non-terminal, non-live-leased row's
`next_attempt_at` (and its `next_attempt_reason`), making a row backing off on an ordinary retry due
on the next drain pass immediately. It refuses -- leaving the row untouched, same `False` as any
other refusal -- when `next_attempt_reason` is `platform_throttle` (only the platform clears a
throttle) or `awaiting_manual` (a human, not a retry, must act); every other reason, including a
pre-migration row whose reason predates this column and reads `NULL`, is treated as an ordinary
backoff and cleared. Same refusal shape as `cancel_delivery` otherwise: `False`, never raises, on a
terminal row or one under another worker's live lease. Deliberately leaves `attempt_count`
untouched — resetting it would let repeated manual retries evade `drain_queue`'s `max_attempts`
ceiling forever, and incrementing it would burn an attempt against a platform call that has not
actually happened yet; the count only moves when the drain loop itself makes and grades the next
real attempt.

**A worker that dies holding a lease has its attempt counted**
(`dispatch.reconcile_interrupted_attempts`, run at the top of every `drain_queue` pass). A worker
killed between `acquire_lease` and any `transition` -- OOM, host restart, SIGKILL -- leaves
`attempt_count` untouched. The lease expires, the row becomes due, and it is retried; that already
worked. What did not is the counting: every `max_attempts` check in `drain_queue` sits inside an
`except` clause, and a SIGKILL raises nothing, so a delivery that reliably kills its worker was
retried forever while the console read "queued, 0 attempts" -- indistinguishable from a delivery
that had never been tried at all.

An EXPIRED lease is the signal, and the only one available: a worker that finishes either
terminalizes the row or releases the lease, so a lease outliving its own expiry means nobody is
coming back for it. Such a row is charged one attempt, given the ordinary retry backoff (whatever
killed the worker is probably still true a second later), has the dead worker's lease cleared, and
records `error_class` `InterruptedAttempt` -- or `MaxAttemptsExceeded(InterruptedAttempt)` when
that attempt crosses the ceiling, which is the point: a crash loop now ends. A LIVE lease is left
strictly alone; that worker may be mid platform-API call, and clearing its lease would let a second
worker publish the same short twice. An unleased row is left alone too -- it was waiting, not
interrupted, and charging it an attempt would spend every ordinary delivery's retry budget.

This does not decide whether the interrupted attempt reached the platform. It cannot, and the row
is retried blind either way; what changes is that the retrying stops. `DrainResult`'s
`reconciled_interrupted` counts them, and `dispatch-queue` prints it only when non-zero, the same
rule `throttled` follows -- a field that reads `=0` on every healthy run is a field nobody reads on
the run that matters.
