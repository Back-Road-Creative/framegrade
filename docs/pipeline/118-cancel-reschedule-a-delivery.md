### Cancel / reschedule a delivery (`publishing/queue.py::cancel_delivery`, `::reschedule_delivery`)

Service-layer only — no console route/button yet. `TERMINAL_STATUSES = ("published",
"failed_permanent", "cancelled")` is the single module constant every terminal-status guard reads
(`transition`'s and `acquire_lease`'s `status NOT IN (...)` clauses, the partial unique index, and
`dispatch._select_due`'s selection query, via the shared `_TERMINAL_SQL` literal) — adding
`"cancelled"` here once, rather than to a hand-kept copy in each site, is what makes it actually
terminal everywhere instead of just in the function that set it.

`cancel_delivery(conn, delivery_id, now=None) -> bool` moves a non-terminal row to `"cancelled"`.
Returns `False` (never raises) on an ordinary refusal: already terminal, or held by another
worker's live lease (`lease_owner IS NOT NULL AND lease_expires_at > now`) — a leased row may
already be mid platform-API-call, and cancelling it out from under a live worker would make the
console lie about what happened. An *expired* lease is not live and may be cancelled.

`reschedule_delivery(conn, delivery_id, scheduled_at, now=None) -> bool` sets a non-terminal,
non-live-leased row's `scheduled_at`. Not a `transition()` field (rescheduling changes no status),
so it is its own guarded `UPDATE`. A row waiting on retry backoff carries both `next_attempt_at`
and `scheduled_at`, and `dispatch._select_due` gates on both: moving the schedule to a **future**
time while `next_attempt_at` is still in the past clears `next_attempt_at` (an operator pushing a
slot out means "not due before this new time"; a stale backoff would otherwise leave the row due
immediately once the new schedule arrives). Moving the schedule **earlier** while `next_attempt_at`
is still in the future leaves it untouched — that backoff is a platform rate limit, not a display
artifact, and an unrelated reschedule must not shorten it.

**`scheduled_at` must be a readable timestamp, refused in the queue rather than at a
caller.** The column is TEXT and `_select_due` compares it as a string, so an unparseable value
does not fail loudly -- it makes the row never due, or due for ever. `reschedule_delivery` raises
`ValueError` on one, the same guard `transition` applies to `platform_published_at`, so the CLI, the
console and any future worker are all covered by one rule instead of three.

**Console surface: `POST /publishing/{id}/reschedule`**, a form field on the delivery detail
page, via `console/pubqueue.py`'s `reschedule` -- the same shape as `operator_action`, so a refusal
is a sentence on that page rather than a redirect to an apparently unchanged one. The route is
registered BEFORE `POST /publishing/{id}/{action}`, which would otherwise match the path and 404 it
as an unknown action. The split: 409 when the row's own state refused (terminal, or another
worker's live lease -- "not now"), 400 when the value did ("not like that"); either way the
timestamp the operator typed is still in the field, because working one out twice is the cost of
throwing it away. The control is offered only for a non-terminal row, like cancel and retry.
