### Delivery detail (`console/pubqueue.py::delivery_by_id`, `GET /publishing/{delivery_id}`)

One-row drill-down (each queue row's project cell links here); 404 for a missing DB or unknown
id. Enriched like `rows_by_state` plus `lease_state` (`"none"`/`"live"`/`"expired"` from
`lease_owner`/`lease_expires_at` vs now -- expired differs from no lease, both differ from "a
worker is working on it right now"), `error_explanation`, and `platform_paused`
(`queue.py::is_platform_paused`, read on the connection the row itself came from). A paused
platform is skipped by every drain pass, so while it is paused the page says so instead of
promising an automatic retry or that the next drain will pick a manual retry up -- the same
pause check `console/notify.py` makes before calling a slot missed.

**The failure explanation is built from `error_class`, never `error_detail`** -- the latter is
untrusted platform text that can carry a signed URL/account id/hostname; no redactor is
attempted, since one that looks like it works invites trust it can't earn. Mapped classes:
`PermanentDeliveryError`/`ThrottledDeliveryError`/`TransientMediaError` (`publishing/dispatch.py`
per-exception `transition` calls), `quota` (`publishing/youtube_adapter.py`'s
`YouTubeQuotaError`), `tiktok_publish_failed` (`dispatch.py`'s `reconcile_awaiting_manual`),
`InterruptedAttempt` (`dispatch.py`'s `reconcile_interrupted_attempts` -- a killed worker, no
exception, so the sentence says the platform may or may not have received it), and
a `MaxAttemptsExceeded(...)` prefix (`dispatch.py`'s generic `except Exception` branch). Any
other class -- that branch records `type(exc).__name__` for whatever the SDK raised, an
open-ended set -- reads "no description yet; see the server log," and `error_detail` is logged
at warning level, the split `console/library.py`'s `_damage_summary` already uses.

**A permalink is only a link when it is one** (`pubqueue.py::linkable_permalink`, enrichment sets
`permalink_href`). `permalink` reaches the queue from a platform's API response and from an
operator typing into the manual-confirmation form -- neither is trusted input -- and both the queue
list and this page used to render it straight into an `href`. Jinja's autoescaping does not help:
it escapes quotes, and `javascript:alert(1)` needs none, so one click ran script in the console's
own origin. The guard is an **allowlist** of `http`/`https` with a non-empty netloc (`data:` and
`vbscript:` are only the schemes anyone thought to name today), and it lives in
`enrich_rows_with_sidecar` so both templates get it from the same place. A schemeless
`example.com/p/1` is refused for a different reason: in an `href` that is a RELATIVE path back
into the console, so the link would go somewhere other than where it reads. A refused value is
still **shown**, marked "not a link this console will open" -- an operator has to see what was
recorded to fix it, and silently dropping it would hide a hostile value instead of surfacing it.

**Cancel, retry, per-platform pause and reschedule are all wired.** `POST
/publishing/{id}/cancel` and `POST /publishing/{id}/retry` call the queue's own operator controls
through `console/pubqueue.py`'s `operator_action`, then redirect back to this page. Both rules stay
in the queue layer -- a terminal row is final, and a row under another worker's live lease may
already be mid platform-API call -- and both refusals arrive as a bare `False`. The console never
redirects on one: it re-reads the row and raises with the state it is in NOW (409), because a
redirect to an apparently unchanged page is indistinguishable from a cancel that worked. An unknown
id is 404, the same split `mark-published` uses. The page offers a control only for a non-terminal
row; a live lease is deliberately not filtered out, since it expires on its own and a button that
vanishes for invisible reasons is worse than one whose refusal explains itself. Retry clears
`next_attempt_at` and never moves `attempt_count` -- see `publishing/queue.py`'s `retry_delivery`
for why either direction would be wrong.
