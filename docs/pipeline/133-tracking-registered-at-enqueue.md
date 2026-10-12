### Tracking store filled at enqueue, not at publish

`tracking/store.py`'s `register_short`/`register_shorts_batch` had no production caller: the only
write into `TrackingStore` was the publish hook, `queue.py._register_platform_id_best_effort`'s
`update_platform_id`, and that method only ever *updates* a short that already exists — it raises
`ValueError` and is swallowed as a debug line when the index isn't found. Nothing ever created the
record in the first place, so the store held nothing for `update_platform_id` to find.

`cmd_enqueue_plan` (`cli/handlers/enqueue_plan.py`) is the fix: after a sidecar's plan enqueues
successfully, `_register_enqueued_shorts_in_tracking` registers every short this run actually
spooled — `select_capped_shorts(list(plan.shorts), max_shorts)`, the same selection
`_media_verification_errors`/`_planned_deliveries` already use, so a `--max-shorts`-deferred short
is never registered either. Records are grouped by `project_name` (one `register_shorts_batch`
call per project) and keyed by `short_index`, matching exactly the `(project_name, short_index)`
pair `update_platform_id` looks up later.

Grounded fields only: `file` (`record.media.path`), `duration_sec`
(`record.media.duration_seconds`), `tags` (the YouTube payload's tags, when present), and a
project-level `location` label built from the union of every spooled short's `sampled_regions`.
`ContentMeta.hook`/`cta`/`weather`/`title_format` and any per-short series id are left at their
schema defaults — no field on `PublishRecord`/`PlatformPayloads` carries a title-format template,
a series identifier, hook text, weather, or a per-short location; inventing one would violate the
no-fabricated-values rule this store's own docstrings already enforce (see `ShortRecord`'s
`posted_at`/`posted_at_source` provenance guard).

A `--shadow` enqueue registers nothing: a shadow delivery can never publish, so a record for it
would be a short the store counts forever and never sees posted.

Idempotent by construction: `register_shorts_batch` finds-or-updates by index rather than
appending, so re-running `enqueue-plan --apply` against the same sidecar (already idempotent at
the queue layer — `enqueued_count`/`already_present_count`) never duplicates a tracking record.

Non-fatal by design, matching the publish hook's own contract for this same store: a registration
failure is logged as a `WARNING` and swallowed — tracking bookkeeping must never block a delivery
from being spooled.
