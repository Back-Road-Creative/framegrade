### Queue-published ids feed tracking (`publishing/queue.py::transition`)

Reaching `published` on any delivery now registers the delivered platform id against the
project's tracking record via `TrackingStore.update_platform_id` — the first (and only) import
edge from `publishing/` into `tracking/`, imported lazily inside the hook to keep it out of
module scope and off every other caller's import graph. `registered_platform_id(platform,
platform_upload_id, platform_post_id)` picks the column: YouTube's adapter
(`youtube_adapter.py`) stores the video id in `platform_upload_id` and the playlist id in
`platform_post_id`, so YouTube registers the upload id; every other platform's `platform_post_id`
IS its trackable id. No id on either column (nothing delivered yet) is a silent no-op — the store
is never even constructed. Non-fatal by construction, same contract as the Phase-D post-upload
hook it mirrors: `ValueError` (short not yet registered with the tracking store) is expected and
logged at debug; any other exception is caught and logged at warning — a publish must never fail
because tracking bookkeeping did.

`gomoveshift metrics backfill-ids` is the one-shot catch-up for rows that published before this
hook existed: it walks every `published` row in the queue, applies the same
`registered_platform_id` selection, and calls `update_platform_id` per row (best-effort, per-row
`ValueError` skipped and printed, never aborts the walk). Idempotent — `update_platform_id`
overwrites in place, so re-running registers the same ids again with no duplication.

Both bridges also carry the posting time and, inseparably, WHICH moment it is. The queue
already distinguishes a human's late click from a platform's confirmed second
(`deliveries.published_at_source`, four provenances, enforced by `transition`), but that contract
used to stop at the queue: the bridges transferred the id alone, and `ShortRecord.posted_at` was a
bare ISO string. `tracking/decline.py` and `tracking/collector.py` anchor age-based judgements to
that string, so an `operator_confirmed` time hours after the fact made a short look younger than
it is and its early decline look steeper — the exact error the publishing-side provenance exists
to prevent. `ShortRecord` gained `posted_at_source`, and `__post_init__` refuses a time without a
source, a source without a time, and any source the queue does not issue, so the gap cannot
reopen by someone setting `posted_at` and forgetting the label. The vocabulary is a local literal
in `tracking/schema.py` rather than an import — `publishing` → `tracking` is the only edge between
these packages and importing back would make them mutually dependent — with the two lists pinned
equal by `tests/test_tracking_posted_at_provenance.py`.

`recorded_publish_time(published_at, published_at_source)` in `publishing/queue.py` is what both
bridges call, so neither can drift from the other: a complete pair crosses intact, and anything
else (a pre-provenance row holding a time and no source) crosses as `(None, None)` — the id still
registers and the unknown stays an unknown, because handing the bare time over would make it look
like a measurement. `from_dict` assigns both fields after construction for the same reason: a
legacy record on disk still LOADS, since refusing to read data that exists would turn an honest
unknown into an outage. `TrackingStore.update_platform_id` and `register_short` move the pair
together or raise.
