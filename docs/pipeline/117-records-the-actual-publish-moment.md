### `published_at` records the actual publish moment (`publishing/queue.py`)

`deliveries.updated_at` is rewritten by every transition (a retry, a lease renewal, a reconcile),
so it can never answer "when did this actually publish" — only "when was this row last touched".
`published_at` is a dedicated column, set exactly once, inside `transition()`, only on the
transition into `"published"` (`fields["published_at"] = _now_iso()`, alongside the existing
`error_class`/`error_detail` clear). It is deliberately absent from `transition()`'s `allowed`
field allowlist — callers may not set it directly, and passing `published_at=...` explicitly
raises `ValueError` before the write, the same guard that rejects any other unknown field. The
once-only guarantee rides on the existing CAS: the UPDATE is already gated on `status NOT IN
('published', 'failed_permanent')`, so a row that has already reached `published` can never match
that WHERE clause again, and `published_at` can never be overwritten by a later call. A DB created
before this column existed gains it via the same idempotent `ALTER TABLE ... ADD COLUMN` migration
pattern as `permalink`/`shadow`/`scheduled_at`/`sidecar_path`; existing rows get `NULL` rather than
a backfilled guess from `updated_at`, which would be a fabricated timestamp.
