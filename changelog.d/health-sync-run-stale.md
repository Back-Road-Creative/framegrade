### Added

- `gomoveshift health sync-short-text` now writes a run record (`<incidents dir>.short_text_sync_run.json`: `schema`, `completed_at`, `changed`, `refused`) whenever the sync returned, atomically; a crash writes nothing and a write failure never changes the exit code. `gomoveshift health check` rule 13 gains `seo_feedback_stale:sync`, which opens when that record is missing, unreadable or over 26 h old and resolves on a fresh one, so a nightly sync that stops running is noticed. See `docs/seo-feedback-stale.md`.
