### Fixed

- A `--replace-scheduled` drive with an unfinished replaced LF is retried automatically: `health check` closes its publish claim `failed` (not `complete`) and `health backlog` republishes it with the heal backoff, so a deferred or partial LF finishes with no human step. An abandoned LF stays terminal.
- After the old video is deleted a slot counts as passed only within 5 minutes of now (was 6 hours); a truly passed slot is rebooked at Waypost's next free slot, and a `slot_passed` LF is retried instead of skipped.
- Replace incidents carry their LF and resolve themselves when the LF is done or abandoned; the pre-upload deferral and the unfinished check now open incidents, and the quality gate failure is reported beside the replace check.
- A 403 whose message mentions "404" is no longer read as not-found; `upload_if_needed` flushes its checkpoint on the non-playlist branch too; a raising store reopen or launcher puts the drive back or names the `_superseded` run, and a repeat of `--replace-scheduled` plans from it.
- The replaced video's ledger entry takes the new title and slot. The fake `youtube_uploader` modules in the Phase D tests carry `QUOTA_REASONS`.
