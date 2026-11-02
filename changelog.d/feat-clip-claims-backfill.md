### Added

- `gomoveshift clip-claims backfill OUTPUT_ROOT... [--apply]` rebuilds the clip-claims registry from past outputs: every project folder with an `lf_provenance.json` whose long-form has a `video_id` in `youtube_uploads.json` gets its source clips claimed, so the Phase A reuse gate also sees footage published before the registry shipped. Dry-run by default, idempotent, and a folder without `lf_provenance.json` is reported skipped, never guessed.
