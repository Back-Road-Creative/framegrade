### Fixed

- A live-catalog video with a drive's expected title and no ledger row or receipt (a publish that died after the YouTube upload) is now an `orphan_upload` incident that refuses that drive's `health publish` with the video id, title, privacy and `publish_at`, instead of the next run uploading a second copy. Nothing on YouTube is deleted, unlisted or adopted, other drives are untouched, the refusal counts no attempt, and an unreadable ledger refuses too.
