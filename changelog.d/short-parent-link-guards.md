### Fixed

- A Short links its full drive only when that drive is public by the time the Short is. The first comment (at enqueue and at backfill, judged at each platform's own slot) and the plan's YouTube description no longer link a drive that is still scheduled, nor one whose publish time is missing or has no timezone. The reviewed description link uses the same rule, `first_comment.parent_is_public_at`.

### Changed

- For a Short that goes out before its drive, a rebuilt plan's YouTube description drops the trailing link, so that Short's plan text and hash change. Rows already queued keep the first comment they were enqueued with.
