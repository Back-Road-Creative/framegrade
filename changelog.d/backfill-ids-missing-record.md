### Fixed

- `gomoveshift metrics backfill-ids` skips a published row whose project has no tracking record and goes on with the rest. On 2026-11-27 one such row (an old split drive) raised `FileNotFoundError` and ended the run, so no row after it got its id.
