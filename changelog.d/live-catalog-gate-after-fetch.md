### Fixed

- The publishing gate named a YouTube short "missing from the live catalog" when the short was published after the catalog snapshot was taken. The nightly snapshot runs at 02:50 ET, before the 03:05 drain, so every short that drain published held the rest of its project's YouTube shorts until the next night, and that night's drain repeated it. A short whose recorded `published_at` is later than the snapshot's `fetched_at` is now skipped by the missing-video check; a short with no recorded publish time is still judged against the snapshot.
