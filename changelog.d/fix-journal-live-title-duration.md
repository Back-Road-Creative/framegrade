### Fixed

- `publish-journal --replay-pending` now writes each ready entry's title and duration as YouTube shows them at replay time, not the values stored at drive time. A video retitled live before it published (Ace Basin LF 1) was going to reach the journal under its old title. A video still processing at upload reports a zero duration, and `plan_merge` now treats that as missing and falls back to ffprobe instead of journaling `00:00` (Ace Basin LF 2).
