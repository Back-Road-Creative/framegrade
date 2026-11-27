### Fixed

- `health backlog --supersede` relaunching a drive's own `failed` or `auditing` row is no longer refused "in the content ledger" when the drive's ledger entries hold no video id (Shorts placeholders a failed publish left behind). An entry with a video id still refuses, every other ledger row is still skipped, and the automatic failed-row relaunch still never takes a ledger row.
