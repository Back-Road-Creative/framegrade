### Added

- `gomoveshift queue sync-short-text [--apply]`, the command for the `queue_short_text.sync_short_text` library: a dry run unless `--apply` (a dry run writes no file); it exits 1 when any Short is refused or any sidecar is unusable. See `docs/pipeline/183-queue-sync-short-text.md`.

### Fixed

- `gomoveshift queue sync-short-text` commits the audit row and queue hash move before it replaces the sidecar, so a failure or crash never leaves them silently apart and the next run converges; a queued row left on a stale hash is moved to the sidecar's (`reconciled`, with an audit row), and a row with an unknown hash refuses the Short. It re-runs the enqueue gates on the new text (`validate_payloads` and the anti-spam batch check with the queue's recent text and the brand's per-location cap), skips hand-titled Shorts, and refuses a sidecar edited during the run. It snapshots the queue's recent text per sidecar (other projects' text is seen) and runs one anti-spam check across the run's rewritten Shorts, refuses a Short whose place repeats its neighbour's final place or whose new hash a row already holds (also on reconcile), retries a failing LF Short by Short, and rolls a sidecar back when a row was leased during the write. See `docs/pipeline/183-queue-sync-short-text.md`.
