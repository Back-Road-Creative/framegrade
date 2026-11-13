### Fixed

- `gomoveshift queue sync-plan-tags` now also rewrites the sidecar tags of a `failed_permanent` YouTube row the final-text review refused before any upload (no upload id), so `queue recover-terminal` can re-queue it. Dry run stays the default, nothing is sent to YouTube, and the Short tag rule is unchanged.
