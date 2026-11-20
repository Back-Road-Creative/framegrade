### Added

- `gomoveshift queue sync-short-text [--apply]` re-writes the title, description and tags of every queued YouTube Short more than 24 hours from its slot with the current generator (no repeated place, seeded from the stored text of its neighbours). `experiment_arms` are kept as stored. A Short is refused whole when any of its delivery rows is in flight, leased, unslotted or inside the lead time, or when the Short review refuses the new text. The command exits 1 when any Short is refused. See `docs/pipeline/183-queue-sync-short-text.md`.
