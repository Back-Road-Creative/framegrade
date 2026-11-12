### Added

- `gomoveshift queue reslot-overdue-youtube` moves a `queued` YouTube Short whose slot has
  passed to the region's next free Waypost shorts slot (the allocator `reslot-overdue-manual`
  uses), in the queue row and in the sidecar's `publishAt`, never to a time already past and
  never onto a slot another YouTube row holds. Dry run unless `--apply`.

### Fixed

- `gomoveshift queue recover-terminal` no longer refuses a final-text-review `failed_permanent`
  YouTube row for a passed slot: it assigns the next free Waypost slot through the same
  allocator. Every other gate, and gate (f) for every other row kind, is unchanged.
