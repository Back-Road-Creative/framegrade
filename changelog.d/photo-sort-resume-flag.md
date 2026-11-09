### Added

- `photos sort --resume <run folder>` finishes an earlier run's unfinished stacks, HDR
  fuses and panoramas without re-developing or re-scoring.

### Changed

- `photos sort` drops to the lowest CPU priority (`nice 19`, was 10) so it never slows a
  GoMoveShift render running on the same machine.
