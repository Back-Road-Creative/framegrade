### Fixed

- `photos sort` no longer hangs with nothing developed: its develop workers are now spawned
  instead of forked, so they do not inherit the OpenMP lock state rawpy left in the parent
  while measuring corrections.
