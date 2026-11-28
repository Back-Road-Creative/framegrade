### Fixed

- A drive resumed after a boundary-proof fix kept the route record its first run
  built, so Galley Head (2025-09-20) kept a record with no County Cork proof and its
  part 1 upload was refused nine times (`region_gaps`, `place_dropped`). Phase B+C's
  code version is now 5, so a resume rebuilds the route record and the coastal
  county proof from 0.3.182 applies. A test now refuses a boundary-proof version
  change that doesn't also bump the phase B+C version.
