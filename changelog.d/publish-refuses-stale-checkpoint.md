### Fixed

- `health publish` approved and claimed a drive whose checkpoint the current code re-runs (Kenmare: `phase_bc_day_first` 3 → 4 after its audit), so Phase D refused its prerequisite after the slot was booked. It now refuses before any write, naming the stale phases, so the drive gets re-audited first.
