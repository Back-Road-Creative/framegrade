### Added

- The drive orchestrator now records wall time per phase (start, end, seconds, work units and frames/s when known) in `phase_timings.json` beside the checkpoint, and preflight's wall-clock estimate uses the median of those recorded rates when any exist, falling back to the built-in constants otherwise. A new `Phase rates:` line in the plan says which source the estimate used.
