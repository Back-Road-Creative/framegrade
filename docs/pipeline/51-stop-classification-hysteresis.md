## Stop-classification hysteresis (`gomoveshift/analysis/motion.py`)

`_find_dead_spots` classifies each sampled energy as stopped/moving. A single
hard threshold makes decel boundaries jitter frame-to-frame at the crossing and
spuriously splits/merges brief sub-threshold creep. Optional dual enter/exit
thresholds (a small state machine over the energy series) debounce this:

- **Enter "stopped"** the moment `energy < motion_threshold` (the enter
  condition is unchanged whether hysteresis is on or off).
- **Exit "stopped"** only when `energy > motion_threshold * exit_multiplier`
  (`exit_multiplier > 1` raises the exit band above the enter band, so
  near-threshold flicker can't re-open/close a span).
- Knob: `config.stop_detection.hysteresis` → `{enabled: bool, exit_multiplier:
  float}`. `enabled` defaults **False** and `exit_multiplier` defaults **1.0** —
  the disabled path is byte-for-byte the original single-threshold loop (stay
  stopped while `energy < motion_threshold`, exit on `>=`), so an existing run's
  output is unchanged unless an operator opts in. Even an accidental `enabled:
  true` at the default 1.0 multiplier collapses the exit band onto the enter
  band and reproduces today's spans.
- Threshold sourced from `config.dead_spots.motion_threshold` (`MotionAnalyzer`
  already reads it); hysteresis only changes the exit comparison, never the
  enter threshold or `min_duration` span filter.
