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

## The same hysteresis in the per-day stop mask

`pipeline/stop_mask.py` (`classify_motion_states`, behind `build_stop_mask`) applies
the same enter/exit rule when `config.stop_detection.hysteresis.enabled` is true,
so that one key now also governs the day and drive-wide mask, not only the optical
dead-spot pass. A second enters a stop on the normal test (GPS speed below the stop
floor, or optical energy below `motion_threshold` when speed cannot be measured);
once in a stop it stays until the signal exceeds its threshold times
`exit_multiplier`. The sustained-run filter (`min_stop_duration`) still applies
afterwards. Disabled (the default) the mask is identical to before.

## Motion states and the creep policy (`pipeline/stop_mask.py`)

Each second of a day is classified as a `MotionState` (int8):

| value | state | meaning |
| --- | --- | --- |
| 0 | MOVING | not a stop |
| 1 | STOPPED | a sustained stop; the only state the mask marks True |
| 2 | CREEPING | sustained slow GPS speed kept on purpose (`creep_policy: keep`) |
| 3 | UNCERTAIN | no usable GPS speed and no usable optical reading; treated as not a stop, as before, but now visible |

- `stop_detection.creep_policy: remove` (default) masks every sustained span under
  `stop_speed_kmh` (10 km/h), exactly as before. `keep` leaves speeds in
  `[creep_min_speed_kmh, stop_speed_kmh)` as CREEPING and unmasked; speed below
  `creep_min_speed_kmh` is still a stop.
- `stop_detection.creep_min_speed_kmh` (default 2.0) is **provisional**, pending
  tuning on labelled real drives.
- Parked with passing traffic (0 km/h, high optical motion) stays STOPPED under both
  policies: GPS speed is authoritative.
- `stop_mask_<date>.npz` holds `mask` (bool, True == a stopped second; unchanged) and
  `states` (int8, the table above). CREEPING is never True in `mask`.
