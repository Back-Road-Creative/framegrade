### Added

- The per-day stop mask is now derived from a per-second motion state (`MotionState`: moving, stopped, creeping, uncertain; `classify_motion_states`). `stop_mask_<date>.npz` gains an int8 `states` array beside `mask`; `mask` keeps its meaning (True is a stopped second, creeping is never True). Default output is unchanged.
- `stop_detection.creep_policy` (`remove`, the default and today's behaviour, or `keep`) and `stop_detection.creep_min_speed_kmh` (default 2.0, provisional pending tuning on labelled real drives). With `keep`, sustained GPS speed from `creep_min_speed_kmh` up to `stop_speed_kmh` is creeping and not masked.
- The day-first and per-day stop mask now honours `stop_detection.hysteresis`, which until now only affected the legacy optical dead-spot pass. The same key therefore now also changes the mask when enabled. It stays off by default.
