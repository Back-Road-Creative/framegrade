### Added

- `metrics collect` now builds a daily Facebook reel-plays series from the nightly observation log (`tracking.facebook_series.facebook_reel_plays_series`) and feeds it to `classify_facebook_reel_reach`, so Facebook is classified shadow-first beside YouTube. The run prints `SUPPRESSION facebook=<status> days=<n>` (`WARMUP` below 28 days) and the equivalent `SUPPRESSION youtube=` line.
- A backtest fixture (`tests/test_suppression_facebook_backtest.py`) shows the detector flags the August 2026 Facebook reel-reach collapse.
