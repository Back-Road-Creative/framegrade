### Added

- The Phase A sign pass logs progress every 50 sampled seconds per day (done/total, hits, elapsed, ETA) and counts the stage each sample reached (sampled / grabbed / region / ocr_text / claims); the counters ride on the day summary and the A6 line and persist as `state.sign_pass_stats`, so a day's `0 hit(s)` can be read as "0 frames decoded" or "0 sign regions" without re-reading the footage.
