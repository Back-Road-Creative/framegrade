### Fixed

- A skipped source-frame proof is never silent: every path that writes none logs a WARNING naming the day and the reason, a failed write removes its `.tmp`, and the stage path (not a real day) writes no proof.
- Day-master cleanup keeps the master unless the raw is present or the persisted source-frame proof validates, so an mtime-only change to a purged day's mask no longer leaves it `mask_frame_count_unknown` forever.
- The publish gate no longer skips a `day_<date>` dir silently: one holding finals but no `day_<date>_*` file is `mask_frame_count_unknown`; an empty one is logged.
