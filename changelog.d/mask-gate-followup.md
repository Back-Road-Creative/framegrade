### Fixed

- A skipped source-frame proof is never silent: every path that writes none logs a WARNING naming the day and the reason, a failed write removes its `.tmp`, and the stage path (not a real day) writes no proof.
- Day-master cleanup drops a master only once the publish gate can measure the day without it: a raw that probes to a positive frame count, else a persisted source-frame proof that validates (written from the master itself when none does; otherwise the master is kept with a WARNING). A 0-byte, unprobeable or dangling raw no longer counts. A mask change after the drop still leaves that day `mask_frame_count_unknown` (fail closed).
- The publish gate no longer skips a `day_<date>` dir silently: one holding finals but no `day_<date>_*` file is `mask_frame_count_unknown`; an empty one is logged.
