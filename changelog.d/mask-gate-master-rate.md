### Fixed

- The blur-mask frame-count gate no longer refuses a 59.94 fps day as a mismatch because its master was encoded at 29.97 fps. When the raw concat is gone and the master runs at another rate than the sidecar's `source_fps`, the source-frame proof the encode persisted is the count. With no valid proof the day is `mask_frame_count_unknown`, naming both rates. A master at the source rate still wins over the proof.
