### Added

- `gomoveshift.pipeline.review_set_stops` (library only, no CLI yet): reads a day's `stop_mask_<date>.npz` and emits one schema-versioned JSON per sequence with the contiguous MOVING/STOPPED/CREEPING/UNCERTAIN runs in source seconds, the kept/cut decision for each (from `mask`, so CREEPING is never cut), the mask summary, and, when an assembly receipt is given, each kept run's output time range from `EditTimeline`. Without a receipt the output times are omitted and the document says why.
