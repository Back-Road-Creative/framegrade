### Fixed

- Publishing now blocks a day whose blur mask frame count differs from its source's. A drive rendered before the redactcam frame-count fix carries a mask a few frames longer than the picture, so the blur lags it. `health publish`, the backlog republish and Phase D's upload run refuse such a day, naming it and both counts, and record one `mask_index_mismatch` incident per drive-day. A missing sidecar, a missing count or an unprobeable source also refuses (`mask_frame_count_unknown`). There is no override; a re-render that rewrites the sidecar clears it.
