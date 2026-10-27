### Fixed

- Deflicker and vidstab-transform intermediates no longer inherit ffmpeg's defaults (h264, crf 23 for an `.mp4`). They are written as x265 at the long-form CRF and preset, keep the input's pixel format (10-bit stays 10-bit), and copy the input's colour range, matrix, primaries and transfer from an ffprobe, with primaries and transfer also set through `-x265-params`.
- The deflicker and vidstab-transform intermediates now pick their encoder like the day master: `hevc_nvenc` (long-form NVENC preset and CQ, main10 `p010le`, no `-r`) when a GPU works, libx265 only as the fallback. The choice is injected as `has_nvenc` from `stabilize.nvenc_available()`, so tests need no GPU.
