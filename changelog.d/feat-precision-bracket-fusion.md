### Added

- `photos develop --hdr` fuses a 16-bit (RAW) bracket as float32 (OpenCV's 0-255 scale) instead of stepping each frame down to 8 bits first, so a smooth gradient keeps its sixteen-bit tonal precision through the merge; an all-8-bit bracket is byte-identical to before. This is still exposure fusion, not scene-linear HDR. See `docs/pipeline/83-photo-pipeline.md`.
