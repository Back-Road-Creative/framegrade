### Fixed

- The privacy mask was paired with the picture by timestamp, so a source with timestamp gaps took each frame's blur from the wrong frame. The blur encode now retimes mask and picture to frame-index time, so mask frame `j` is picture frame `j`.
- A blur run now stops, naming all three numbers, unless the source frame count, the mask frame count and the detection sidecar's `source_frame_count` agree. This runs before the cabin gate and before the encode.
- A detection sidecar or mask built on a different frame count is rebuilt instead of reused, so a day detected before this fix does not fail the gate on every retry.
- The cabin verdict and the day-mask reuse evidence now include the redactcam version and decode mode (`REDACTCAM_DECODE`).

### Changed

- redactcam pinned to `b0861b2`: the decode emits exactly the container's frames and raises when the count differs.
