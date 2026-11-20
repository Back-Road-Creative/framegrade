### Changed

- redactcam is pinned to 143b3be. The blur mask is now rendered only inside the box regions, and frames with no box are skipped (redactcam #21). The output is byte-identical to the previous full-frame render, so the privacy result does not change; the blur stage does less work per frame.
- The box-region crop is kept out of OpenCV's small-image threading race (redactcam #22), so the faster render is deterministic under multiple threads.
