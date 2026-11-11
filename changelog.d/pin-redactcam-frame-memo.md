### Changed

- redactcam pinned to `1937312`: each sampled frame now letterboxes its detection
  tiles once for the face and plate models, and runs the COCO model once for the
  vehicle and person detectors (both specs resolve to the same model, sha
  `5ad2aa77f7f6`). Detected boxes are byte-identical. Measured on a 20 s 4K clip
  under load: 4.82/3.64 fps before, 4.94/3.95 fps after.
