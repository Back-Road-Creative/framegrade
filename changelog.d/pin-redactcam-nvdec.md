### Changed

- redactcam pinned to `ce6a1c2`: stage-1 detection decodes the day master through
  ffmpeg NVDEC (`-hwaccel cuda`, `hevc_cuvid`/`h264_cuvid`/`av1_cuvid`) piped as
  bgr24, falling back to `cv2.VideoCapture` when ffmpeg, the cuda hwaccel or the
  cuvid decoder is missing, the probe fails, the clip is rotated, or the run resumes
  from a checkpoint (`start_frame > 0`). Measured on a 4K60 HEVC master: 18.1 fps
  cv2 vs 31.8 fps NVDEC. `REDACTCAM_DECODE=cv2|nvdec|auto` overrides.
  `tests/test_redactcam_parity.py` pins the new `decode="auto"` default.
