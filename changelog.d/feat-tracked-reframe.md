### Added

- Opt-in tracked vertical reframe for shorts: `render_short_ffmpeg(..., reframe="tracked")` slides the 9:16 crop to follow the road using a low-pass, a dead zone and a pan-speed cap, so it never chases. The default `reframe="center"` produces the identical filter and ffmpeg arguments. Crop size, scaler and the declared upscale are unchanged, and shorts stay muted.
