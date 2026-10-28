## Source-cadence output

By default every long-form master is published at the frozen 29.97 cadence (`OUTPUT_FPS = "30000/1001"` in `gomoveshift/pipeline/color_filters.py`): a 59.94/60 fps source is decimated 2:1 by the explicit `fps` filter at the tail of the video chain. Source-cadence output is an opt-in library parameter that keeps a 59.94/60 fps source at its own rate. Nothing calls it with a non-default value yet; the default argv and filter strings are unchanged.

## Parameter

`output_fps` (default `OUTPUT_FPS`) on:

- `cfr_filter(fps)`: returns `fps=<rate>`.
- `get_encoding_args(..., output_fps=)`: stamps `-r <rate>` and sets the GOP to 2 s of that rate (`-g 60` at 29.97 or 30, `-g 120` at 59.94 or 60).
- `SinglePassProcessor.process`, `_build_video_filter`, `_build_blur_filter_complex`: the `fps` filter stays the chain tail, strictly after the privacy-blur composite, because the mask pairs 1:1 with source frames.
- `ost_applier.apply_ost`: the OST burn re-encodes the master, and its `-r` is the only rate control (no `fps` filter), so source-cadence mode passes the master's own rate and `-r` never re-decimates.

## Allowed values

`SUPPORTED_OUTPUT_FPS` is `30000/1001`, `30`, `60000/1001`, `60`. Anything else raises `ValueError` ("unsupported output rate"). A non-default rate must also equal the probed source rate (`check_output_fps`, run in `process` and the OST encode args); a mismatch raises `ValueError` rather than upsampling or decimating. `source_output_fps(probed_fps)` maps a probed rate to its supported string, or refuses.

## Verification

`tests/test_source_cadence.py` encodes a synthetic 59.94 fps lavfi clip with real ffmpeg: the default gives `r_frame_rate` 30000/1001 and half the frames; source mode gives 60000/1001 and every source frame.
