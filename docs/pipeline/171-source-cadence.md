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

## Turning it on

`gomoveshift drive ... --output-cadence source` (config `drive.output_cadence`, default `fixed`) makes the day-master encode and the long-form and superlong OST burns keep the probed source rate instead of the frozen 29.97. `resolve_output_fps` turns the request into the supported rate string at the encode, so a source that is not 29.97, 30, 59.94 or 60 fps stops with the "no supported output rate" error before the encode starts; nothing is resampled. `fixed` (or omitting the flag) passes no `output_fps` and leaves every argv as it is. Shorts and inserts are not part of this option. The flag is a resume sentinel: a resume that leaves it out keeps the first run's choice.

### Compatible and refused combinations

Some stages re-encode a span at the fixed 29.97 rate and stream-copy-join it to the master, which would leave a 59.94 file at two cadences. They are handled one of two ways:

- Rate follows the source (compatible): the long-form and superlong segment extraction, including its stop-mask join repairs (`extract_and_concat_segments`), and the montage prepend (`prepend_montage`, `build_montage`, and the re-encode fallback in `_trim_peak_to`) pass the master's probed rate to `lf_reencode_args(output_fps)`. The default call is unchanged.
- Refused up front (before any encode, after the resume merge, so a saved choice counts): `--output-cadence source` with `--insert-photos` or `--insert-clip` (the insert splice), or with the trail film opening on. The trail film is tri-state, so only an explicit `--no-trail` clears it. The run stops with a one-line `PipelineValidationError` naming the option.

Shorts, thumbnails and the legacy per-episode concats (`stop_processor`, `assembly/encoder`, `route_story/assembly`, `ingest/concatenator`) stay at the fixed rate and are not part of the day-first path this option covers.
