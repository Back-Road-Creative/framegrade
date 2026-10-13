## Insert engine

`gomoveshift/pipeline/inserts.py` key-cuts stills and clips into a rendered long-form without re-encoding the rest of it.

## Data model

- `Insert(at_s, media, hold_s=4.0, style="full")`: `style` is `full` or `letterbox`; a clip ignores `hold_s` and uses its own duration. A still is recognised by its suffix, anything else is a clip.
- `plan_inserts(lf, inserts, *, profile, rendition, ...) -> list[InsertPlan]`: probes the LF, renders still cards, fits clips, computes the cuts. Raises before any encode: `ValueError` for a bad style, `UpscaleRefusedError` for a small clip under `media.upscale: off`.
- `apply_inserts(lf, plans, output, *, run=subprocess.run) -> InsertResult`: `result.spans` lists each `Span(start, end, stream_copied)`; `result.upscaled` lists the source sizes of upscaled clips. All ffmpeg calls are argv lists.

## Keycut rule

`cut_in` is the keyframe at or before `at_s`; `cut_out` is the first keyframe at or after `at_s + duration` (the file end if none). Only `[cut_in, cut_out)` is re-encoded, with the join-repair args (`montage_builder.lf_reencode_args`); `[0, cut_in)` and `[cut_out, end)` are `-c copy`, then joined with the concat demuxer. Keyframes are the probed ones (`keycut.probe_keyframes`), not arithmetic on `GOP_DURATION_SEC` (2.1 s, `montage_builder.py`): that constant is a merge threshold and scene cuts add real keyframes, so a fixed grid can name a time that is not a keyframe. No keyframe index means `InsertError`. Inserts whose bounding GOPs touch are refused.

## Audio rule

Under a still the LF's own track (already attenuated by the loudness pass) is kept; no new gain is applied. Under a clip the LF audio is muted for the span and the clip's own audio plays.

## Upscale record

A clip is fitted to the LF frame through `quality.upscale_plan(w, h, rendition, policy)`, with the rendition's size set to the LF's. When the plan is not `None` the clip is an upscale: `InsertPlan.upscaled_from = [w, h]`, `scaler_filter` begins with the `zscale` chain, and the result repeats the source size.

## Not here

No CLI flag or config surface and no Phase D wiring yet; that is a follow-on change.
