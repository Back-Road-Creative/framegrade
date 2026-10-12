"""`day_pipeline` carries no bare stream-copy slice helper.

`extract_slice` cut with a naive `ffmpeg -ss/-to -c copy` and no keyframe probe, so
its start snapped BACKWARD to the previous source keyframe — it replayed up to one
GOP of the preceding footage. Nothing in production ever called it: regular LFs are
cut by `extract_and_concat_segments` (which snaps every cut start FORWARD onto a
probed keyframe, and re-encodes when it cannot), and shorts are rendered by
`render_short_ffmpeg` in one pass.

The earlier version of this file swept the package AST for call sites, which held
the line only for as long as the function existed to be called. The function is now
deleted, so this pins the stronger property directly — the symbol is not there to be
wired back in.
"""

from __future__ import annotations


def test_day_pipeline_defines_no_bare_stream_copy_slice_helper():
    from gomoveshift.pipeline import day_pipeline

    assert not hasattr(day_pipeline, "extract_slice"), (
        "extract_slice is a backward-snapping stream copy that replays up to one "
        "GOP of the preceding footage; cut LFs with extract_and_concat_segments "
        "and render shorts with render_short_ffmpeg instead"
    )
