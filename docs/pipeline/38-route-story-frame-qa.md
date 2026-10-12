## Route Story frame QA (`gomoveshift/pipeline/route_story/frame_qa.py`)

`check_frame` is a pure-numpy detector for three ways a rendered frame can go
wrong: page-white (the browser's own default background showing through
where the canvas hasn't painted — `scene/index.html` sets no `body`
background), unloaded-tile (`#101418`, the colour `scene/scene.js:buildStyle`
paints both the base `background` layer and the fog beyond bundle coverage,
which is therefore also what shows through wherever a raster tile hasn't
loaded), and flat-black terrain (all channels at or below `FLAT_BLACK_MAX`).
Each fraction is computed vectorised over the whole frame, with an optional
`hud_mask` excluding HUD/letterbox pixels from both the numerator and the
denominator, and a per-channel `tolerance` so near-exact colour matches
(`#fefefe`) still count without pulling in unrelated pixels. `assert_frame_ok`
raises `FrameQaError` naming the frame, every problem, and the pose, and
`summarize` rolls a sequence of results into the receipt numbers (frames
checked/failed, worst fraction of each kind and which frame it came from).
`proof.py:render_frames` runs `check_frame` on every frame's raw array before
label drawing and calls `assert_frame_ok` on the result, aborting the render
on a bad frame — see the Route Story terrain-proof render section below.

The film's scene renderer does the same: `SceneMapRenderer` runs `check_frame`
on every RAW capture — preflight samples and every map-leg frame of the long
render — before labels and the dashboard are drawn, and raises `FrameQaError`
carrying the pose and the failing frame (`rgb`). The detectors must never see
a composed frame: the label pills (`proof._draw_labels`) and the dashboard
rail are dark fills inside the unloaded-tile tolerance, and on the real example-route
trip the pills alone read 0.52% "unloaded" over a raw capture that was 0.05%
shadow speckle with no flat block in it. Photo and video frames never pass
through the renderer, so they are never judged. `frame_qa=None` turns the
check off; `checks_raw_frames` tells a caller the renderer owns this gate.

`check_frame` judges one frame in isolation; it can't see a jump BETWEEN
frames. `SceneMapRenderer.render_leg` calls
`continuity.check_camera_continuity` over its own planned per-frame camera
states (center, bearing, zoom, pitch) before any capture worker starts for
that leg, and raises `continuity.CameraContinuityError` on a flagged jump —
never after a frame is captured. A leg passing the check appends a
`{"checked", "cuts", "issues": ()}` record to `camera_continuity`, which
`film_receipt` carries as `receipt["camera_continuity"]` (`None` for
`PillowMapRenderer`, which has no follow camera to plan).

The plan-level check above never sees the actually-decoded output file —
`continuity.analyze_rendered_continuity` is the rendered pass (b): a
consecutive-frame diff over the delivered mp4 (or a decoded/synthetic frame
iterable), reporting spikes, frozen spans and label-flicker stutter windows,
skipping anywhere a planned cut or hold said a discontinuity was expected.
`render_route_story_film` builds that plan as it appends frames —
`assembly.FilmTimeline`, one small class recording each hold's exact span and
every part/photo/recap boundary cut in per-clip-local frame numbers, then
resolving them to absolute output-frame indices once every clip's own length
is known (the placed source video's length isn't known until after it is
re-encoded). `film.main` calls `_delivery_continuity` right after
`check_delivery` and stores the result under
`receipt["delivery_check"]["continuity"]`. It is record-only: an analysis
failure is recorded as `{"error": ...}` and never changes the delivery
pass/fail, and nothing downstream reads the numbers back — they are for a
human reviewing the receipt.
