## Route Story preflight (`gomoveshift/pipeline/route_story/preflight.py`)

A full film render captures frames at roughly 3-4 s each and can run for hours, so a visual or
tile defect (unloaded tile, page-white void, flat-black terrain) must be caught in minutes before
that render starts, not by an operator remembering to run a separate proof first. `film.main` runs
`run_preflight` by default (`--preflight-frames`, default 12; `0` disables it, diagnostic-only)
before the long render: it samples that many evenly spaced route-clock times across the whole
story (`preflight_times`, endpoints included) through the SAME `MapRenderer` instance and
single-frame `render` call the long render later drives — real bundle, tiles, labels, marker, HUD
path — checks each with `frame_qa.check_frame`, and raises `RoutePreflightError` naming the
failing route time and detector before any map leg of the long render is produced. Frames are also
tiled into one captioned contact-sheet JPEG next to the film output (`<output stem>-preflight.jpg`,
written even on a QA failure so a human or an agent can triage it) — the surface for a
defect frame QA cannot see. `--preflight-only` runs just this gate and exits without the long
render; a successful run adds `preflight: {frames, passed, sheet}` to the film receipt.
Preflight deliberately calls `render`, never `render_leg`: only `render_leg` carries a follow
camera's inertia state (`SceneMapRenderer._last_bearing`/`_last_bearing_velocity`) across calls,
so sampling cannot disturb the bearing plan the long render's first leg depends on.
Who judges a frame depends on the renderer. One that declares `checks_raw_frames` (the scene
renderer) has already run the detectors on its raw capture, so preflight does not re-judge its
composed frame — the label pills alone read 0.52% "unloaded" on a clean example-route frame — and a
`FrameQaError` out of `render` stops the sample at once, with the rejected raw frame tiled last
on the sheet. Any other renderer's composed frame is judged here, with the dashboard rail masked.

<a id="route-story-review-pack"></a>
