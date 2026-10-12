## Route Story terrain-proof render (`gomoveshift/pipeline/route_story/proof.py`)

`build_shot_plan` assembles the descent → reveal → chase → arrival → pullback `ShotPlan`; before any
rendering, `render_proof_clip` computes `planner.max_shot_zoom(imagery_maxzoom)` (the 1.25x
enlargement cap) and calls `plan.assert_within_zoom(max_zoom)` — an over-enlarged plan raises before
`capture_factory` or `sink_factory` ever run, so a bad plan never starts a browser or opens an
ffmpeg pipe.

`render_proof_clip` takes a required `recorded_on` and selects that single calendar day via
`gpx.select_gpx_day` (the same day-selection `prepare_bundle` uses) for both the route points and
`trip_epoch` (the selected day's own first timestamp) — a GPX carrying other-date or duplicated
tracks (a common real-world export) never widens the plan's route bbox or shifts the sun's clock off
the day actually being filmed. The `route-story-proof` CLI resolves its bundle through
`prepare.find_prepared_bundle` — the identical day-selection/footprint/exaggeration/source-version
key `prepare_bundle` writes under, never a second, independently-derived key — so it takes the same
`--recorded-on`/`--exaggeration`/`--regional-margin-km` a `route-story-prepare` run used, plus its own
`--workers` (parallel render workers; defaults to `render_proof_clip`'s own `workers=None` resolution).

`route_light_fn` builds the per-frame `sun.SceneLight` lookup `render_frames` calls every frame:
- **Chase frames** get the sun's real position and elevation at *that frame's own camera position*
  (`shot.pose_at(frame)`) and the route's own recorded clock time, via `chase_route_time_s` —
  which reuses `chase()`'s own arc-length-to-position mapping (same `_smootherstep` fraction, same
  segment/local lookup, via the shared `_arc_length_progress`/`_segment_at_fraction` helpers) rather
  than inventing a second frame-to-time mapping, then interpolating `RoutePoint.time_s` instead of
  longitude/latitude.
- **The closing (pullback) shot** — the plan's last shot — sweeps from the chase's own end time to
  the trip's end time via `sun.sun_sweep`, guarding the pathological one-frame case (sweep needs
  ≥2 frames) by falling back to `light_at` at the trip's end time.
- **Every other shot** (descent, reveal, arrival — none has its own route time) uses
  `sun.light_for_pose(None, ..., fallback=trip_start_time)`, i.e. the sun at the trip's own start
  time and the route's centre — the same honest-but-static light the whole clip used before
  per-frame lighting.

`render_frames` runs `frame_qa.check_frame` on every frame's raw array (before `_draw_labels`) and
`frame_qa.assert_frame_ok` on the result, aborting on a bad frame; the frame sink is always closed
(even on that abort) so a failed render never leaves an orphaned ffmpeg process. `RenderStats` carries
`light_summary` (min/max azimuth and elevation over the clip, plus a night-frame count) and
`frame_qa` (`frame_qa.summarize`'s per-clip rollup); `render_proof_clip`'s receipt
(`route-story-proof-receipt/2`) folds both in, plus a `zoom_cap` block (`imagery_maxzoom`,
`enlargement`, `max_zoom`) recording the enlargement-cap check that ran before rendering started, and the
measured `render_fps`/`x_realtime` actually achieved.

`render_frames(..., workers=N, capture_factory=...)` (`N > 1`) renders in parallel: frames are
striped round-robin across workers (worker `i` owns `i, i+N, i+2N, ...`, not a contiguous block),
each with its own capture built from `capture_factory` and driven entirely inside its own thread
(Playwright's sync API is not thread-safe across threads sharing one browser). Striping matters
because the consumer requests frame `f` from queue `f % N`, cycling every worker's queue in turn —
a contiguous range would leave every worker but the one currently being drained idle behind its own
bounded queue for most of the render, since the consumer only reaches worker `k`'s queue after
worker `k-1`'s whole block is exhausted. Every worker's frames still reach the single `sink` in
strict `range(frame_count)` order — one `FfmpegFrameSink`, one encode, never per-shard files — via
a bounded per-worker queue, so the whole clip is never held in memory at once. A frame QA or render
failure in any worker records the original exception and cancels every other worker; the consumer
polls with a timeout so it can never block forever on a worker that stopped without producing the
frame it owed, then waits for every worker (and so every capture) to exit, closes the sink, and
re-raises that original exception. `render_proof_clip`'s `workers` argument defaults to
`GMS_ROUTE_STORY_RENDER_WORKERS` if set, else `default_render_workers()` (half the box's cores, capped
at 6); the receipt records the `workers` actually used and the `readback` format (`SceneCapture`'s
`readback="jpeg"|"png"` screenshot kwarg — JPEG quality 95 by default, replacing a PNG-encode-then-PIL-
decode round trip). `scene.js`'s settle poll (`SETTLE_POLL_MS`) is 60ms, not 250ms.
