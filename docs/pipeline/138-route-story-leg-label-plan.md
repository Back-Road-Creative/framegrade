## Route Story 3D-leg label plan (`gomoveshift/pipeline/route_story/layers.py`, `scene_renderer.py`)

A map label could drop out for one frame in the middle of a 3D leg and then return, while it was still in view. Cause: `scene.js`'s `projectLabels`
projects labels per pose, and `scene_renderer.SceneMapRenderer._render_frame` lays out the
pills for each frame with `proof.place_labels` -- a stateless, order-dependent greedy that
leaves out any label whose pill collides with one placed earlier in the list. A same-frame
newcomer sorting ahead of an already-shown label is enough to bump it for exactly that frame.

The opening's fix for the same shape of bug (`docs/pipeline/13-route-story-satellite-opening.md`)
is `_opening_label_windows`: it walks the WHOLE analytic camera path once, before any frame
renders, and never evicts an incumbent label to make room for a better-ranked newcomer. A 3D
leg cannot walk its whole path first: a leg's per-frame labels are the terrain-aware screen
positions `scene.js` computes in-browser for that pose alone (`map.project`, elevation-aware),
known only as each frame is actually captured -- there is no whole-leg sample list, and
capturing every frame twice (once to plan, once to draw) would double the render's slow part
(waiting for tiles to settle), not just its cheap screenshot readback.

- `layers._advance_active_labels` is the never-evict-an-incumbent rule extracted out of
  `_opening_label_windows` into a shared step function: given this step's ranked candidate keys,
  an already-active key keeps its slot regardless of rank, and a newcomer only takes a free one.
  `_opening_label_windows` still walks its whole analytic path up front and calls this once per
  sample; nothing about its own behaviour changed.
- `layers.LegLabelPlanner` applies the identical rule ONLINE, one real captured frame at a time,
  since that is genuinely all a 3D leg has available. `SceneMapRenderer.render_leg`'s SERIAL
  branch (`workers <= 1`) creates one for the leg's duration and clears it when the leg ends;
  `_render_frame` runs each frame's captured `labels` through `planner.step()` before
  `proof._draw_labels`. `step()` reorders (does not filter) the labels: an incumbent from the
  previous frame sorts first, so it wins `place_labels`'s collision greedy over a same-frame
  newcomer, instead of losing to whichever of the two the raw list happened to put first.
- **Wired into the parallel capture path too** (`workers > 1`): frames there are still
  striped across workers (worker *i* captures frames *i, i+N, i+2N, ...*, see
  `_capture_leg_parallel`), so no single worker ever sees two consecutive frames -- but capturing
  and finishing a frame are no longer the same call. `SceneMapRenderer._capture_frame` is the
  CAPTURE step alone (the slow browser render plus frame QA) and is what a worker actually runs,
  returning a `_CapturedFrame` (raw pixels + labels + HUD boxes) instead of a finished image.
  `_finish_frame` is everything after capture -- the never-evict reorder, `_draw_labels`, and the
  dashboard/profile/gap-badge draw -- and is cheap enough to run entirely on the consumer thread,
  which already drains `result_queues` in strict final frame order (`for index in
  range(len(times)): yield result_queues[index % len(ranges)]...`). `_capture_leg_parallel` builds
  one `LegLabelPlanner` for the leg and calls `_finish_frame` on each `_CapturedFrame` as it comes
  off the queue, in that same order -- the identical rule, driven by the identical class, as the
  serial branch's `self._leg_label_planner`, just applied in the consumer instead of inline in
  `_render_frame`. No frame is ever captured twice: only the pixel-drawing step moved.
- No fade-in/out ramp: `proof.place_labels`/`_draw_labels` draw every pill fully opaque and have
  no alpha-compositing path (unlike the opening's `_draw_opening_map_labels`), so a leg label
  still pops in/out at its true dwell boundary -- only the WHILE-IN-VIEW pop measured
  is fixed. Adding a fade would mean threading alpha through `place_labels`, `_draw_labels` and
  every other caller (`proof._render_one_frame`); left for a follow-up if wanted.

## Keep-clear zones for the dashboard rail and elevation profile

A leg label's pill could land on top of the dashboard rail or the elevation-profile chart --
both are drawn in the same frame, `place_labels` only ever knew about the marker's own
keep-clear box, and neither HUD element is "a label" so the greedy had no way to avoid them.

- `proof.place_labels`/`_draw_labels` take an optional `extra_keep_clear` sequence of
  `(left, top, right, bottom)` pixel boxes, folded into the same internal `keep_clear` list the
  marker box already uses -- a colliding pill gets the identical above/below recovery a marker
  collision gets, never a special case. The default is `()`, so every caller that predates this
  (including `film._repair_scene_frame`, which draws no dashboard/profile) is unchanged.
- `SceneMapRenderer._render_frame` resolves `layout.hud_layout(width, height)` once per frame
  and passes the dashboard-rail box (`dashboard_left`/`dashboard_panel_top`/`dashboard_right`/
  `dashboard_panel_top + dashboard_panel_height`) and the elevation-profile box (`profile_left`/
  `profile_top`/`profile_right`/`profile_top + profile_height`) as `extra_keep_clear` to
  `_draw_labels`. The `hud_layout()` call moved earlier in the method, before `_draw_labels`
  runs instead of after, purely so the boxes exist in time -- nothing else about the frame's
  draw order changed, and the later `draw_profile` call now reuses that same `layout` instance
  rather than resolving it a second time.
- **A visible photo pin's keep-clear box is out of scope here.** A photo pin's pixel
  box is planned in `layers.LegLabelPlanner`/drawn in `scene/scene.js`, neither of which this
  change touches or has the geometry for -- `hud_layout()` has no photo-pin field to read. Adding
  it is a later addition once a pin's screen box is available to whichever module plans it.
