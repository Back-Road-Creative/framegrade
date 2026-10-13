## Route Story film (`gomoveshift/pipeline/route_story/film.py`, `layers.py`, `hud.py`, `assembly.py`)

`render_route_story_film` (`film.py`) is the local private-export composition layer that sequences scenes; it composites a credited national raster, optional credited regional raster levels, and a separately credited local raster through one frame-driven geographic camera. Each frame selects one raster that fully covers the viewport; partial rectangles are never pasted over another level, preventing hard seams and grainy handoffs. The compact dashboard rail reserves map space for the marker and uses separate pace and speed cells. A derived display route removes long gaps and implausible GPS jumps before telemetry or route animation, while the source GPX remains the evidence in the receipt. RAW stills pass through the measured photo developer; only a nearly all-digital-black outer edge is cropped, with no generated replacement pixels.

The viewport crop is warped with Pillow's `QUAD`, whose source corners run NW, SW, SE, NE — not clockwise. Listing them clockwise transposes the picture (east at the bottom, south on the right) under labels that stay north-up; every earlier Pillow map view shipped that way, and `tests/test_route_story_raster_orientation.py` now pins north-up, east-right on a four-colour raster.

The map projection, label selection and on-map drawing (raster compositing, the opening camera move, the route line and marker, orientation cues, public-name labels) live in `layers.py`. Overlays that are not the map itself — the dashboard, title/recap/photo-label cards, and the small unit-conversion formatters and still-photo fitting helpers they share — live in `hud.py`. Turning rendered frames into encoded clips (per-leg/per-photo frame sequencing, `ffmpeg`/`ffprobe` invocation, and source-video evidence reading) lives in `assembly.py`.

A RAW photo hold has two possible pictures, chosen with `--still-source` (`hud.STILL_SOURCES`, recorded as `export.still_source` in the receipt). `developed`, the default, is the service's measured develop of the sensor data. `camera` shows the full-size JPEG the camera embedded in the RAW — the camera's own white balance, tone curve and colour, the picture the photographer saw on the back of the body. The developer's natural grade can hold a still visibly darker and less saturated than the camera rendering, which reads as a dull photo between bright satellite legs; the flag is chosen per film and changes nothing about the developer.

`hud.still_card_frame(still, width, height, progress)` is the single recipe for what a photo hold
looks like at any point (0.0 -> 1.0) across its seconds on screen; `assembly._append_photo` calls it
once per delivery frame instead of building the card itself. A still whose aspect ratio is close
enough to the frame's (`hud.STILL_FULL_BLEED_MIN_ASPECT_MATCH`, measured as `min(ratio)/max(ratio)`
so 1.0 is exact) goes full-bleed edge to edge -- a 3:2 or 4:3 photo in a 16:9 frame both clear it --
with no blurred backdrop and no shade. A narrower/portrait still keeps the blurred, shaded backdrop
(`hud._photo_frame`) but the foreground now fills the frame's full HEIGHT with no inset, where it
used to sit at a fixed 0.91-0.945 of the frame. Either way the still pushes in continuously from
1.0x to `hud.STILL_PUSH_IN_MAX_SCALE` across the hold, eased in and out (`hud._ease_in_out`,
smoothstep) and sampled sub-pixel per frame via `hud._zoomed_resize` -- one Pillow `resize` with a
float `box` that shrinks toward the centre as the scale grows, from the single still
`_append_photo` decodes once and bounds with `hud.prepare_still` -- sized to what the card COVERS
(the whole frame for a full-bleed still, the frame height for a portrait one) times
`STILL_PREPARE_MARGIN` (1.25), so even the fully pushed-in frame is a downscale; a contain-fit
bound once left a 4:3 photo ~13% stretched and soft at full push-in -- never a fixed ladder of prefiltered sizes stepped through
per frame, which used to read as a visible 6-step jerk.

`hud._camera_preview` turns the preview upright (its own EXIF orientation, else
LibRaw's `flip`) and refuses one narrower than `CAMERA_PREVIEW_MIN_WIDTH`; a RAW with no usable
preview logs a warning and is developed instead, so a film never loses a hold. A file Pillow opens
directly (JPEG, PNG) is unaffected by the flag.

A film render reports its own progress. `assembly._append_map_leg` logs, every
`PROGRESS_EVERY_FRAMES` map frames and again when a leg finishes, the leg's frame count, the
measured seconds per frame, the seconds left in that leg and the running film frame number; the
film CLI turns that on with `configure_progress_logging()` (INFO to stderr, so a run's log file
collects it). Without it a multi-hour render prints nothing, and the only way to estimate one is
to count JPEGs in its temp directory. Logging never changes a
frame: the lines are written after the frame is saved.

The film CLI delivers **1920x1080 at 30 fps** unless a run names another size
(`DELIVERY_WIDTH`/`DELIVERY_HEIGHT`/`DELIVERY_FPS`). A low-resolution default would let a run
that said nothing about size quietly produce a preview.
Below the moving-speed threshold, the PACE and SPEED cells read "stopped" rather than an absurd pace
or a blank dash; above the plausible-travel ceiling (a rejected GPS jump) they still read the unknown
dash. GRADE follows the same two limits (`hud._moving_grade`): it is rise over the last edge's
distance, which standing still is GPS jitter (a "stopped" pace could otherwise sit beside a large negative grade) -- so it
shows the dash unless the speed is one the rail would print. The dashboard's HR/CAD/TEMP cells are optional and appear only when the current frame's
`DashboardState` actually carries that value, so a GPX without a Garmin `TrackPointExtension` shows
the same five-cell rail as before. Attaching a visible-label interval to the receipt needs `layers.py`
wiring and is left for a later pass.

A map-leg frame is produced through the `MapRenderer` protocol (`layers.py`): `name`, a
per-frame `render(segments, at_time, width, height) -> Image`, and a leg-level
`render_leg(segments, times, width, height) -> Iterator[Image]` covering one leg's whole
list of route-clock frame times up front. `assembly._append_map_leg` calls only `render_leg`
(never loops `render` itself) and `render_route_story_film` depends only on this protocol, not
on satellite rasters directly. `PillowMapRenderer` is the existing raster-composited
implementation and the default when `render_route_story_film`'s `map_renderer` keyword is left
unset; its camera has no inertia, so its `render_leg` is exactly a per-frame `render` loop with
byte-identical output. The film receipt records which renderer produced the film under
`map_renderer` (`"pillow-2d"` by default). Opening-camera and recap frames still call
`_render_map`/`_render_opening_map` directly — only map-leg sequences go through the renderer seam.

The recap ("the day, at a glance") is the one Pillow map that does not follow the marker:
`film._render_recap_map` asks `layers._render_map` for `overview=True`, and `layers.overview_viewport`
fits the route's bounding box (`OVERVIEW_MARGIN` either side). A marker-centred view at the end of the
day showed only the last corner of the route — and, near the edge of the finest raster, a viewport
that raster no longer fully covered, so the recap fell back to a coarser level and drew from blocky
upscaled pixels. The overview gives up margin, down to
`OVERVIEW_MIN_MARGIN`, before it gives up the finest raster: when the route fits inside it the view is
slid to lie wholly inside it; a route larger than it is still shown whole, from the coarser level.

<a id="route-story-recap-rail"></a>
**The recap's numbers live in the rail, never over the map.** The recap map is drawn with no
dashboard, which left the bottom rail an empty bar while a stats panel laid over the map hid the
start of an out-and-back route. `hud._draw_recap` now writes the title and the four day totals
(`hud._recap_cells`: distance, ascent, time, moving average) into that rail, in the dashboard's own
type and geometry, and fades only the text in — the map stays clear edge to edge for any route shape.
The fade takes the recap's first `hud.RECAP_FADE_SHARE` and the numbers then hold fully lit: a fade
across the whole recap left them readable on its last frame only.

`SceneMapRenderer` (`scene_renderer.py`) is the second `MapRenderer` implementation: it draws a map-leg
frame as an oblique follow shot over the offline MapLibre terrain scene (`scene/capture.py`) instead of
the flat Pillow satellite composite. Per frame it reuses `layers._point_at_time`'s gap handling (nearest
point plus the gap badge), builds a pose centred on the route position at that clock time with a fixed
follow zoom clamped to the bundle's own `planner.max_shot_zoom`, and lights it with the real sun at that
route clock time (`sun.light_for_pose`, the same function `proof.py`'s chase shot uses). The captured
frame is composited under the same dashboard (`hud._draw_dashboard`) and scene-projected labels
(`proof._draw_labels`) the Pillow path draws, so the overlay is identical between renderers.

The route line on the map is built from geometry fed from the SAME pose the marker/camera are built
from -- never a separate fraction re-applied to a differently-segmented geometry
(`prepare._route_geojson` can split the bundle's `route.geojson` differently than
`film.clean_display_segments` splits the display track, which is what let the bright "travelled"
line drift ahead of or behind the marker). `scene_renderer.py` puts the whole display track on
EVERY pose (`pose["route"]`, the dim/base line) and the already-travelled portion on every pose too
(`pose["travelled"]`, the bright line) -- one coordinate list per display segment that has started,
cut at exactly the same GPS-gap boundaries the segments themselves are split on, whose last vertex
is always the marker's own coordinate for that pose (interpolated mid-segment, or held on its gap
point). Both ride on every pose, never a "send once" flag on the renderer instance: a pose dict is
the frame cache's key (`scene/frame_cache.py`), so it must fully determine the frame's pixels on
its own -- a cached hit for an early pose, or a fresh browser page opened mid-leg, must never leave
a later pose's page without a route because some earlier "first" pose already claimed to have sent
it. `scene.js`'s `drawRouteOverlay` is what actually paints both lines, straight off `pose.route`/
`pose.travelled` onto the screen-space `#overlay` canvas (see `docs/pipeline/41-travelled-line-minimum-width.md`);
the `route`/`travelled` MapLibre GeoJSON sources and their `route-casing`/`route-line`/
`travelled-line` layers are `visibility: "none"` and paint nothing. `renderPose` does not feed those
sources any more either (removed, see `docs/pipeline/41-travelled-line-minimum-width.md`)
-- `setRoute`/`setTravelled` are still defined, just uncalled, so the layers' paint properties stay
documented and easy to wire back up if the overlay is ever reverted.

Scene label pills are placed before they are drawn (`proof.place_labels`). The text height follows the
frame (`proof.label_font_px`: `LABEL_FONT_FRACTION` of the frame height, floored at
`LABEL_FONT_MIN_PX` — about 23 px at 1080p, 15 px at 720p) in DejaVu Sans Bold, falling back to
Pillow's own scalable face, never the fixed ~10 px bitmap font a 1080p film could not be read in.
Labels are taken in the label file's order and a pill that would touch one already placed (with a
small gap) is left out: where several names project within a few pixels of each other,
they would otherwise draw as a stack of pills covering one another. A hidden or nameless label is never placed.

<a id="route-story-label-clear-marker"></a>
**No pill covers the traveller.** When the marker stands on a labelled point, that label's pill
would otherwise be drawn straight over it and the film's subject would vanish. `scene.js` now appends the marker's own projected position to the label
list it returns per pose (`kind: "marker"`, no name -- projected by the same camera and terrain as
the labels, so it is where the marker was actually drawn), and `proof.place_labels` treats that
entry as a keep-clear square (`proof.marker_clear_box`, `MARKER_CLEAR_FRACTION` of the frame
height). A pill that would touch it moves just above the marker, else just below, else is left out;
a pill elsewhere is placed exactly as before. The marker entry never becomes a pill.

**No pill runs off the side of the frame.** A pill is anchored at its label's own `x`, and only the
top and bottom edges were guarded — so a label near a side edge would otherwise draw its box past the frame and the
name come out cut in half. `place_labels` now slides the box back inside
instead of dropping it: the label still belongs to a feature the viewer can see, and a whole name a
few pixels off its anchor reads better than half a name on it. The slide happens before the
keep-clear and collision checks, so a pill that slides onto an earlier one is still left out, and a
name whose pill is wider than the frame has no honest placement at any position and is dropped
outright — which is why a fixture label on a toy canvas has to be short enough to fit it.

The camera bearing needs a leg-wide plan, not a per-frame value: a film map leg routinely compresses
thousands of route seconds into a handful of display frames (a whole drive day into one ~14 s leg is
common), so two adjacent *displayed* frames can be tens of real seconds apart on the route — long
enough for the road to have turned by more than any camera could plausibly pan in 1/30 s, and for a
short-range heading (even smoothed over a fixed window) to still just resample GPS jitter and
switchback noise at that scale. `render_leg` fixes this by planning the whole leg before rendering any
frame. `_chord_bearing` targets the route's travel direction at the LEG'S OWN scale: the bearing of the
chord from `at_time - W` to `at_time + W`, `W` a fraction of the leg's own route-time span
(`_CHORD_WINDOW_FRACTION`, close to the whole leg — a leg that doubles back on itself needs the window
near its own endpoints before the target stops requiring more net rotation than
`bearing_speed_deg_s`/`bearing_accel_deg_s2` can physically cover in the frame budget) and floored at
`_MIN_CHORD_WINDOW_S`; below `_MIN_CHORD_DISTANCE_M` the chord holds the previous target rather than
chase a turnaround or a stop. `_plan_bearings` then chases that target sequence under a second-order
(trapezoidal) motion profile — each frame's desired angular velocity is capped by both
`bearing_speed_deg_s` and the velocity that lets it decelerate to the target without overshoot
(`sqrt(2 * bearing_accel_deg_s2 * |delta|)`), so both `planner.assert_smooth`'s speed and acceleration
limits hold on every frame, not just speed. `SceneMapRenderer` carries the last displayed bearing AND
angular velocity across `render_leg` calls (it is not a frozen dataclass, unlike `PillowMapRenderer`:
the follow camera has real inertia) so the camera never snaps to a raw target or a dead stop at a leg
boundary. On a long, heavily time-compressed route, a
per-frame independent bearing swings far past the speed cap on most frames, and a speed-only limiter with
too narrow a chord window sits at the cap for nearly every frame with a large heading lag. With the wide chord window and second-order limiter both speed and acceleration stay within their `DEFAULT_MOTION_LIMITS` caps, only a small fraction of frames touch the speed cap, and the median heading lag is near zero. `film.main`'s `--renderer scene` (paired with `--bundle
<route-story prepare output>`) selects it; `--renderer scene` without `--bundle` fails argument parsing
immediately rather than partway through a render. One headless-Chromium session captures a frame in
≈3-4 s, so `--workers N` (default: `proof._resolve_workers`' own resolution) fans a map leg's CAPTURE
step across `N` threads. `render_leg` still plans every bearing serially first — the follow camera's
inertia state is never touched from more than one thread — then each worker opens its OWN capture
from `capture_factory` (Playwright's sync API is not thread-safe across threads sharing a browser) and
renders a striped slice of frame indices into a bounded per-worker queue. The consumer drains the
queues strictly in frame order, so output order and every pose match a serial render, a leg of
thousands of multi-megabyte frames is never held whole, and a worker failure re-raises the original
exception rather than encoding a partial leg. `workers=1` or no factory is the serial path.

**By default that machinery is dormant, because scene browsers may not paint concurrently (only ONE scene browser can paint at a
time).** `proof._resolve_workers` clamps any requested count to
`scene.capture.max_concurrent_captures()` (default 1) and logs when it does, so `--workers 6` renders
serially instead of failing. At the film's own capture size, a single capture reaches ready in a couple of seconds, but a second one, whether in the same process or in a SEPARATE OS process, can fail to reach ready at all, reporting style
loaded, tiles loaded, labels loaded, zero tile failures and `loadFired: false` — the scenes load and
never paint. Starting the second one long AFTER the first has settled does not help it either, while
the first keeps painting normally, so this is not a startup race.
Chromium's anti-backgrounding flags (`--disable-renderer-backgrounding` and friends) do not change
it, so it is neither throttling, shared
memory, nor memory pressure. Without the clamp such a render dies only after a long prepare and preflight.
`GMS_ROUTE_STORY_MAX_CAPTURES=N` lifts the clamp on a host where browsers do coexist.
A lone browser can stall with that same
signature, with `map.loaded()` already true and no frame pending — MapLibre fires `load` only
from inside a render pass, and schedules a pass only when something is dirty, so a map whose last
source finished between two passes never gets the pass that fires it. The scene's `isReady()` poll
now calls `map.redraw()` in exactly that state (loaded, `load` not yet fired), which is that pass;
with that fix lone browsers become ready within seconds. The concurrency clamp predates this fix and
stays in place. Since
parallelism is unavailable either way, the GL backend is now purely a speed choice:
`GMS_ROUTE_STORY_GL_BACKEND=gpu` reaches ready quickly and captures at roughly twice
the software rate.

Public map context is supplied as a local JSON label manifest rather than fetched while rendering.
`load_map_labels` validates coordinates, and `select_visible_labels` applies per-kind visibility bands
(towns at regional scale and peaks/gaps at local scale) plus stable priority ordering before
`_draw_map_labels` places names with a reserved-space and collision check. The receipt includes the
label names and kinds so the geographic context is auditable. `TerrainGrid` consumes a bounded USGS
3DEP sample grid and adds deterministic relief shading to the satellite view; it is source-derived,
not generated landscape detail.

The renderer accepts only regular original media files. A still is shown at a map moment only when
its `MediaAnchor` was produced from embedded coordinates or a short observed timestamp edge.
All imagery and terrain inputs are loaded from local files, and the renderer has no network fallback.

<a id="route-story-located-beat"></a>
**A located video plays where it was shot, not as a trailing interlude.** Video creation-time and
location tags are read through `ffprobe`. When the resulting `MediaAnchor` resolves to an
undisputed route clock time (`track_time_s` set, status not `coordinate_time_conflict`),
`film._ordered_media_beats` places it among the mapped photos in true route-clock order --
`film._map_leg_bounds` no longer always appends the video leg after every photo leg. When the
render loop (`render_route_story_film`) reaches that beat: the map leg flies to its route moment,
a 1.2 s "VIDEO ARRIVAL" badge shows, the frame sequence is split there (`main-1-frames`,
`main-2-frames`, each numbered from 0), the clip itself is encoded and concatenated in place with
its own near-silent audio, and the map resumes into the second part from the same moment to the
end. A photo with no defensible coordinates, or a video whose anchor never resolves (absent, no
track time, or a GPS/clock conflict), is **unplaced media**: `render_route_story_film` never draws
it -- it is not appended after the final map leg, and there is no longer an end-of-film tail or
title card for it. `film.main` computes the same "would this be left out" check upstream of any
renderer or `SceneCapture` construction and, by default, refuses to render at all
(`parser.error`, naming every unplaced file and its anchor status) until an operator passes
`--allow-unplaced`; either way the receipt lists it under `unplaced_media`
(`media_id`/`anchor_status`/`reason`). Each `photos[]` and `source_video` receipt entry also now
carries `interpolation_method`, `rejected_gps` (`{"latitude","longitude"}` or `null`), and
`required_speed_mps` straight off the `MediaAnchor` that placed it.

`assembly._interlude_card`/`_interlude_subtitle`/`_video_backdrop` are no longer called from the
render loop -- the located-beat above replaced the old "unlocated interlude" tail entirely -- but
stay in `assembly.py` because `tests/test_route_story_interlude.py` still exercises them directly;
a later pass may retire both together. `assembly._encode_source_video` still does not pad a
portrait clip with flat bars: the clip is split, one copy cover-scaled, blurred and dimmed
(`_BACKDROP_DIM`) behind, the other shown whole and undimmed on top, so a 9:16 phone clip fills a
16:9 frame with its own picture -- unchanged by the beat move, since the clip's own encode is the
same either way.

**A small pin stays on the map where each played photo/video beat happened.** `film._beat_pins`
resolves every entry `film._ordered_media_beats` produced to its own anchor coordinates --
`(time_s, longitude, latitude)` -- once, right before the render loop starts, and
`dataclasses.replace`s that onto whichever `MapRenderer`'s own `pins` field (only a renderer that
declares one; a caller-supplied renderer with no `pins` field renders exactly as before).
**Start and finish pins.** Separate from the photo pins above and fixed for the whole film:
`layers.route_endpoints(segments)` returns `((lon, lat) start, (lon, lat) finish)` from the
track alone -- there is no operator override and nothing is inferred. The finish is the last fix
of the LAST segment, not of the first: a recorder pause splits the day into several segments and
the day does not end at the first pause. A track with no points returns `None`; a one-point track
is degenerate but not an error, and reports the same place twice.

Both are drawn after the photo pins and before the marker, so an endpoint outranks a photo pin
sitting on it and the marker outranks everything -- at route-second 0 the marker IS the start,
and a pin painted over the marker would hide the one thing the frame most needs to show.
`layers.ROUTE_START_RGB` and `ROUTE_FINISH_RGB` are each far in colour from
`frame_qa.TRAVELLED_LINE_RGB`, from the marker's amber and from the white photo pin: a pin lying
on the route in a near-enough yellow would be measured as travelled-line width, and the gate
built to catch a hairline route would pass on the pin instead. `scene.js` carries the same two
values as `ROUTE_START_COLOR`/`ROUTE_FINISH_COLOR` and a test asserts they match the Python
constants byte for byte -- the scene renderer is the production path, so a pin that exists only
in `PillowMapRenderer` does not ship at all, and two renderers that disagreed would change the
colour of the same place mid-story. `SceneMapRenderer` sends them on the pose as `endpoints`,
omitted entirely for an empty track so a pose that predates the feature stays byte-identical and
the frame cache keeps its frames.

Unlike a photo pin the finish is present from frame one: the untravelled route line already runs
to it, so withholding its pin would only make the map disagree with its own line. Both pins sit
under the translucent dashboard rail where the route passes through the lower map field, which
dims them -- longstanding behaviour they share with the photo pins.

`layers.visible_pins(beats, at_time)` is the one pure filter both renderers and the recap map draw
from: a beat is included once `at_time` reaches its own `time_s` (`<=`, not `<`) -- the map leg
that resumes right after a still starts exactly at that beat's time, so the pin is already there on
the very first frame after it, never before the still has played and never a frame late -- and two
beats within 15 m of an already-kept pin collapse to one (kept in beat order, so the first wins).
`PillowMapRenderer._render_map` draws each visible pin (white fill, dark outline, radius scaled
with frame height) after the route/travelled lines and before the marker, so the marker still
paints over a pin at its own exact spot. `SceneMapRenderer` instead sends the filtered
`(longitude, latitude)` pairs on the pose as `pins`; `scene.js`'s `drawRouteOverlay` draws them on
its screen-space overlay canvas the same way -- after the travelled line, before the marker's own
destination-out knockout, so a pin under the marker's current position is knocked out with it. The
pin's white/dark colours are deliberately far (by `frame_qa`'s tolerance) from both
`TRAVELLED_LINE_RGB` and the marker fill, so a pin beside the marker never perturbs
`frame_qa.travelled_line_min_width`'s measurement. The film receipt carries the count under
`photo_pins`.

<a id="route-story-best-pick"></a>
**Best-of-look-alike-group photo selection (`gomoveshift/pipeline/route_story/select.py`,
`still_selection.py`).** A photographer shoots several frames of the same view; the film shows only
the best one of each look-alike group, never a loser. `film.main` runs this over already-placed
(`is_mapped`) photos only -- an unplaced photo is refused/logged by the existing gate above and
never reaches the selector. `still_selection.build_photo_candidates` decodes each photo's preview
through the same `hud._load_still` the film render uses (honouring `--still-source`) and feeds the
selector the EXIF-derived route-clock time already computed by anchoring (`anchor.track_time_s`) --
never file mtime -- and the anchor's resolved lat/lon as placement, when both are known.
`select_best_photos` chain-links consecutive candidates (by capture time, then name) into groups
where every adjacent pair passes a time-gap test, a placement-distance test, and a visual
"same view" test (ORB keypoints, histogram-correlation fallback); each group's winner is scored by
sharpness, then exposure, then colour vibrancy. Only `.kept` photos become `PhotoMoment`s the film
render loop ever sees or draws; losers are dropped from the render entirely but never silently --
every group, its winner, and every loser's name is written to the receipt.

- `--no-best-pick` disables selection (`SelectionOptions(enabled=False)`): every photo is kept, one
  single-member "group" per photo, `photo_selection.enabled` is `false`.
- `--keep NAME` (repeatable, case-insensitive match against a file actually in `--photos-dir`;
  an unknown name is a parser error naming the available files) rescues a specific loser alongside
  its group's winner. Passed through as `SelectionOptions.keep`.
- **Pacing cap** (`still_selection.resolve_photo_seconds`, a pure function): kept stills may not
  take more than `STILLS_SHARE_CAP` (35%) of the film's total runtime (intro + the map seconds this
  run will actually use, computed the same way `render_route_story_film` itself would via
  `_map_leg_bounds`/`plan_map_legs` over the *kept* photos + kept-stills seconds + recap; the
  located video's own duration is left out of the denominator, since it is not cheaply knowable
  before its own encode). When `kept_count * --photo-seconds` would exceed the cap, every still's
  seconds shrinks evenly to fit, but never below `PHOTO_SECONDS_FLOOR` (2.5s, matching
  `--photo-seconds`'s own default).
- **Stills over the cap at the floor are trimmed, not just warned about**: before pacing runs,
  `main()` calls `still_selection.max_stills_for_cap` (a pure function: given the intro + map +
  recap seconds, the largest kept-still count that still fits `STILLS_SHARE_CAP` at
  `PHOTO_SECONDS_FLOOR`) and drops the lowest-ranked survivors down to that count -- a bookend is
  dropped before any mapped still (it carries no route position of its own), and within each tier
  the lowest-sharpness photo (from the scores `select_kept_photos` already computed) goes first.
  Dropped names are recorded on `stills_pacing.dropped_for_cap`. `resolve_photo_seconds`'s
  `floor_hit` (`stills_pacing.floor_hit: true`) is kept as a guard for a shape this trim doesn't
  reach -- it should not fire in a normal run. Only wired for the `--map-seconds` (explicit or
  default-auto) path; `--target-seconds` mode still relies on `solve_map_seconds`'s own clamp.
- Receipt: `photo_selection` is `SelectionResult.to_dict()` (every group/winner/loser/score) plus
  `{"enabled", "kept_override", "capture_time_source": "exif"}`; `stills_pacing` is
  `{"requested_seconds", "used_seconds", "stills_share", "cap", "floor_hit", "dropped_for_cap"}`.
  Both blocks are written for a `--preflight-only` run too (before the long render), so an operator
  can review which photos were dropped from the receipt alone.
- The operator review pack (sheets, JSON, and the delivered film's hectic-motion clips) is
  documented in `docs/pipeline/152-route-story-review-pack.md`.

### Metadata

`route_story/metadata.py:build_route_story_metadata(receipt, *, recorded_on, title=None, credits=(),
published_titles=None, ledger_status=None, activity=None)` turns a `film_receipt` dict into upload
metadata (title/description/tags/`format`/`playlist_hint`), pure-Python and offline. The derived title
(`"<Activity> <route name>"`) mirrors `hud.resolve_opening_title`'s own default-card logic: the
same first/last label pair and arrow separator, so the burned-in opening card and the shipped
video title never disagree when no `--title` is given. `Activity` comes from the explicit
`activity` argument (a resolved `route_story.activity.ActivityProfile.name`) when given, else from
`route_summary.moving_average_speed_mps` against `HIKING_MAX_SPEED_MPS`/`RIDING_MAX_SPEED_MPS`
— editorial display-copy bands, not physical constants, operator-adjustable in `metadata.py`.

`film.py`'s own `--activity {hiking,cycling,driving,auto}` (default `auto`) resolves one
`route_story.activity.ActivityProfile` via `resolve_activity(explicit, observed_max_speed_mps)` --
an explicit name wins outright; `auto` (the default) falls back to the same speed-band guess above.
The resolved profile's `max_speed_mps` becomes the display speed filter's cap everywhere it's
applied (`drawn_segments`, `render_route_story_film`, `film_receipt`'s own qualified-segment
filter, replacing the previous universal `8.0`); the receipt's `display_filter` records the
resolved `activity` name and `activity_source` (`"explicit"`/`"auto"`). A profile YAML's
`activity:` key maps onto this flag like any other (`gomoveshift/pipeline/route_story/profile.py`). Title
uniqueness is checked against the published-titles ledger (`longform_metadata.read_published_titles`)
via `check_title_against_ledger`; an `UNREADABLE` ledger status hard-fails (`LedgerUnreadableError`)
rather than silently skipping the dedup check. The builder emits `format: "route_story"`
(`playlist_hint` on the returned metadata) for `playlist_resolver.resolve_playlist` to route
against a `{match: {format: "route_story"}, playlist_name: …}` config rule.

CLI wiring — a `--credit` flag on `film.py` and writing the built metadata JSON beside the
receipt — is not implemented.

### Units

`film.py`'s own `--units {metric,imperial,auto}` (default `auto`) resolves display units via
`route_story.activity.resolve_units(explicit, centroid)`: an explicit value wins outright;
`auto`/absent falls back to `route_story.activity.region_units(latitude, longitude)` on the
track's centroid (`track_centroid(selection.segments)`) — imperial inside the continental US,
Alaska, Hawaii, Puerto Rico, or the UK, metric everywhere else (Ireland included); no centroid
(no GPS) defaults to metric. The receipt's `display_filter` records the resolved `units` and
`units_source` (`"explicit"`/`"region"`/`"default"`), and the receipt's top-level
`mapping_version` carries `route_story.activity.MAPPING_VERSION`, bumped whenever the activity
table, the region table, or the units rule changes. A profile YAML's `units:` key maps onto this
flag like any other (`gomoveshift/pipeline/route_story/profile.py`). This is data-layer only — HUD
rendering of km/h vs mph from these fields is not implemented; `hud.py` is untouched.

### Pacing mode

`film.py`'s own `--pacing-mode {recorded}` (default `recorded`) names the presentation-timing
table explicitly. `recorded` is the only implemented value today — a real, disclosed default
establishing the versioned-in-the-receipt contract before a second value is added; an
unrecognized value fails fast at argument parsing. The receipt carries it verbatim as
`pacing_mode`, plus a dedicated `presentation_mapping_version` (`route_story.film.
PRESENTATION_MAPPING_VERSION`) bumped whenever the pacing table changes — versioned
independently of `mapping_version` (`route_story.activity.MAPPING_VERSION`, which versions the
unrelated activity/region/units tables), so a re-render under a different pacing table is
distinguishable from a re-render under a different activity/units table. A profile YAML's
`pacing-mode:` key maps onto this flag like any other (`gomoveshift/pipeline/route_story/profile.py`).

### Thumbnail and metadata outputs

`--thumbnail <path>` runs `drive_phases.thumbnail_generator.generate_lf_thumbnail` against the
delivered `--output` after the delivery check, reusing the same `resolve_opening_title` the opening
card burns for its title/subtitle text. It only runs on the `full` preset -- a `social` (9:16) cut
has no long-form upload for a thumbnail to sit beside, so `--thumbnail` is a no-op there even when
passed. A generator miss (`None`) records `receipt["thumbnail"] = {"status": "failed"}` rather than
failing the export -- the film is still good even without a thumbnail; a hit moves the chosen
candidate to `--thumbnail` and records `{"path", "sha256"}`.

`--metadata <path>` builds upload metadata (title/description/tags) from the assembled `film_receipt`
via `route_story.metadata.build_route_story_metadata` (`--credit` supplies extra attribution lines
beyond the imagery credits already on the receipt), checks the resolved title against the
published-titles ledger with `check_title_against_ledger`, and writes `dataclasses.asdict(meta)` as
JSON to `--metadata`. A duplicate or unverifiable title calls `parser.error` -- the run fails before
anything is written, rather than silently publishing a re-used title. The receipt records
`{"metadata": {"path", "title", "format"}}`. When `route_story/metadata.py` isn't present on a given
branch, the import is attempted lazily and a failure degrades to
`receipt["metadata"] = {"status": "builder-unavailable"}` instead of crashing the export.

**Preflight**: with `--metadata`, `main()` calls `metadata.preflight_title` right after
activity/units resolve, well before `render_route_story_film`, sharing title derivation with
`build_route_story_metadata` (both call `_resolve_title`) so a duplicate/unverifiable title fails
before a wasted render, not after. `check_title_against_ledger` post-render is still the
authoritative, fresh check.

### Route trim

`--trim-start-m`/`--trim-end-m` drop points near the route start/end via `route_privacy.trim_route`,
applied right after `select_gpx_day` so every downstream surface only sees the trimmed route. See
`150-route-story-route-privacy.md`. The receipt carries the applied trim under `route_trim`.

`--privacy-zone LAT,LON,RADIUS_M` (repeatable; an out-of-range latitude/longitude or a radius that
is not above 0 is refused as a usage error before anything runs) masks an interior privacy zone via
`route_privacy.mask_zones`, applied immediately after the start/end trim above -- same insertion
point -- so every downstream surface (map, dashboard, opening, high-point badge, thumbnail,
metadata) only ever sees the zone-masked segments. A zone that splits a segment yields two kept
segments with a real gap between them; a zone that would mask the entire route raises the same
`RouteTrimError` family as an over-trim, reported via `parser.error`. When any zones were applied,
`route_trim` gains a `zones` block (`{"count", "removed_points", "removed_m"}`) -- counts and
removed points/metres only, never a zone's center or radius. See `150-route-story-route-privacy.md`
"Interior privacy zones".

`--public-receipt PATH` additionally writes a distribution-safe copy of the receipt (see
`150-route-story-route-privacy.md`'s `public_receipt.py`) alongside `--receipt`'s private one, once
the film export finishes. It requires `--receipt` and a nonzero `--trim-start-m`/`--trim-end-m`;
`main()` refuses with a `parser.error` before anything is written — before the trim even runs — when
either requirement is unmet, so a route with no explicit privacy trim can never gain a public
artifact. The private `--receipt` output is unaffected by requesting `--public-receipt` alongside
it.

## Inside a drive: the trail film (`--trail`, `pipeline/drive_phases/trail_film.py`)

`gomoveshift drive` can render a Route Story film beside each LF day. `attach_trail_film(deliverables,
state, config)` runs from Phase B/C's per-day callback, after the day's LFs render and before the day is
recorded, so the outcome persists (and resumes) with the day's `Deliverables`:

1. Resolve the day's GPX from `state.gpx_files` (parallel to `state.episode_dates`).
2. Look up the prepared imagery bundle with `prepare.find_prepared_bundle`. The "prepare" step is a lookup
   only; the network fetch never runs inside a drive, so run `gomoveshift route-story-prepare` first.
3. Render a cheap terrain proof clip (960x540, 24 fps) through `proof.render_proof_with_receipt` -- the same
   entry `route-story-proof` uses -- and re-check its receipt with `verify.verify_receipt`.
4. Render the film with `film.main`. The imagery rasters and credits come from the Route Story launch
   profile named by `--trail-profile` (see `route_story_profiles/README.md`); the day's GPX, date, output,
   `--renderer scene`, the verified bundle and the LF override the profile. The LF is the placed source
   clip: the day's lowest-index regular LF.

The film is a sibling deliverable: `Deliverables.trail_films` (a `TrailFilm` with the LF index, path and
receipt), written to `day_<date>/trail/trail_film_<date>.mp4`. Phase D does not publish it yet.

`--trail` / `--no-trail` is tri-state. Neither flag is auto: on when the day has a GPX and a prepared
bundle. `--no-trail` is a silent no-op. A day that cannot get a film (no GPX, no prepared bundle, no
`--trail-profile`, a failed proof or verify, a film that exits nonzero) logs a warning and records the
reason in `Deliverables.trail_skips`; it never fails the drive, because the LFs are the primary product.
A skip is recorded with the day; on resume it is restored and, once its cause is resolved, retried
(trail step only). The film's photo slice and the resume retry are in `47-trail-film-photos-and-resume.md`;
its title (the LF's label) and the coverage gate on its GPX span are in `48-trail-film-label.md`.

## Motion summary: a before number for "does this feel hectic"

`planner.motion_report`/`assert_smooth` only ever see the WORST frame of a plan -- pass/fail against
`DEFAULT_MOTION_LIMITS`, nothing else. That answers "is this plan legal," never "is this plan calmer
than the last one." `planner.motion_summary(report, fps, limits=None)` (and
`planner.summarize_plan_motion(plan, fps, limits=None)`, which composes `motion_report` +
`motion_summary` for one call) reduce the same per-frame series to `mean_abs`, `p95_abs` (nearest-rank
percentile of `|value|`), `max_abs`, and `frames_over_limit` for bearing speed and bearing acceleration,
plus `pitch_speed_max_abs`, `zoom_velocity_max_abs`, `frame_count`, `fps`, and the `limits` dict used --
the camera path only, nothing about rendered pixels, terrain, labels, or grade.
`MotionSummary.as_receipt()` returns the plain JSON form under a `"motion"` key (3-decimal rounding).
`planner.motion_report_from_poses(poses, fps)` builds the same series from any pose sequence, not only
a `ShotPlan` -- `motion_report` is now a thin wrapper over it, so the two can't drift. A pose needs
`bearing`/`pitch` (`opening.CameraPose`) OR `bearing_deg`/`pitch_deg` (`continuity.CameraState`, a
scene renderer's rendered poses), plus `zoom` either way. `SceneMapRenderer` accumulates every
`render_leg` call's `CameraState`s, in render order, onto `rendered_poses` (parallel to
`camera_continuity`, appended beside it once a leg passes `check_camera_continuity`) -- the path the
film actually rendered, never a `ShotPlan` `film.py` doesn't build. `film_receipt` accepts a
`motion: dict | None` keyword and records it verbatim when given; left `None`, it computes one itself
from `map_renderer.rendered_poses`/`.fps` once there are at least two poses, so a scene-rendered film's
receipt carries real motion with no caller wiring a `ShotPlan` by hand. No `rendered_poses`
(`PillowMapRenderer`) or fewer than two entries omits `"motion"` entirely, never a fabricated number.
To judge "calmer," compare `bearing_accel.p95_abs` on the SAME plan; `max_abs` alone is too easily
dominated by one switchback.

<a id="route-story-still-grade"></a>
