## Route Story local satellite raster (`gomoveshift/pipeline/route_story/satellite.py`)

`SatelliteRaster` validates a local readable image, non-empty source credit, and geographic bounds.
It crops and resizes the source from a saved opening camera pose. `render_map_frame` and
`render_video` accept the same optional opening/raster pair, so a preview cannot use one imagery
path while the MP4 uses another. `receipt` records a content hash, dimensions, bounds, and source
credit for reproducibility without leaking the local source path. This module has no URL, tile
request, or network fallback.
`prepare` uses the source checksum and extension as its local cache key. A matching prepared
derivative is reused rather than copied again; the original source remains untouched.

The film's opening credits the raster each frame is drawn from, not always the first one.
`layers.opening_credit` runs the same selection the frame does (`_viewport_raster`, shared with
`_render_opening_rasters`), so the descent from the national overview through the regional levels
to the local imagery changes its on-screen credit at the frame the imagery changes. Printing the national credit over another agency's regional and local imagery for the whole descent is the defect this selection prevents.

**The opening camera frames the country before it travels to the trailhead** (`layers._opening_camera`).
An opening locked on the trailhead's own coordinates for the
WHOLE move (`target.latitude, target.longitude` at every fraction) with a fixed 90-degree starting
span puts mostly ocean/off-country content in frame for a trailhead that is off-centre in its raster
set, and does not read as the intended country. `layers._country_framing` instead derives the
opening's starting centre and span from the rasters actually supplied: the widest raster whose OWN
bounding box, at its own resolution, would still resolve sharply if drawn full-frame
(`_OPENING_COUNTRY_FRAMING_MIN_RATIO`), so the country framing is never a hardcoded continental box.
The camera's centre holds
that framing exactly (no drift) until the span itself starts easing in
(`_OPENING_SPAN_EASE_START`), then eases to the target by `_OPENING_CENTER_LOCK_FRACTION` and holds
there, locked, for the rest of the move — so the last third (the steepest part of the zoom) has no
lateral drift at all, only scale. The final span (`_OPENING_TARGET_SPAN_LON`) is unchanged from
before this camera learned to travel.

**A whole-world raster is excluded from the framing candidates by its SPAN, not by its name**
(`_OPENING_MAX_COUNTRY_SPAN_LON`, 120°). The sharpness ratio above was, on its own, the whole of the
selection, and the assumption underneath it — that a whole-world band is too coarse to pass — is
simply false: the ladder's coarsest band reaches z4, whose mosaic is ~4096 px wide, so it clears the
ratio with room to spare and then wins "widest" with a ~336 degree span. Such a raster produces an opening on a
whole-world map with Africa, Europe and Australia in frame that crosses the Atlantic and reaches the
United States three-quarters of the way down. The sharpness gate below refuses such an opening only on a technicality (a ratio right at the floor) rather than on the
real defect, which is that no pixel count can make a 336 degree span read as one country.

Excluding the backdrop band by name (`satellite.BACKDROP_BAND`, carried onto each `SatelliteRaster`
as its `band`) is not enough. A name only excludes sources that
label themselves, and the render command also passes `--national-raster`, which `film.py`
builds with hardcoded `-180, -90, 180, 90` bounds and **no band at all** — so `band != BACKDROP_BAND`
is true of it, it clears the sharpness floor on raw pixel count, and it wins anyway. With such a raster both the film and its own review pack frame
`(0.0, 0.0, 360)`. How much of the planet a raster covers is visible whatever it came from, so the
span test closes the hole for every imagery source added later; the band exclusion is kept beside it
as a statement of intent. The 120° cap only ever REJECTS — the framing is still whichever surviving
raster is widest, never a hardcoded box — and the contiguous United States (~59°) clears it easily.
When nothing clears the cap, the fallback is the **narrowest** raster available: falling back to the
widest there would reinstate exactly the framing just refused.

**The review pack's opening sheet renders from the film's own raster set**, never a subset. It was
fed only the flat `(national, *regional, local)` rasters while the film rendered its opening from
those plus the bundle's band pyramid — so the one gate built to catch a bad establishing shot could
show an opening the film would not produce.

**A raster's width is read from its file header, never from decoded pixels** (`_raster_width`,
unbounded cache). The country framing, the sharpness check and `finest_last` each ask every
candidate raster how wide it is, and the framing is recomputed on every opening frame. They used to
ask `_raster_image(source).width` -- a full decode behind an 8-image cache -- while a bundle film
passes 18 sources (four from the command line plus one per pyramid level, the largest a multi-megabyte JPEG).
The cache thrashed, so each of the opening's 360 frames re-decoded about ten mosaics to read ten
integers; reading the header instead gives identical results at a negligible cost. `_raster_image` and its graded twin stay bounded -- they hold pixels, and
the opening only ever draws from the one raster covering the current frame.

**The opening is graded like every other still** (`layers._graded_raster_image`, `grade.grade_still`
— haze removal, exposure, vibrance, local punch). Each source raster is graded once and cached; a
per-frame grade of a full mosaic is too slow to repeat across an opening's whole frame sequence, and
every frame drawn from the same raster looks identical anyway. `_render_opening_rasters` is a
grading-scoped copy of `_render_geographic_rasters`'s projection so the following-map renderer every
other leg uses is untouched.

**`layers.check_opening_sharpness` raises before the long render** if the camera's own selected
raster, at any sampled fraction of the move, would have to be upscaled past
`_OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX` source pixels per output pixel to fill the frame — a silent
per-frame upscale is what read as "dark and fuzzy when zooming in". A first cut of this gate
still raised late in the descent: with only the finest complete
zoom emitted per band and the focal (`naip`) band's bbox sized for the trail alone (~0.13° wide),
there was nothing between `naip_coarse`'s finest zoom (z12, 2913 px/deg) and `naip`'s own too-narrow
bbox, so the camera sat on an upscaled `naip_coarse` mosaic for the last third of the descent (a very low ratio) — the exact defect the gate exists to catch. Widening that band left a second, narrower below-floor window (0.05 of the move wide) that a
coarse 8-point fraction sample stepped clean over; the default is now 51 points, a 0.02
step (`layers._OPENING_SHARPNESS_CHECK_FRACTIONS`) — one cached image open per raster per sample, so
the density is nearly free. Changes in `bands.py`/`satellite.py` close the geometry gap itself; see
below. `film.main` only builds and checks the real opening frames (passed to
`preflight.run_preflight`'s `opening_frames`) when a bundle pyramid is supplied — the flat
`--national-raster`/`--local-raster` fallback path predates this gate and was never tuned against
it.

Given a scene bundle (`--bundle`), the opening also descends through the bundle's own tile pyramid.
`satellite.bundle_opening_rasters` reads `imagery.bands` from the bundle manifest and mosaics
**every** zoom that is complete on disk and no wider than `OPENING_MOSAIC_MAX_WIDTH_PX` in each
band — not just the finest — coarse to fine, leaving out (with a warning) only a band with no
complete zoom at all. Every zoom in a band shares that band's bbox by construction
(`bands.plan_imagery_bands`), so this lets the camera fall back to a coarser zoom of the SAME band
before it ever drops to the next, much coarser band; `layers._viewport_raster` already picks the
sharpest covering raster, so nothing downstream needed to change to use this. Tiles are Web Mercator
and the opening camera reads a raster as evenly spaced in latitude, so `_even_latitude` respaces the
rows; without that a 27-degree band puts places hundreds of kilometres from their labels. Each mosaic
carries its band's `provenance.attribution` as its
credit, is written to a scratch directory the film CLI removes, and is listed in the receipt as
`opening_pyramid` (zoom, credit, bounds). `layers.finest_last` then orders the supplied rasters
and the pyramid by degrees per pixel, so the frame shows the sharpest raster that fully covers it.
Without the pyramid the opening only has the supplied rasters: when the trailhead sits
very close to the local raster's south edge (for example 0.002 degrees), the final viewport never fits inside it, and
the end of the descent is a coarse regional raster enlarged many times over.
The pyramid's fine band is cut around the route with a margin, so it covers the arrival; if a
future bundle's does not, the opening falls back to the next coarser raster, never to a partial
paste. The credit is outlined because trip-day imagery can put white cloud under it.

**A raster hand-off cross-fades instead of cutting on one frame, and a finer raster is tone-matched
to its coarser neighbour** (`layers._render_opening_rasters`, `_tone_matched_opening_image`,
`_opening_raster_crop`, `_OPENING_RASTER_CROSSFADE_BAND`). Without it the map flips in one frame from vivid green NASA GIBS to grey-blue USDA NAIP. A
`naip_coarse` mosaic (`satellite.bundle_opening_rasters`) has a
second, deeper limitation: it is itself a visible patchwork of separately toned NAIP source tiles, baked
in at bundle time (`satellite._mosaic`/`_even_latitude` stitching different acquisitions with no
cross-tile matching). Matching tiles there is not done, since it is not a per-frame
rendering defect. The cross-fade addresses the band-to-band hand-off: `_viewport_raster` already
picks the sharpest fully-covering raster (unchanged); when a second, coarser raster also still fully
covers the viewport, the two are blended by `Image.blend` over
`_OPENING_RASTER_CROSSFADE_BAND` of resolving-power headroom (the same source-px-per-output-px units
`check_opening_sharpness` already gates on) instead of switching in one frame. Each raster is also
colour-matched to its nearest coarser neighbour once, cached like the grade it wraps
(`_tone_matched_opening_image`): the finer raster's mean colour, measured over the geographic patch
it shares with that neighbour, is shifted onto the neighbour's own, so the hand-off reads as one map
continuing rather than a different layer starting. Only the mean moves: matching the standard
deviation too scaled the finer raster's contrast down to its blurrier neighbour's, and the preflight sharpness gate refused the result.

**`bands.plan_imagery_bands` is a five-band ladder, each finer band's bbox sized off the opening
mosaic cap.** Widening only the finest (`naip`) band closed the
`naip_coarse`→`naip` handoff but left a second, coarser window open: `naip_coarse`'s own bbox (sized
for `regional_margin_km` alone) was too narrow to cover the span where `gibs`'s z8 first drops below
the sharpness floor, so the camera fell back from `gibs` straight to a too-narrow `naip_coarse`/`naip`
pair with nothing between. A walk of the whole move at a 0.01 step finds both windows: one at the `gibs`→
`naip_coarse` hand-off and one at the `naip_coarse`→`naip` gap.

The fix sizes `naip_coarse` (z9-11), a new `naip_mid` band (z12-14), and `naip` (z15) each to
`OPENING_MOSAIC_MAX_WIDTH_PX / px_per_deg(band's own max zoom)` — the widest bbox that band's own
finest zoom can still mosaic under `satellite.bundle_opening_rasters`' `_complete_level` tile-column
cap (`bands._cap_bbox_width_deg`). That cap is a **two-sided fit**, not a floor
(`bands._fit_snapped_bbox_to_cap`, applied after the existing whole-tile snap): it grows a narrow
band up to the cap AND trims a wide one back down to it, moving whole tiles at the band's own
minimum zoom so the finest zoom's tile-column count lands exactly at the cap rather than a tile
over it. Treating it as a floor alone — honouring `regional_margin_km`/`focal_margin_m` past the
cap — is a bug, not a courtesy: a band one column too wide has that whole zoom
level dropped by `_complete_level`, so the wider margin buys no imagery and costs the entire level.
Otherwise a band can come out 9 columns wide where 8 fit, a 9216px mosaic against the 8192px
ceiling. The one thing the trim never cuts into is
the raw route bbox: a route too wide for a band's finest zoom keeps its own columns and accepts the
drop, because no bbox can serve it there and the next coarser band is what covers that span (a 0.4°
route loses z15 and lands at ratio 0.516, still over the floor; a route narrower than the 0.352° `naip` cap keeps z15). Because
`px_per_deg` doubles every zoom, each band's cap width is ~4x its own zoom span, always at least the
span the next-coarser band's own floor requires — closing both windows by construction, not by
picking numbers against one input:

| band | zooms | bbox width (~35°N route) | px/deg at maxzoom |
|---|---|---|---|
| `blue_marble` | 2-4 | 180° (global, unchanged) | 11.4 |
| `gibs` | 5-8 | ~33.75° (national margin, unchanged) | 182.0 |
| `naip_coarse` | 9-11 | cap: 5.625° | 1456.4 |
| `naip_mid` (new) | 12-14 | cap: 0.703° | 11650.8 |
| `naip` | 15 | cap: 0.352° | 23301.7 |

A cap-widened band is also widened north-south to the delivery aspect (`film.DELIVERY_HEIGHT` /
`DELIVERY_WIDTH`, duplicated as `_ASSUMED_OPENING_HEIGHT_PX`) — a bbox wide enough east-west but left
at its margin-only height cannot fully cover a full-width viewport on the north/south edges, and
`_viewport_raster` requires whole-rectangle coverage. `plan_terrain_bands`'s `terrain_focal` (z13,
now inside `naip_mid`'s range rather than the now z15-only `naip`'s) shares `naip_mid`'s bbox exactly
via the shared `bands._naip_mid_bbox` helper; `terrain_regional` shares `naip_coarse`'s via
`bands._naip_coarse_bbox` the same way. `prepare_bundle` fetches and writes `naip_mid` exactly like
`naip_coarse`/`naip` (its own `fetch_naip_imagery` call, `_write_band`, manifest entry, provenance),
bound by the same `terrain_focal` written-tile extent `naip` always was, since neither has a
terrain zoom finer than z13 of its own. Proven by geometry (`tests/test_route_story_bands.py`'s
ladder-property and whole-move-walk tests, `tests/test_route_story_opening_camera.py`'s
realistic-band-layout walk) rather than measurement.

`frame_qa.FrameQaThresholds.min_opening_sharpness` is deliberately not recalibrated against
the ladder: it is held at its provisional value of 50.0, rather than lowered to whatever a still-
broken opening needs to pass — see the field's own comment for the reasoning behind
that floor. `check_opening_sharpness` (proven by geometry above) is the actual, resolution-based
upstream gate for this defect; `min_opening_sharpness` is a late, pixel-content gate kept as a
backstop, not the proof.

**Opening labels are planned once for the whole move, not re-ranked per frame**
(`layers._opening_label_windows`, `_opening_visible_labels`, `_draw_opening_map_labels`). A per-frame path (`select_visible_labels`) recomputes the visible set
statelessly every frame: a hard in/out on each label's visibility span, ranked by distance to the
moving frame centre, truncated to `max_labels`, so ranking or a span edge shifting by one sample
changes the whole on-screen set and pops labels in and out in single frames. Because `_opening_camera`'s pose is a pure function of `fraction`,
the camera's entire path is already known before a single frame renders, so
`_opening_label_windows` walks it once at a fixed 241-point grid and decides, in order, exactly which
fraction-space window each label occupies: a label already on screen is never evicted by a
higher-ranked newcomer (the newcomer waits for a free slot), and a label whose qualifying run is
shorter than `_OPENING_LABEL_DWELL_FRACTION` (~1.5s of a nominal 12s intro) is dropped entirely
rather than flashed. `_opening_visible_labels` reads that plan for one frame's exact `fraction` and
fades each label in/out over `_OPENING_LABEL_RAMP_FRACTION` (~12 frames of a nominal 360-frame intro)
at the edges of its own window; `layers.py` never sees the run's real fps/duration (only `fraction`),
so both constants are fixed fractions of the move rather than real seconds/frames.
`_draw_opening_map_labels` is `_draw_map_labels`'s layout (collision avoidance, screen-edge margins)
unchanged, drawing each label at its own fade alpha instead of always opaque; the shared
following-map renderer every other leg uses (`select_visible_labels`/`_draw_map_labels`) is
untouched.

The private console's `GET /route-stories/{id}/opening` requires exactly one stored GPX and one
stored local raster plus a bounds/credit recipe. It returns deterministic endpoint camera samples
and the imagery receipt for editor inspection without spawning a renderer or fetching any data.
Every stored source hash is rechecked before it is consumed, which prevents a changed GPX or raster
from silently altering an already-reviewed draft.
