## Route Story imagery zoom bands (`gomoveshift/pipeline/route_story/bands.py`)

A single raster sized for the trail can never fully cover a wide, coarse-zoom tile
(`_fully_covered` in `bundle.py`), so fetching one NAIP raster over the focal footprint for every
zoom silently writes nothing at the wide zooms and the bundle floor becomes z13 — no zoom-out from
the national scene is possible. `plan_imagery_bands` plans four imagery zoom
bands so a scene can descend all the way from the national view to the trail: z2-z4 from NASA GIBS
Blue Marble (`"blue_marble"`), z5-z8 also from NASA GIBS Blue Marble by default (`"gibs"`;
`trip_date_imagery=True` opts into the trip-date true-colour scene instead, with its Blue Marble
cloud-fraction fallback untouched), z9-z12 from a coarse NAIP pull (`"naip_coarse"`), z13-z15 from
2 m NAIP (`"naip"`, the existing focal footprint). Each `ZoomBand` carries its own bbox and target
resolution. `blue_marble` and `gibs` were originally one z2-8 band snapped at z2; a national-margin
bbox snapped outward to a whole z2 tile is ~90x66 degrees at mid-latitudes, so writing z2-8 over it
was ~10.9k tiles per bundle. Snapping `gibs`'s z5-8 at z5 instead keeps its bbox to roughly the
national margin, and `blue_marble`'s z2-4 is fetched once into a shared cache rather than per
bundle (below).

**Backdrop grading and the cloud-free default (`gomoveshift/pipeline/route_story/backdrop.py`,
`prepare.py`).** Both GIBS Blue Marble layers come back dark and desaturated, and a trip-date true-colour scene over the wide gibs
band bbox often carries cloud. `prepare_bundle` defaults `trip_date_imagery=False`: the `gibs`
band fetches the dateless Blue Marble layer directly — no daily-scene request, no cloud-fraction
call — so it is cloud-free by construction rather than by a threshold. `backdrop.grade_backdrop`
(a fixed, eyeballed-against-real-samples recipe: midtone gamma lift, saturation boost about each
pixel's own channel mean, black/white stretch, light S-curve) is applied to every Blue Marble
raster before tiling — both the shared `blue_marble` band and the per-bundle `gibs` band, and the
`gibs` band's own Blue Marble fallback when `trip_date_imagery=True` lands on a cloudy day. The
daily true-colour scene is never graded. Unlike the shared z2-4 band, the z5-8 `gibs` band's bbox
is footprint-dependent — z5 has 512 tiles globally versus z2's 8, so different trips rarely land in
the same tiles — so its Blue Marble raster is fetched and written per bundle, not through the
shared cache. `BACKDROP_RECIPE_VERSION` and a bumped `BLUE_MARBLE_CACHE_VERSION` are folded into
`_band_source_versions`, so a bundle prepared before grading existed is never mistaken for one
prepared after. `gibs_decision`'s manifest entry (`imagery.bands[].cloud_decision`) always records
which imagery was used, including a `trip_date_imagery` flag alongside `chosen`.

A band's bbox is snapped outward to whole tiles **at the band's own coarsest (minimum) zoom**,
never its finest. A Web-Mercator tile grid is a strict refinement across zooms, so a tile boundary
aligned at a coarse zoom is automatically a valid boundary at every finer zoom nested inside it —
snapping at the finest zoom instead gives no such guarantee, and reproduces the exact bug being
fixed one level up. Because `lonlat_to_tile` and `tile_bounds` are numeric (not bit-exact) inverses
of each other, a bbox built to land exactly on a tile boundary can re-project a few ULPs to the
wrong side of it; `plan_imagery_bands` nudges each edge inward by `1e-10` degrees (far above float
noise, far below `_fully_covered`'s own `eps=1e-9`) to keep `tiles_covering` from reading that noise
as belonging to a neighbouring tile.

`band_raster_size` returns the pixel size to fetch a band at its `target_m_per_px`, or — when that
exceeds the source's request-size ceiling (`NAIP_MAX_REQUEST_PX`, which is `sources.NAIP_MAX_TILE_PX = 2048`:
NAIP advertises 4000 but a 4000 px export outruns the 30 s client timeout;
`GIBS_MAX_REQUEST_PX = 8192`, a conservative WMS default, not a measured NASA limit) — a list of
`RasterRequest` sub-fetches tiling the bbox. `bands_receipt` renders a band plan for the bundle
manifest. `sources.gibs_scene_for` wraps the existing truecolor/Blue-Marble fetch-plus-decide
sequence as the single call a band planner needs.

**Wired into `prepare_bundle`**: each band is fetched and written over its own bbox.
`blue_marble`'s z2-4 tiles are dateless and shared across every bundle rather than fetched per
trip — `prepare.py`'s `_ensure_blue_marble_tiles` computes the tile list a band would need purely
from geometry (every zoom in a band is fully covered by its own bbox by construction, so this is
knowable before any fetch), checks a cache dir under
`gms_route_story_cache_dir() / "shared" / "blue-marble" / <version>` for them, and only calls
`sources.fetch_gibs_blue_marble` when at least one is missing. `_install_blue_marble_tiles` then
hard-links (falling back to a copy on `OSError`, e.g. across filesystems) each needed tile from the
shared cache into the bundle's own `imagery/<z>/<x>/<y>.jpg`, at the same path `write_tile_pyramid`
would have used — the bundle stays self-contained since MapLibre only ever serves the bundle dir.
`imagery_zooms` excluding the whole z13-15 focal band raises before any fetch: without a written
focal band, the extent check below would fall back to the raw, unsnapped focal bbox, which can
reach past the terrain extent.

`plan_terrain_bands` plans the terrain analogue: a `terrain_regional` band (z9-12) built with the
same margin and snap-zoom as `naip_coarse`, so the two share an identical bbox by construction, and
a `terrain_focal` band (z13) built with the same margin and snap-zoom as `naip`, likewise identical
by construction. See
*Route Story bundle preparation* above for how the two are fetched, written, and cached.

<a id="route-story-borders"></a>
