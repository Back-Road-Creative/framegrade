## Route Story bundle preparation (`gomoveshift/pipeline/route_story/prepare.py`)

`prepare_bundle` orchestrates the sibling modules (DEM, imagery, places, tiles, vendored assets) into one content-addressed bundle: DEM (3DEP, with
a Copernicus GLO-30 fallback when 3DEP coverage is short), NAIP orthoimagery at ~2 m/px, one
Overpass query for named places, the tile pyramid, the vendored MapLibre build, then
`verify_bundle`. Network access arrives as injected callables — the module imports no HTTP client,
so its tests are fakes end to end, and the CLI adapter (`gomoveshift route-story-prepare`) is the
only place a real `requests.Session` is wired.

**`recorded_on` selects one calendar day; a raw GPX file never reaches the tile writer.**
`prepare_bundle` requires `recorded_on: date` and runs `select_gpx_day` (`gpx.py`), the same
day-and-duplicate filter the 2D film already used, before computing the footprint or drawing the
route. A real-world GPX export commonly carries other-date test points and byte-identical
duplicate tracks (a device recording the same ride twice); without this, a single stray point on
another date can widen the footprint far past the trip and the route line gets drawn on top of
itself. `route.geojson`'s top-level `properties` and `PrepareReport` both carry
`excluded_other_date_count` / `excluded_duplicate_count` so the receipt says what was left out;
`verify_bundle` rejects a `route.geojson` holding two byte-identical feature geometries.

**Terrain's tile grid defines the footprint both layers share.** `write_tile_pyramid` skips a tile
the source raster only partially covers, and a coarser zoom's larger tiles get truncated more at
the raster's edge than a finer zoom's — so imagery (finer, z13-15) and terrain (coarser, z9-13),
fetched over the same bbox, used to end up with different real-world extents, leaving an unshaded
ring of imagery with no terrain mesh under it. `prepare_bundle` now writes terrain first, then
passes `write_tile_pyramid`'s `extent` parameter (the terrain tile grid's own written bounds at its
finest written zoom) when writing imagery, so imagery is only ever written inside the area terrain
actually covers. The manifest records each kind's real `extent` and `footprint_km`, and
`verify_bundle` rejects a bundle whose imagery extent reaches past its terrain extent.

**Terrain exaggeration defaults to 1.0** (real relief, no invented steepness); an editor may raise
it up to 1.25 (`--exaggeration`, clamped in `prepare_bundle` itself so any caller gets the same
ceiling, not just the CLI).

**Terrain is two independently-fetched DEM bands, mirroring the imagery bands (`bands.plan_imagery_bands`, see `docs/pipeline/14-route-story-local-satellite-raster.md`).**
`bands.plan_terrain_bands` plans a `terrain_regional` band (z9-12) over the *same* snapped bbox as
imagery's `naip_coarse` band, and a `terrain_focal` band (z13) over the *same* snapped bbox as
imagery's `naip` band. An unsnapped focal DEM over a route narrower than one z13 tile fully covered
no z13 tile, wrote nothing, and failed verification with the focal imagery reaching past the terrain
extent. Before `terrain_regional` existed, terrain used one
DEM fetched over the small focal bbox for every zoom 9-13; `write_tile_pyramid` only writes a tile
its source raster *fully* covers, so a focal-sized raster could never fully cover a regional-sized
z9-12 tile and those zooms silently wrote little or nothing — the terrain analogue of the imagery
bug fixed by `plan_imagery_bands`. `terrain_regional` fetches at ~30 m/px (the z12 Web-Mercator
resolution, close to Copernicus GLO-30's own native resolution so a fallback there loses little
detail); `terrain_focal` keeps 3DEP's native ~10 m/px. 3DEP requests are split into tiles of at
most `DEM_3DEP_MAX_TILE_PX` (2048) a side: the service advertises 8000 but answers HTTP 500 for a
~2700x3100 F32 export, and an HTTP error is not a coverage error, so it never reaches the Copernicus
fallback. Every ArcGIS export tile (3DEP and NAIP) is square in **degrees**, not metres:
`exportImage` does not stretch data to the requested `size`, it grows the returned extent until it
matches the pixel aspect (for example, lat 35.000..35.108 asked at 1200x1200 can come back as
34.988..35.120), and with `imageSR=4326` the raster is plate carree. A metre-square grid at 35°N was
returned ~22% taller than the box it was then labelled with, which put the focal DEM tens of metres
off the GPX track's own elevations and the regional DEM kilometres out. `_grid_plan`
drives both axes from the finer step, so neither is coarser than asked, and
`ARCGIS_EXPORT_GRID_VERSION` is part of the bundle cache key so a bundle fetched on the old grid is
never a cache hit. Each band writes with
`write_tile_pyramid(kind="terrain", zooms=<band zooms> ∩ terrain_zooms)`; the two write reports are
unioned for the manifest's declared `terrain` zoom range and `PrepareReport`'s tile counts.
`manifest["terrain"]["extent"]` still comes from the **focal** band alone, exactly as before the
regional band existed, so `verify_bundle`'s imagery-inside-terrain check is unaffected.
`manifest["terrain"]["sources"]` lists both bands' provenances, de-duplicated by name (a 3DEP-only
run has one entry); `manifest["terrain"]["bands"]` carries each band's plan receipt plus its written
tile counts and provenance. The regional imagery band (`naip_coarse`) is written with its `extent`
clipped to the regional terrain band's own written-tile footprint rather than its raw bbox — the
per-band form of that check — so z9-12 imagery never extends past the DEM mesh actually under it. The two
terrain fetches can fall back to Copernicus independently (3DEP might cover the small focal box but
not the much wider regional one): `_band_source_versions` keys them as separate cache-key fields
(`"terrain"` for focal, kept for cache-directory compatibility with every bundle written before the
regional band existed; `"terrain_regional"` new), and `_find_cached_bundle` cross-products both
fallback outcomes together with the gibs daily/Blue-Marble choice (only probed when
`trip_date_imagery=True` — the default resolves the gibs band to Blue Marble alone, so only that one
outcome is checked).

**Labels are a GNIS+OSM merge, fetched independently and never fatal to the bundle.**
`prepare_bundle`'s `label_sources` (default `("gnis", "osm")`) names which sources to try; each is
fetched on its own — a raised `GnisFetchError`/`SourceFetchError`/`OSError`, or a source that
answers with zero features, is recorded as a one-line problem (`"gnis: …"` / `"osm: …"`) rather than
aborting the fetch, so a dead Overpass mirror still ships a GNIS-only bundle and vice versa. The two
feature lists are deduped by `(normalised name, kind)`; GNIS wins a collision as the more precise,
curated source. `PrepareReport.labels_degraded` is True whenever any configured source failed/came
back empty OR the merged set is empty; `label_problems` carries the one-line reasons. The manifest's
`labels` block mirrors this: `sources` (one provenance per successful source), `degraded`,
`problems`, `count`. `gnis_cache_dir` (default `gms_route_story_cache_dir() / "gnis"`) is threaded
to `gnis.gnis_features_for_bbox` so its per-state zip cache lives beside the bundle cache rather than
a hidden default. `_report_from_existing` reads these same manifest fields, so a cache hit reports
identical degradation flags to the run that built it.

**A degraded bundle repairs its labels on the next cache hit** (`prepare._repair_degraded_labels`).
The label sources are not part of a bundle's cache address, and `verify_bundle` calls a GNIS-only
bundle perfectly clean — so a prepare that lost Overpass (read timeouts or an unreachable network across every mirror) wrote a degraded bundle that then
answered *every* later request for that footprint, and every film rendered without OSM labels until
someone deleted the directory by hand. Refusing the cache hit outright would re-fetch thousands of
terrain and imagery tiles to repair one small JSON file, so the retry is scoped to what failed: on a
hit whose manifest records `labels.degraded`, the label fetch alone runs again, over the bbox the
manifest itself recorded (re-deriving it here could label a different area than the bundle covers).
The bundle is upgraded in place only when the retry is strictly better — undegraded AND no smaller
than what is on disk: an undegraded fetch only means every source answered, not that it answered as
fully as last time, and silently trading a complete label set for a thin one is worse than staying
degraded. `labels.geojson` and the manifest's `labels` block are rewritten together, with only that
one entry in the manifest's `files` map re-hashed (`bundle.rewrite_manifest` /
`bundle.bundle_file_digest` — `write_manifest` would re-hash every tile in the bundle to record it),
so the repaired bundle still verifies and still answers the same cache probe. An undegraded bundle
performs no network I/O at all on a hit; `find_prepared_bundle` is unchanged and still never fetches.

**Labels are fetched through the service's own
Overpass mirror list** (`OVERPASS_ENDPOINTS`, tried in order, first answer wins, every failure named
if all are down): `fetch_osm_labels` shipped with the canonical host hard-coded, which is precisely
the host that can refuse IPv4 connections on 443 (see the network allowlist), so bundle preparation failed with `Errno 101` wherever that host is unreachable, while another mirror
answered the identical query. `prepare_bundle`'s `overpass_endpoint` defaults to `None` (try the
mirror list); an explicit value still bypasses it, for a caller that wants one specific mirror.
**Overpass's own server-side budget is `sources.OSM_SERVER_BUDGET_S` = 60s** (the `[timeout:60]`
clause in the query itself), and the client-side HTTP timeout (`OSM_QUERY_TIMEOUT_S` = 90s) is kept
longer than that budget so a slow-but-alive mirror reports its own giveup in the Overpass response
body instead of `requests` severing the connection first and mislabeling a real answer as a client
timeout. The Overpass response itself is not cross-run cached from this module — `chapters/
overpass_source.py`'s shared cache is keyed to that module's own query shape and is not reused here.

**Labels are fetched over the regional `naip_coarse` band bbox, not the route footprint.** A film
descends from national scale through regional zooms before reaching the focal route, so a town, named
summit, or gap just outside the small footprint margin still needs to become a label. Fetching the
much wider `naip_coarse_band.bbox` instead risks flooding the regional ring with minor features
(streams, trails, etc.) that happen to fall inside it, so `_filter_context_labels` runs on the merged
GNIS+OSM set, after the merge and before the count/labels.geojson are written: a feature inside the
route footprint bbox is kept whatever its kind, one outside it is kept only when its `kind` is one of
the ladder's coarsest, most orienting entries (`place`, `summit`, `gap`). The manifest's `labels`
block also records `bbox` (the `naip_coarse` band bbox actually searched), so the receipt states the
area labels came from.

**Roads are an offline vector layer, fetched over the route bbox.** `prepare_bundle` makes one
extra Overpass query per prepare, `fetch_osm_roads`, for the `highway` ways inside the route
footprint (the route plus its focal margin, not the wider regional labels bbox), and writes them
as `roads.geojson`: a FeatureCollection of `LineString` features
`{highway, osm_id, name?}` sorted by `osm_id`, so a later map inset can draw nearby roads. Only the
classes in `sources.OSM_ROAD_CLASSES` are kept (motorway through residential, plus the `_link`
ramps OSM defines); service roads, tracks, paths and footways are not. The fetch shares
`sources._overpass_json` (mirror fallback, user agent, timeout) with `fetch_osm_labels`. The
manifest's `roads` block holds `bbox`, `count`, `degraded`, `problems` and a `provenance` string
naming OpenStreetMap and ODbL 1.0. Films do not need roads, so a failed or empty fetch never fails
the prepare: it writes an empty FeatureCollection and records `degraded: true` with the reason
(`"roads: no highway ways in bbox"` for an empty answer, never silently fine). Manifest schema is
now `route-story-bundle/2`; `verify_bundle` still accepts `/1`, and flags a `/2` bundle with no
`roads.geojson` or no `roads` block. `ROADS_SOURCE_VERSION` is part of the cache address, so a
pre-roads bundle is never a cache hit for the new layout and is re-prepared in full. A cache hit
whose manifest records `roads.degraded` re-runs the road fetch alone over the recorded bbox
(`prepare._repair_degraded_roads`) and rewrites `roads.geojson` and the manifest entry only when
the retry is undegraded.

The zoom ranges are chosen against each source's native resolution rather than picked: 3DEP is
~10 m/px so z13 (~15.6 m/px at this latitude) is the finest terrain zoom that fabricates no
detail; NAIP at 2 m/px matches z15. **What the manifest declares comes from the write report, not
from the requested list** — `write_tile_pyramid` skips any tile the source raster only partly
covers, so at a small footprint the wide zooms have no fully covered tile at all, and a manifest
that advertised them made the scene request tiles that 404 (see *bundle coverage truth*). A kind
whose requested zooms wrote nothing at all raises instead of shipping an empty pyramid.

Re-running for the same footprint and source versions returns the same directory without touching
the network, and only when that bundle still verifies clean — a bundle that fails verification is
re-fetched rather than reused. `bundle_dir_for`'s cache key also folds in `terrain_exaggeration`
(default 1.0, matching a caller that omits it), so two bundles differing only in the exaggeration
baked into their terrain tiles get distinct cache directories instead of colliding.

**A bundle's address carries its shape, not just its sources.** `bundle_dir_for` folds in the
requested `imagery_zooms` and `terrain_zooms` (canonicalised, so order and duplicates don't
matter), and `_band_source_versions` folds in `bands.BAND_PLAN_VERSION`. Each closes a real hole:

- A bundle built for z13-15 answered a request for z2-15. `_find_cached_bundle` skips the entire
  fetch on any existing directory that verifies clean, and `_report_from_existing` states the
  zooms that were *asked for* rather than the ones on disk — so the receipt claimed a wide bundle
  and the scene found nothing at the wide end. With the shape in the address a differently-shaped
  bundle is a different directory, so the report can only ever describe the bundle it came from.
- Changing which source covers which zoom, or how wide a band reaches, moved none of the key's
  other fields — so a ladder fix could not reach a trip already prepared, and the only
  remedy was to know the hash and delete the directory by hand. `BAND_PLAN_VERSION` (`bands.py`)
  is bumped whenever a change there would make an already-cached bundle's tiles wrong or stale for
  the same request; `"bands-2"` is the five-band ladder sized off the opening mosaic cap.

Both default to the pre-shape key when omitted, so bundles already on disk keep their addresses
and are superseded rather than orphaned.
