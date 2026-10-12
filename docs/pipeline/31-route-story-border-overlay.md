## Route Story border overlay (`gomoveshift/pipeline/route_story/borders.py`)

A satellite backdrop with no coastline/state/national lines is hard to recognise at the
z2-4 national scale — viewer feedback on the opening shot was that it "doesn't look like
the USA". `sources.fetch_gibs_reference_features` fetches NASA GIBS's `Reference_Features_15m`
WMS layer as a transparent PNG (`FORMAT=image/png`, `TRANSPARENT=TRUE`) from the same
`gibs.earthdata.nasa.gov` host `_fetch_gibs` already uses — no new allowlisted host. Its
`GeoRaster.array` is `uint8 HxWx4`; the overlay's own RGB is not meaningful (unused), only
its alpha channel marks the reference lines. Provenance `licence` is written as
`"unverified"` rather than invented: the line data underlying this NASA layer may
be third-party, and no licence claim should ship without confirming it.

`borders.overlay_borders(rgb, overlay_rgba, *, colour, opacity, halo)` is a pure numpy/PIL
compositor (no I/O) that blends the overlay's alpha as coloured lines onto a backdrop,
resizing the overlay first if its pixel size doesn't match. `opacity=0` or an all-zero
alpha overlay is a byte-identical no-op. `halo=True` (the default) first darkens a
1-px-dilated copy of the line mask (`ImageFilter.MaxFilter(3)`, `HALO_DARKEN_FRACTION`
toward black) under and around each line, so a light line recipe colour stays legible over
both bright desert/snow and dark ocean. `border_coverage(overlay_rgba)` returns the
fraction of pixels above `COVERAGE_ALPHA_THRESHOLD`, letting a caller detect a near-empty
or failed overlay fetch. `BORDERS_RECIPE_VERSION` versions the blend recipe for a later
cache key.

**Wired into `prepare.py`.** Every Blue Marble raster gets borders drawn
after grading, never before — `overlay_borders` must blend onto the graded pixels, or the line
colour itself would get re-graded. `_ensure_blue_marble_tiles` (shared z2-4 band) and
`_gibs_band_raster` (per-bundle z5-8 band, both the default path and the `trip_date_imagery=True`
fallback to Blue Marble) each fetch `fetch_gibs_reference_features` for the exact bbox and request
size their own raster was fetched at, then call `overlay_borders(grade_backdrop(raster.array),
overlay.array)`. A chosen trip-date daily scene (real satellite imagery, already showing real
coastline) never gets an overlay and makes no reference-features request.

Robust-by-construction split on what a near-zero `border_coverage` means, since the two bands'
bboxes differ in kind: the shared z2-4 band's bbox is the whole globe, which always has real
coastline, so `_ensure_blue_marble_tiles` treats coverage below `prepare.MIN_SHARED_BORDER_COVERAGE`
(0.002 — well under the 0.0277 a real CONUS bbox measured) **or a fetch error** as a hard failure,
raising `SourceFetchError` *before* the shared dir is created or any tile written, so a bad fetch can
never poison the shared cache with an ungraded/borderless tile silently mistaken for correct. The
per-bundle z5-8 `gibs` band's bbox can be a small inland footprint that legitimately has no
coastline/state line in it at all, so zero coverage there is recorded, never a failure.

`BLUE_MARBLE_CACHE_VERSION` is bumped to `"blue-marble-3"` (the shared tiles' pixels changed again)
and `_band_source_versions` folds in `"borders_recipe": BORDERS_RECIPE_VERSION`, so a bundle
prepared before borders existed is never mistaken for one prepared after — same reasoning as the
`backdrop_recipe` bump above.

`gibs_decision`'s manifest entry (`imagery.bands[].cloud_decision`) gains a `"borders"` key: `{"layer":
"Reference_Features_15m", "coverage": <float>}` when an overlay was applied, `None` for a chosen
daily scene.

Both the shared z2-4 band's and the gibs band's own borders-overlay provenance are appended to
`manifest["imagery"]["sources"]`, deduplicated by name (`_dedup_provenances`, shared with the
terrain sources list) — in the default (cloud-free) mode both bands draw from the identical GIBS
layer, so listing both unconditionally would name the same source twice. `_ensure_blue_marble_tiles`
persists the shared band's own overlay provenance to a second file beside the base one
(`shared/blue-marble/<version>/borders-provenance.json`, next to `provenance.json`), so a bundle
resolved from an already-warm shared cache (no new HTTP calls at all) still names it — previously
this file didn't exist and the shared band's borders source was dropped from every such bundle's
manifest entirely. A cache hit with that file missing (but the base
`provenance.json` and every needed tile present) is treated the same as a missing `provenance.json`
always was: refetch, never silently fall back to an undated placeholder.

`route-story-prepare --trip-date-imagery` (default off) is the CLI surface for
`trip_date_imagery`; the default renders the cloud-free Blue Marble backdrop, the flag opts into
the trip-date satellite scene with its existing cloud-fraction fallback. `route-story-proof` takes
the identical `--trip-date-imagery` flag and threads it into its `find_prepared_bundle` lookup
(`cli/handlers/route_story_proof.py`) — it must match whatever `route-story-prepare` call wrote the
bundle being resolved, since the flag is folded into the bundle's cache key (`_band_source_versions`'s
`imagery_gibs` version differs by which scene was actually chosen).
