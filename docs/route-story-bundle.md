# Offline map asset bundle

`gomoveshift.pipeline.route_story.bundle` builds the local, offline "map
asset bundle" a headless MapLibre GL renderer reads. This module is the pure
core: it never makes a network call. Fetching imagery/terrain arrays into
memory is the job of `route_story.sources`; vendoring `maplibre-gl.js` is the
job of the scene package (`route_story/scene/vendor.py`).

## Layout

```
<bundle>/manifest.json
<bundle>/imagery/{z}/{x}/{y}.jpg        256px XYZ tiles, Web Mercator, y down from north
<bundle>/terrain/{z}/{x}/{y}.png        256px Terrain-RGB (Mapbox encoding)
<bundle>/labels.geojson                 FeatureCollection of Point features {name, kind, source}
<bundle>/roads.geojson                  FeatureCollection of OSM highway LineStrings {highway, osm_id, name?} (schema /2)
<bundle>/route.geojson                  FeatureCollection of LineString segments
<bundle>/vendor/maplibre-gl-6.10.0/     maplibre-gl.js, maplibre-gl.css (written by the scene package)
```

Terrain-RGB decodes as `height_m = -10000 + (R*65536 + G*256 + B) * 0.1`,
giving an encodable range of roughly -10,000 m to +1,667,721.5 m in 0.1 m
steps.

## Building a bundle

- `Footprint(west, south, east, north)` — a validated bounding box.
  `footprint_for_route(points, margin_m)` expands a route's bbox by a metric
  margin, correcting longitude for `cos(latitude)` so the margin is a real
  distance at any latitude.
- `lonlat_to_tile`, `tile_bounds`, `tiles_covering` — standard Web Mercator
  XYZ tile math.
- `encode_terrain_rgb` / `decode_terrain_rgb` — the Mapbox Terrain-RGB codec.
  Encoding raises on `NaN` (never invents elevation) and on heights outside
  the encodable range.
- `write_tile_pyramid(root, kind, source_array, source_footprint, zooms,
  resample=...)` samples a georeferenced equirectangular source array (row 0
  = north) into every covering 256px tile at each zoom, writing JPEG quality
  92 for `kind="imagery"` and Terrain-RGB PNG for `kind="terrain"`. Terrain
  heights are always bilinearly sampled before encoding; `resample` (
  `"bilinear"` or `"lanczos"`) only affects imagery. A tile only partially
  covered by the source raster is **skipped**, not filled with invented
  data — the returned `TilePyramidReport` lists both `written` and
  `skipped_partial` tiles so a caller can tell the two apart. Output is
  deterministic: sorted iteration order, fixed encoder parameters, no
  embedded timestamps.
- `write_manifest(root, manifest_fields)` writes `manifest.json` (schema
  `"route-story-bundle/2"`) with a sha256 of every other file under `root`,
  sorted keys, UTF-8, trailing newline — identical inputs give byte-identical
  manifests. `verify_bundle(root)` returns a list of problems: a missing
  manifest, a missing/checksummed-wrong file, an extra file not in the
  manifest, or a schema mismatch. Schemas `/1` (legacy, no road layer) and `/2`
  are both accepted (`MANIFEST_SCHEMAS`); a `/2` manifest must also carry a
  `roads` block and a `roads.geojson` file. An empty list means the bundle is intact.
- `bundle_dir_for(footprint, source_versions)` returns the content-addressed
  cache directory (first 16 hex chars of a sha256 over the footprint rounded
  to 1e-6 and the sorted source versions) under
  `gomoveshift.paths.gms_route_story_cache_dir()` (env
  `GMS_ROUTE_STORY_CACHE_DIR`, default `route-story` under the cache home).
  Same footprint and source versions always resolve to the same directory,
  regardless of dict key order.
