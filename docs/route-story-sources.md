# Route Story map source fetchers

`gomoveshift.pipeline.route_story.sources` fetches free public source data for a
footprint (a `(west, south, east, north)` EPSG:4326 bbox) and returns in-memory
georeferenced arrays plus provenance. It owns fetching only — the bundle module tiles the
results and writes the one-time map asset bundle and its manifest
(`route_story.bundle`).

Every fetcher takes an injected `session` (anything shaped like `requests.Session`:
`.get(url, params=..., timeout=..., headers=...)`) so tests never hit the network.

## Data model

- `SourceProvenance(name, url, licence, attribution, acquired)` — `acquired` is the ISO
  date of the underlying data when known (satellite scenes), else `"unknown"` (surveyed
  terrain/imagery with no single capture date attached by the service).
- `GeoRaster(array, west, south, east, north, provenance)` — `array` row 0 is north,
  columns run west→east, on a linear lon/lat (EPSG:4326) grid. Elevation arrays are
  `float32` metres; imagery arrays are `uint8` RGB.

## Fetchers

- **`fetch_3dep_dem`** — USGS 3DEP `ImageServer/exportImage`, ~10 m/px, `float32` TIFF.
  Vertical datum is **NAVD88**. Requests larger than `max_px` per side are split into a
  grid and stitched. Any nodata pixel (≤ -1e30, or matching the TIFF's `GDAL_NODATA`
  tag) raises `SourceCoverageError` naming the fraction missing — the fetcher never
  fills nodata. Licence: public domain (USGS).
- **`fetch_naip_imagery`** — USGS National Map NAIP `ImageServer/exportImage`, `uint8`
  RGB, same grid/stitch behaviour. An all-white or all-black region covering more than
  2% of the stitched array raises `SourceCoverageError`. Licence: public domain (USDA
  NAIP via USGS).
- **`fetch_gibs_truecolor`** / **`fetch_gibs_blue_marble`** — NASA GIBS WMS 1.3.0
  `GetMap`. WMS 1.3.0 with `CRS=EPSG:4326` orders the bbox `south,west,north,east`
  (lat/lon, not lon/lat) — the fetcher builds that axis order internally. The true-colour
  fetch requires a `date` and sets `TIME`; Blue Marble is dateless and carries no `TIME`
  param. Licence: NASA open data (attribution requested).
- **`cloud_fraction`** / **`choose_opening_scene`** — a deterministic heuristic (near-white,
  low-saturation pixels) picks between a daily GIBS scene and the Blue Marble fallback
  when the daily scene is too cloudy (`max_cloud`, default `0.35`). The decision dict
  (`chosen`, `cloud_fraction`, `threshold`) is receipt material for the bundle manifest.
- **`fetch_osm_labels`** — one Overpass QL query for named `natural=peak|saddle|ridge`,
  `mountain_pass=yes`, `place=city|town|village` and `waterway=river|stream` features,
  returned as a GeoJSON `FeatureCollection` sorted by `(kind, name, osm_id)`. Licence:
  ODbL 1.0, attribution "© OpenStreetMap contributors".
- **`fetch_copernicus_dem`** — worldwide fallback DEM: Copernicus GLO-30 COGs on the
  public AWS bucket, one request per covering 1°×1° cell, cropped and stitched to the
  bbox. A missing cell (HTTP 404) means open ocean and raises `SourceCoverageError`.
  Licence: Copernicus DEM licence (free, attribution required).

## Errors

- `SourceCoverageError(ValueError)` — the fetched data does not adequately cover the
  requested bbox (nodata, blank imagery, missing ocean cell).
- `SourceFetchError(ValueError)` — the HTTP request itself failed; the message names the
  source and URL (no secrets are ever part of these requests).
