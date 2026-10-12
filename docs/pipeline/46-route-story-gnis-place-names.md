## Route Story GNIS place names (`gomoveshift/pipeline/route_story/gnis.py`)

Source for the Route Story label ladder (peaks, gaps, streams, populated places, trails):
USGS The National Map's staged Domestic Names export, one small pipe-delimited zip per
state. `states_for_bbox` picks which states a route's bounding box actually touches from
an embedded coarse per-state bounding-box table (no network round trip needed to answer
that), `fetch_gnis_state` downloads each touched state's zip once and caches it under
`cache_dir/gnis/<ST>/`, and `parse_gnis` reads the zip's header row by name (never by
column position) to build `GnisFeature` records filtered to `DEFAULT_CLASSES` and the
bbox. The real file opens with a UTF-8 byte-order mark, uses CRLF, names the state
`state_name` (not `state_alpha`) and carries no elevation column, so the reader decodes
`utf-8-sig` and `elevation_m` is `None` for staged-products rows. Reading it as plain
`utf-8` turned the first header into `\ufefffeature_id`, blanked every id, and the
id-keyed dedupe collapsed a whole trip to one label; `gnis_features_for_bbox` now also
keys an id-less row on its name, class and position so that failure cannot recur. `to_label_geojson` renders the result in the same `name`/`kind`/`source` property
shape `prepare.py`'s OSM label writer already uses, so the two label sources merge
without a reshape. This module fetches and parses only; the merge, dedupe, and
`labels_degraded` loud-degradation path live in `prepare.py` (see *Route Story bundle
preparation* above).
