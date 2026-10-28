### Added

- Route Story map bundles now carry an offline road layer: `roads.geojson`, the OpenStreetMap `highway` ways (motorway through residential, plus link ramps) inside the route bbox, fetched through the existing Overpass mirror list and recorded in a manifest `roads` block with ODbL provenance. The bundle manifest schema is now `route-story-bundle/2`; `/1` bundles still verify. A failed or empty road fetch marks the layer degraded without failing the prepare, and a cached bundle with degraded roads re-fetches only its roads.
