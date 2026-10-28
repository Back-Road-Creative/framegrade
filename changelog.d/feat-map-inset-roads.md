### Added

- `--map-inset roads` (`config.drive.map_inset = "roads"`): an opt-in third mode that renders like `route` and also draws the bundle's nearby roads under the route line, with the OpenStreetMap attribution on the credit strip. It refuses before any encode when the bundle has no usable road layer (missing file or manifest entry, degraded, or zero roads) and when no road lies in view. `route` and `off` are unchanged.
