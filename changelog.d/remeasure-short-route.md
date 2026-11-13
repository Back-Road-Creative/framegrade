### Added

- `gomoveshift queue remeasure-short-route --id N --stages-dir <run>/_stages` re-derives the
  route `bins` a Short planned before the place gate lacks, from the day's `gps_points_<date>.json`
  and the clip range in `shorts_ost_audit_<date>.json`, and writes it to the sidecar's
  `route_record`. Dry run unless `--apply`; refuses (writing nothing) when a source is gone, the
  clip does not match the Short's media, or the clip holds no GPS fix. It never writes to YouTube
  and never touches the place gate; `recover-terminal`'s gates then decide.

### Changed

- The place gate's "route record has no 'bins'" refusal now names `queue remeasure-short-route`.
