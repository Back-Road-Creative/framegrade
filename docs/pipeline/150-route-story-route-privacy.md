## Route start/end privacy trim

`gomoveshift/pipeline/route_story/route_privacy.py` — pure module, no I/O, no
CLI wiring yet.

- **`RouteTrim(start_m=0.0, end_m=0.0)`**: the requested along-track privacy
  radius to drop from each end of a route, in meters.
- **`trim_route(segments, trim) -> TrimResult`**: drops every point within
  `start_m` of the route's first point and within `end_m` of its last point,
  measured as cumulative travelled distance in recording order (never by
  wall-clock offset or point index — a paused/slow start would otherwise leak
  more of the operator's driveway than intended). Distance accumulates only
  across edges actually measured within a segment; a gap between segments
  carries the running total forward without adding distance for the
  unobserved crossing, so a short segment sitting entirely inside the trim
  radius is dropped whole rather than anchoring a phantom edge. A segment
  left with fewer than two points after trimming is dropped entirely.
  `RouteTrim(0, 0)` is a no-op and returns the input segments unchanged
  (same list, same point objects).
- **What it guarantees**: once wired, a point removed by this trim is absent
  from every downstream surface — the film, dashboard, HUD, metadata, and any
  future export — because nothing downstream re-reads the original segments.
  Removal only: points are never moved, offset, or synthesized to paper over
  a gap.
- **What it does not do**: it does not touch photos — still privacy (blur,
  EXIF GPS strip) is `privacy.py`'s gate, entirely separate. It does not blur,
  fuzz, or approximate the remaining route; the points it keeps are the exact
  source fixes.
- **Over-trim is a hard fail, not a silently empty film**: a `RouteTrim` that
  would remove the entire route (leaving zero segments with at least two
  points) raises `RouteTrimError`, with the route's actual along-track length
  in the message.
- **`receipt_fragment(result) -> dict`**: `{"route_trim": {"start_m", "end_m",
  "removed_start_points", "removed_end_points", "removed_start_m",
  "removed_end_m"}}`, ready for a later change to merge into `film_receipt`.
  `removed_start_m`/`removed_end_m` are the *actual* distance dropped — point
  granularity means this is always `>=` the requested `start_m`/`end_m`,
  never less.
- **Wiring status**: wired to `route-story-film` (`film.py`'s `--trim-start-m`/`--trim-end-m`,
  applied to `select_gpx_day`'s segments before anything else reads them; see
  `16-route-story-film.md` "Route trim"). `receipt_fragment` is folded into `film_receipt` under
  `route_trim`, always present once `--receipt` is passed.
- **`public_receipt.py`**: `public_receipt(receipt, trim_result) -> dict` derives a
  distribution-safe copy of a `film_receipt` dict for a second, public artifact. Refuses
  (`PublicReceiptRefusedError`) when `trim_result` is absent, `receipt` has no `route_trim`, or the
  applied trim is `RouteTrim(0, 0)` — an explicit, nonzero privacy trim must exist before any
  public artifact can exist. Keeps `route_trim`, `route_summary`, `outputs`, and `map_renderer`
  verbatim; reduces `national_imagery`/`regional_imagery`/`local_imagery` to `{"credit": ...}`
  (drops `bounds`/`sha256`/`dimensions`); reduces `map_labels` to `{"count": ...}` (drops
  `names`/`kinds`); reduces `photos` to `{"count": ...}`. See `16-route-story-film.md`
  "Route trim" for the `--public-receipt` CLI flag.

## Interior privacy zones (mid-route redaction)

Extends the same module with a second, complementary redaction: `RouteTrim`/`trim_route` only ever
cut a prefix/suffix off the route; a `PrivacyZone` masks a radius anywhere along it (a stop at
someone's house partway through a drive, say).

- **`PrivacyZone(center_lat, center_lon, radius_m)`**: one requested interior privacy radius, in
  meters. Construction raises `ValueError` for a latitude outside -90..90, a longitude outside
  -180..180, or a radius that is not above 0 -- a swapped LAT,LON or a zero radius would
  otherwise mask nothing, silently.
- **`mask_zones(segments, zones) -> ZoneMaskResult`**: drops every point within any zone's radius
  (great-circle distance, same `distance_m` helper `trim_route` uses). Reuses `trim_route`'s
  points-only-removed, whole-segment-drop rules, with one difference forced by the zone sitting
  mid-segment rather than at an end: **a zone that splits a segment yields two kept segments with a
  real gap between them** — the removed hole is never bridged into a falsely-continuous segment. A
  kept run left with fewer than two points (a zone masking a neighbour down to a stray fix, or an
  entire short segment sitting inside a zone) is dropped whole, same as `trim_route`. Overlapping
  zones never double-count a point or its along-track span. `ZoneMaskResult.removed_points` is
  every input point absent from the result; `removed_m` is the along-track span the radius-masked
  points actually covered (the size of the hole cut into the route).
- **Over-mask is a hard fail, same as over-trim**: zones that mask the entire route (zero kept
  segments) raise `RouteTrimError`, with the route's actual along-track length in the message —
  the same error family `trim_route` raises.
- **`receipt_fragment(result, zone_result=None)`**: when `zone_result` is supplied, the returned
  `route_trim` fragment gains a `zones` block: `{"count", "removed_points", "removed_m"}` — counts
  and removed points/metres *only*. It never carries a zone's `center_lat`/`center_lon`/`radius_m`,
  so a public receipt (which copies `route_trim` verbatim, see `public_receipt.py` above) can never
  leak a zone's location.
- **Wiring status**: wired to `route-story-film` (`film.py`'s repeatable `--privacy-zone
  LAT,LON,RADIUS_M`, applied immediately after the existing `--trim-start-m`/`--trim-end-m` trim —
  same insertion point — so every downstream surface reads the masked segments); see
  `16-route-story-film.md` "Route trim".
