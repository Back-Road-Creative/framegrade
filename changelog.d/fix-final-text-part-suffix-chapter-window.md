### Fixed

- A multi-part LF title ("… — Part 1 of 2") no longer refuses its final-text review: the GMS text policy (now `gms-route-claims-2`) counts "part" as filler. Reviews stored under the old policy go stale, so Phase D re-runs for pending LFs.
- A chapter's `segment_places` cites only retained-route places. Region ids there refused the LF (`route_place_unknown:region-…`).
- `reattach_with_chapters` on metadata that already carries chapters replaces them instead of raising `TypeError`.
