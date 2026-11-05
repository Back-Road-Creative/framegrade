## Retained-route region proof

The long-form pipeline checks the location label against GPS from footage it
keeps before encoding. The sampled reverse-geocoder gate refuses a county or
state fallback when it sees an unresolved sample or a GPS gap. This is safe,
but it can refuse a route that stayed well inside that region.

`boundary_proof.CensusBoundaryProver` adds a narrower, automatic fallback for
US counties and states. It reads the exact 2026 Census TIGER/Line polygon for
the candidate region. The archives are fetched only from
`www2.census.gov/geo/tiger/TIGER2026/{COUNTY,STATE}/`, cached under the
process's XDG cache (or `~/.cache/gomoveshift/census-2026`), and checked against
the SHA-256 values pinned in code on every read. A missing, changed, corrupt or
unknown archive, redirect, or response over 100 MB refuses the fallback.
Tests use synthetic local ZIP fixtures and make no network requests.
Other US recording years retain the sampled-only refusal until annual sources are pinned.

Irish county claims (no state, country Ireland) use Tailte Éireann's
"Counties - National Statutory Boundaries - Ungeneralised 2026" GeoJSON
(CC BY 4.0, © Tailte Éireann), pinned by SHA-256 and cached beside the Census
archives. The download starts at `data-osi.opendata.arcgis.com` and may follow up
to 3 https redirects to the two ArcGIS hosts listed in
[84](84-network-allowlist-credential-stores.md); any other host, a response over
150 MB, a map not in Irish Transverse Mercator (EPSG:2157), or an invalid polygon
refuses. Fixes are projected to ITM metres. Because a 2026 map may postdate the
recording, an Irish proof accepts any recording up to 2026 and adds a 250-metre
inward margin to every allowance below; the proof records `margin_m` and
`country: "IE"`.

For every retained segment, the verifier checks each valid source fix is at
least 100 metres inside the polygon. It also checks that the distance to the
boundary exceeds 100 metres (plus the Irish margin) plus the distance possible
at 70 metres/second for every interval between fixes and every cut edge. A retained cut with no
fix needs valid fixes immediately before and after it; excluded footage never
adds a place vote. The proof records the archive hash, region, speed limit,
error allowance and longest interval. The original gap list, sampled coverage,
and `containment_proven: false` are preserved. This is a **bounded model**, not
proof of an actual path or an absolute guarantee: the GPX has no DOP or fix
type and cannot rule out a larger GPS error or an impossible-speed event.

The pre-encode route gate and label dry run use the same provider. A county
label with a state suffix also needs the state proof. A route with any GPS gap is
proven even when its label passes, because Waypost reviews every gap of a region
claim, including one bridged to the county. Waypost chooses a title
against the retained route before encode; Phase D reviews each boundary proof.
The pinned reviewer must expose `BOUNDARY_PROOF_VERSION = "tiger2026-tailte2026-bounded-v2"`
or the fallback refuses before encoding. Never alter coverage, gaps or outside
samples to make an unsupported reviewer agree.

A retained route recorded in February was measured against the official 2026
Beaufort County and South Carolina polygons using its existing plan and both
GPX files: 14 retained windows, about 4,321.5 seconds and 4,021 valid fixes.
Both region proofs passed this bounded model, including one three-second
window whose nearest source fixes bracket the cut. `Longwood, SC` remains an
unsupported town label; the county fallback needs a released pipeline and
upload-disabled acceptance run before publication.

Sources: [Census 2026 TIGER/Line](https://www.census.gov/geographies/mapping-files/2026/geo/tiger-line-file.html)
(legal boundaries as of January 1, 2026; released October 14, 2026);
[Tailte Éireann county boundaries 2026](https://data-osi.opendata.arcgis.com/api/download/v1/items/a2dd7924915e4c74ad6a417e33c394eb/geojson?layers=1)
(CC BY 4.0, © Tailte Éireann).
