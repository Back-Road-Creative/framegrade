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
Other recording years retain the sampled-only refusal until annual sources are pinned.

For every retained segment, the verifier checks each valid source fix is at
least 100 metres inside the polygon. It also checks that the distance to the
boundary exceeds 100 metres plus the distance possible at 70 metres/second
for every interval between fixes and every cut edge. A retained cut with no
fix needs valid fixes immediately before and after it; excluded footage never
adds a place vote. The proof records the archive hash, region, speed limit,
error allowance and longest interval. The original gap list, sampled coverage,
and `containment_proven: false` are preserved. This is a **bounded model**, not
proof of an actual path or an absolute guarantee: the GPX has no DOP or fix
type and cannot rule out a larger GPS error or an impossible-speed event.

The pre-encode route gate and label dry run use the same provider. A county
label with a state suffix also needs the state proof. Waypost chooses a title
against the retained route before encode; Phase D reviews each boundary proof.
The pinned reviewer must expose `BOUNDARY_PROOF_VERSION = "tiger2026-bounded-v1"`
or the fallback refuses before encoding. Never alter coverage, gaps or outside
samples to make an unsupported reviewer agree.

A retained route recorded in February was measured against the official 2026
Beaufort County and South Carolina polygons using its existing plan and both
GPX files: 14 retained windows, about 4,321.5 seconds and 4,021 valid fixes.
Both region proofs passed this bounded model, including one three-second
window whose nearest source fixes bracket the cut. `Longwood, SC` remains an
unsupported town label; the county fallback needs a released pipeline and
upload-disabled acceptance run before publication.

Source: [Census 2026 TIGER/Line](https://www.census.gov/geographies/mapping-files/2026/geo/tiger-line-file.html)
(legal boundaries as of January 1, 2026; released October 14, 2026).
