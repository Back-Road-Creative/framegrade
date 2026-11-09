### Added

- `--bracket-gpx BEFORE AFTER`: the specificity ladder's bracket-route tier, fed for
  the first time. A GPS-less drive's on-screen label is COMPUTED from the last fix
  recorded before the footage and the first fix after it (each a `.gpx` or a GoPro
  recording): the narrowest region both fixes reverse-geocode to (same county →
  "County, State" / "County, Country"; else the state; else the country), burned on
  every LF, awarded tier 3 (county) or 4 (state/country) with the whole basis (both
  fixes, their gap, both geocodes) on `state.bracket_route`, in the ladder evidence
  and in the provenance sidecar. Refused when the fixes are out of order, more than
  `BRACKET_MAX_GAP_H` (36 h) apart, a footage date lies outside them, a geocode
  misses or they share no region; refused beside `--location`, `--per-lf-location`,
  `--day-gpx` or `--placeless-label`. `check_specificity_ladder` passes only the
  computed text; the OST system review agrees with it, so the drive does not wait
  on a person. Backlog rows carry the pair as `bracket_gpx` (the launcher skips a
  row whose files are missing or that also types a label). Until now a drive with
  no fix at all could carry only a claim-free placeless label even when the yaml
  notes recorded where the surrounding GPS put it.
