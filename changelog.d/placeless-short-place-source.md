### Fixed

- A GPS-less drive's shorts no longer fail the `places_verified` gate (check 20) and the drive-text review no longer refuses a bracket label. A short records `place_source` `bracket_route` only when its text is exactly the label of a `--bracket-gpx` record within its gap cap, and `placeless_label` only when the text is a claim-free placeless label. The gate re-checks that evidence instead of trusting the string; `lf_label` and a short with clip regions stay unverified, and any other unbacked name is still refused.
