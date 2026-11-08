### Added

- `analysis.sign_label.derive_sign_label`: a county-level label ("Colleton County, SC") derived from an LF's road signs, for an LF the camera never located. It is produced only when at least `sign_verify_min_corroborating_hits` (default 2) sign hits from distinct seconds name one county, no other county is named, exactly one state is named, and the LF has zero GPS fixes; otherwise it refuses with a reason naming the rule. The label carries its evidence (hit count, seconds, sign texts) for tier evidence and the OST system review. This release only derives; a later change writes it into the per-LF labels and shows it in `--list-lfs`.

### Changed

- `ocr.parse_sign_claims`: a full state name is a state claim only beside a place claim ("Colleton County South Carolina") or a boundary cue ("Welcome to", "State Line", "Entering", "Leaving", "Welcomes You"). A destination on an interstate guide or distance sign ("Winchester / West Virginia / Exit 310") is no longer a claim about where the camera is, so it can neither mislabel a GPS-less LF nor raise a false state conflict.
