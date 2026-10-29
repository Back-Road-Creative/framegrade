### Added

- `gomoveshift review-set pairs`: maps scan windows to `[source_t, output_t]` pairs through the stops document's output times, and lists every window it cannot pair (`cut`, `no_output_times`, `outside_runs`, `not_in_output`) instead of estimating. `review-set render --pairs` now also accepts that document.
- A drive writes the day's usable GPS fixes to `<stages_dir>/gps_points_<date>.json` beside `stop_mask_<date>.npz`, the file `review-set scan --gps` reads. The stop-mask file is unchanged.

### Fixed

- `docs/pipeline/169-review-set.md` showed the wrong score-cache and GPS file names for `review-set scan`.
