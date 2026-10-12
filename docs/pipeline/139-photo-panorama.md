## Panorama stitch (`gomoveshift/pipeline/photos/panorama.py`)

`stitch_panorama(paths, out_path)` fuses 2+ ordered, overlapping photos into
one wide frame via OpenCV's `Stitcher`, gated by a seam-quality check that
runs on the stitched frame **before** any write — a rejected stitch never
leaves a partial file at `out_path`.

- Uses `cv2.Stitcher_create(cv2.Stitcher_SCANS)`. SCANS mode fits this
  pipeline's inputs — overlapping frames from roughly one spot with a
  horizontal pan — better than `Stitcher_PANORAMA`'s rotational/spherical
  warp model, which over-crops a near-planar pan (observed: a 396px-wide
  input warped down to 220px on this module's own test scene).
- Fewer than 2 paths, or a file OpenCV can't decode, raises `PanoramaError`
  before the Stitcher ever runs.
- A non-`Stitcher_OK` status maps to a specific message (need more
  overlapping images / homography estimation failed / camera parameter
  adjustment failed) rather than a bare status code.
- Seam QA rejects the stitched frame when either holds: the non-black pixel
  fraction is below `MIN_VALID_AREA_FRACTION` (0.5), or the output width is
  narrower than the widest input — both signs of a degenerate warp that
  `Stitcher_OK` alone does not catch.
- `PanoramaResult` (returned only on success) carries `out_path`, `width`,
  `height`, `valid_area_fraction`.

**Console (`gomoveshift/console/photos.py`).** `validate_panorama_paths`
mirrors `validate_develop_paths` — plus a minimum-of-2-photos check — and
`run_panorama` calls it before calling `stitch_panorama` in-process (no CLI
subprocess; there is no `gomoveshift photos panorama` command). Not yet
wired into the `/photos/run` form — that needs its own branch in
`console/app.py` (and, if this should launch as a queued job rather than
block the request thread the way `ingest`/`develop` do, a
`gomoveshift photos panorama` CLI subcommand in `cli/parser.py` for the job
launcher to invoke) — both outside this module's scope.
