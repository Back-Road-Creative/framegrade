### Added

- Panorama stitching measures each seam's ghosting and exposure step (`measure_seam`). A seam past `MAX_SEAM_GHOST_FRACTION` or `MAX_SEAM_LUMA_STEP` is logged at WARNING and named in `PanoramaResult.message`; the panorama is still written unless `strict_seams=True`, which refuses it before the write. The measurements are on `PanoramaResult` and in the console's `run_panorama` dict (`seam_ghost_fractions`, `seam_luma_steps`, `seam_message`).
