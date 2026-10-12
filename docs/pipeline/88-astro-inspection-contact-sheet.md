## Astro inspection contact sheet

`gomoveshift photos astro-inspect MASTER.tif --out sheet.jpg` is READ-ONLY: it loads an already-
stacked master with a bare `tifffile.imread` (never `astro_develop`/`recover_balanced`) and never
writes back to it. `astro_contact.inspect_master` crops five fixed 400px squares — the four corners
and the centre — at their true 1:1 resolution (never resized by `compose_contact_sheet`, which only
shrinks a panel wider than `CONTACT_PANEL_WIDTH`), and labels each with the region's median star
shape from `astro_triage.detect_stars`/`_measure_star_shapes`/`_median_shape` (FWHM major/minor px,
ellipticity) so judging corner star roundness/coma no longer means opening the TIFF by hand. A
region with no detected star prints "no stars detected in region" rather than a fabricated zero.
