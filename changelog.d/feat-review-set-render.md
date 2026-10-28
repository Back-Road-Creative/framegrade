### Added

- `gomoveshift.pipeline.review_set_render.render_review_set`: library helper that renders paired source/output PNG stills and full-speed re-encoded excerpts for a list of (source time, output time) pairs, with a `provenance.json` (package version, git sha or null, ffmpeg version, effective drive config, input sizes and sha256, synthetic flag). Refuses a non-empty output directory. Not wired into the CLI; nothing existing changes.
