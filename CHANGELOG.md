# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.15] - 2026-10-30

### Changed

- Pinned Waypost (`seo-manager`) to v0.4.0 (`97334eb`, Back-Road-Creative/waypost#95). The final-text review now accepts a region-only drive that has no town to name, and refuses a title, description or chapter that drops the region (Back-Road-Creative/waypost#94).

### Fixed

- The Windows release gate no longer refuses a tag pushed while main's CI is still running on that commit: it waits for the run, up to 30 minutes, and refuses only when CI failed, never started, or the wait ran out. v0.3.14's installer build was refused this way.
- `--day-gpx` slices each later chapter of a split Insta360 recording at its own start: the filename time plus the earlier chapters' durations. Before, every chapter was sliced at the recording start, so the second 30-minute chapter carried the first chapter's GPS (Phase A's 60 min start tolerance let it through). A chapter whose predecessor is missing from the source dir is refused. See `docs/pipeline/71-generic-source-ingest-full-drive-lf-mode.md`.
- Drive chapters no longer bind a town named inside a region's name ("Beaufort County" is not the town of Beaufort), and a chapter's window holds route places only, so Waypost's review stops refusing a whole long-form over `route_place_unknown:region-…`.
- The long-form work-space gate now prices every long-form the run still burns, not just the current one. Each final stays on the cache volume while the next long-form writes its three artifacts, so the gate before a day's first concat now sizes the peak: the worst long-form's work plus the finals burned before it. A later long-form whose final already exists is not counted again. Before this, a day could pass the gate for LF 1 and then stop at LF 2's gate hours later (Ace Basin, 2026-10-27: LF 2 needed 145.8 GiB, and ~125 GiB was left beside LF 1's final).
- A multi-LF day no longer ships an LF with no shorts. The day-wide shorts planner could put every top-scoring window in one LF's footage; the per-LF cap then trimmed that LF and nothing refilled the others. `fill_shorts_to_per_lf_capacity` now plans each under-capacity LF's missing shorts from that LF's own source window (stop mask and GPS-gap veto kept, other LFs' footage and already-picked shorts masked). The SHORTS-CONTRACT gate now judges the final post-cap, post-refill count, so an LF whose window truly cannot hold its share is refused unless `--allow-short-count` is passed.

## [0.3.14] - 2026-10-29

### Added

- `plan_inserts(..., still_motion="push_in")` renders still inserts with the existing eased push-in (up to the existing maximum scale) as a per-frame card sequence; the default `"none"` keeps the static card and encode unchanged.
- `InsertPlan.upscaled` and `InsertPlan.upscale_factor` record whether, and by how much, a clip insert is upscaled to the long-form frame.
- `--short-reframe center|tracked`, `--output-cadence fixed|source` and `--insert-still-motion none|push_in` on the drive command: opt-in switches for the tracked vertical reframe, source-rate output and the insert still push-in. Each defaults to today's behaviour, passes nothing extra to the encode when absent, and is remembered across a resume. `--output-cadence source` refuses a source rate it cannot keep (anything but 29.97, 30, 59.94 or 60 fps) rather than resampling it.
- `gomoveshift review-set {scan,stops,render}`: command group over the review-set libraries. `scan` lists candidate windows per condition from a score cache, `stops` writes the stop/kept runs of a sequence, `render` writes paired stills, excerpts and provenance. Documented in `docs/pipeline/169-review-set.md`. No existing command changes.
- `gomoveshift review-set pairs`: maps scan windows to `[source_t, output_t]` pairs through the stops document's output times, and lists every window it cannot pair (`cut`, `no_output_times`, `outside_runs`, `not_in_output`) instead of estimating. `review-set render --pairs` now also accepts that document.
- A drive writes the day's usable GPS fixes to `<stages_dir>/gps_points_<date>.json` beside `stop_mask_<date>.npz`, the file `review-set scan --gps` reads. The stop-mask file is unchanged.
- `gomoveshift.pipeline.review_set_render.render_review_set`: library helper that renders paired source/output PNG stills and full-speed re-encoded excerpts for a list of (source time, output time) pairs, with a `provenance.json` (package version, git sha or null, ffmpeg version, effective drive config, input sizes and sha256, synthetic flag). Refuses a non-empty output directory. Not wired into the CLI; nothing existing changes.
- `gomoveshift.pipeline.review_set_scan.scan_review_set`: library that proposes candidate review windows per lighting and motion condition (daylight, overcast, dusk/night, golden hour, tunnel, creeping, parked traffic, HDR gradients, GPS gaps) from the per-second score cache and, when supplied, the stop-mask states, GPS fixes and tonemap sidecar. Each window records the rule and thresholds that fired. Backlight, foliage and wet windshield report `unsupported` instead of being guessed; a missing optional input reports `no_input`. Nothing calls it yet, so existing output is unchanged.
- `gomoveshift.pipeline.review_set_stops` (library only, no CLI yet): reads a day's `stop_mask_<date>.npz` and emits one schema-versioned JSON per sequence with the contiguous MOVING/STOPPED/CREEPING/UNCERTAIN runs in source seconds, the kept/cut decision for each (from `mask`, so CREEPING is never cut), the mask summary, and, when an assembly receipt is given, each kept run's output time range from `EditTimeline`. Without a receipt the output times are omitted and the document says why.
- Opt-in source-cadence output (library parameter `output_fps`): a 59.94/60 fps source can keep its own rate through the master encode and the OST burn instead of being decimated to 29.97. The default is unchanged and stays 29.97. Only the source's own rate is accepted (`30000/1001`, `30`, `60000/1001`, `60`); any other value refuses with a clear error. The keyframe interval stays 2 s at the chosen rate, and the `fps` filter stays the chain tail after the privacy-blur composite.
- Opt-in tracked vertical reframe for shorts: `render_short_ffmpeg(..., reframe="tracked")` slides the 9:16 crop to follow the road using a low-pass, a dead zone and a pan-speed cap, so it never chases. The default `reframe="center"` produces the identical filter and ffmpeg arguments. Crop size, scaler and the declared upscale are unchanged, and shorts stay muted.

### Fixed

- `docs/pipeline/169-review-set.md` showed the wrong score-cache and GPS file names for `review-set scan`.
- A drive description's hook and `Location` line now name the first place in the LF label that holds every retained second, not the first town the label happens to say: "Beaufort County, South Carolina" over a route that also crosses Colleton County now prints "South Carolina", not "Beaufort". The upload's zero-tolerance place gate also runs where the text is made, so an audit (`--skip-upload`) pass refuses a description the upload would refuse (`place_outside_text`) instead of agreeing and failing at `videos.insert`.
- `gomoveshift drive --dry-run` (and `--list-lfs`/`--label-dry-run`) with `--youtube-upload` no longer runs the post-Phase-D journal hook, so a plan render no longer creates a draft journey on the site and then fails with "state['youtube_uploads'] is empty".

## [0.3.13] - 2026-10-28

### Added

- Regular long-form videos can carry a route-only map inset: one clip per video, rendered from an offline bundle and timed through the cut receipt (saved beside the concat as `<concat>.receipt.json`), composited in the existing OST encode and recorded in drive state. Not reachable until the `--map-inset` flags land; nothing changes when it is off.
- `gomoveshift/pipeline/map_inset.py`: renders an opt-in, route-only corner map inset as an RGBA `.mov` overlay plus a JSON receipt, from an offline Route Story bundle (imagery with the recorded route and a position marker, no road vectors, no network). The marker follows the source clock through the edit timeline and hides during cuts, insert holds, crossfades and GPS gaps. Privacy trim and zones apply before any framing or drawing, and the render refuses, listing every reason, rather than write a partial clip.
- `--map-inset roads` (`config.drive.map_inset = "roads"`): an opt-in third mode that renders like `route` and also draws the bundle's nearby roads under the route line, with the OpenStreetMap attribution on the credit strip. It refuses before any encode when the bundle has no usable road layer (missing file or manifest entry, degraded, or zero roads) and when no road lies in view. `route` and `off` are unchanged.
- `--map-inset off|route` on the drive command (plus `--map-inset-bundle`, `--map-inset-trim-start-m`, `--map-inset-trim-end-m` and a repeatable `--map-inset-privacy-zone`): opt-in route-only map inset on regular long-form videos, rendered from an offline bundle and composited inside the existing OST encode. Off is unchanged. `route` refuses before any encode, listing every missing input (bundle, trim, GPS), and never falls back silently. The settings persist across a resume, are part of each long-form video's re-burn fingerprint, and the per-video inset receipt is recorded in drive state.
- `apply_ost` takes an optional `map_inset` (`MapInsetOverlay`: an RGBA clip plus an x/y position). It is composited after the title and logo in the same encode, in explicit 10-bit formats so a 10-bit master keeps its gradients; the inset drops away when its clip ends. A missing clip is an error, never a silent encode without it. Without `map_inset` the ffmpeg command is unchanged.
- Route Story map bundles now carry an offline road layer: `roads.geojson`, the OpenStreetMap `highway` ways (motorway through residential, plus link ramps) inside the route bbox, fetched through the existing Overpass mirror list and recorded in a manifest `roads` block with ODbL provenance. The bundle manifest schema is now `route-story-bundle/2`; `/1` bundles still verify. A failed or empty road fetch marks the layer degraded without failing the prepare, and a cached bundle with degraded roads re-fetches only its roads.

### Changed

- Pinned `keycut` to v0.2.0. The day-master cut now goes through keycut's `extract_and_concat_ranges_with_receipt` and checks the live assembly receipt with `load_receipt`; a receipt that fails validation makes `extract_and_concat_segments` log the failing field and return None, since a time mapping built from it would be wrong.
- Pinned Waypost (`seo-manager`) to v0.4.0 (`97334eb`, Back-Road-Creative/waypost#94, #95). The final-text review accepts a town-less route (county and state regions only) instead of refusing it with `missing_field:route_places` and `region_claim_malformed`.

### Fixed

- A multi-part LF title ("… — Part 1 of 2") no longer refuses its final-text review: the GMS text policy (now `gms-route-claims-2`) counts "part" as filler. Reviews stored under the old policy go stale, so Phase D re-runs for pending LFs.
- A chapter's `segment_places` cites only retained-route places. Region ids there refused the LF (`route_place_unknown:region-…`).
- `reattach_with_chapters` on metadata that already carries chapters replaces them instead of raising `TypeError`.

## [0.3.12] - 2026-10-27

### Added

- `photos develop --hdr` now aligns each bracket frame onto the middle exposure before the merge (median-threshold bitmap translation, applied at the frame's own bit depth), so a handheld bracket no longer ghosts. A shift over 3% of the shorter side is skipped and recorded, a tripod bracket is bit-identical to before, and `--no-bracket-align` / `photos.develop_hdr_align = false` restores the unaligned merge. Rotation and deghosting are not handled. See `docs/pipeline/83-photo-pipeline.md`.
- `EditTimeline` in the same module: `output_to_source` (none in an insert hold, two weighted positions in a crossfade) and `source_to_output` (`None` for a cut second, never interpolated), built from a receipt plus optional `Crossfade` and `Insert` extras.
- `gomoveshift.pipeline.edit_timeline`: frozen dataclasses and a strict loader (`load_receipt`, `ReceiptError`) for the `keycut-assembly-receipt/1` assembly receipt, with a frozen test fixture. Every violation names its field. Nothing reads a receipt yet. Described in `docs/pipeline/167-edit-timeline.md`.
- Panorama stitching measures each seam's ghosting and exposure step (`measure_seam`). A seam past `MAX_SEAM_GHOST_FRACTION` or `MAX_SEAM_LUMA_STEP` is logged at WARNING and named in `PanoramaResult.message`; the panorama is still written unless `strict_seams=True`, which refuses it before the write. The measurements are on `PanoramaResult` and in the console's `run_panorama` dict (`seam_ghost_fractions`, `seam_luma_steps`, `seam_message`).
- `photos develop --hdr` fuses a 16-bit (RAW) bracket as float32 (OpenCV's 0-255 scale) instead of stepping each frame down to 8 bits first, so a smooth gradient keeps its sixteen-bit tonal precision through the merge; an all-8-bit bracket is byte-identical to before. This is still exposure fusion, not scene-linear HDR. See `docs/pipeline/83-photo-pipeline.md`.
- Opt-in GPS event captions on regular long-form videos: `--gps-captions {off,events}` (`config.drive.gps_captions`, default `off`, saved in drive state: a resume without the flag keeps the choice, a differing flag wins and is saved). `events` burns one "Entering <town>" card per town event of that LF's own chapters. A card naming a place no retained fix of the LF resolves to is suppressed and recorded (`state.regular_lf_captions`), the rest are spaced by `regular_lf_lower_third_min_gap_sec` (45 s) and shown for `regular_lf_lower_third_duration_sec` (8 s), separate from the superlong knobs. With captions requested and no retained-route evidence for an LF, the run refuses before any encode. `off` renders exactly as before. Cards are placed at chapter time plus the lead the file actually carries (probed `prepended` minus `concatenated`), since chapter times are LF-body-relative.
- The per-day stop mask is now derived from a per-second motion state (`MotionState`: moving, stopped, creeping, uncertain; `classify_motion_states`). `stop_mask_<date>.npz` gains an int8 `states` array beside `mask`; `mask` keeps its meaning (True is a stopped second, creeping is never True). Default output is unchanged.
- `stop_detection.creep_policy` (`remove`, the default and today's behaviour, or `keep`) and `stop_detection.creep_min_speed_kmh` (default 2.0, provisional pending tuning on labelled real drives). With `keep`, sustained GPS speed from `creep_min_speed_kmh` up to `stop_speed_kmh` is creeping and not masked.
- The day-first and per-day stop mask now honours `stop_detection.hysteresis`, which until now only affected the legacy optical dead-spot pass. The same key therefore now also changes the mask when enabled. It stays off by default.
- `grade_fallback` on the single-pass result and in the `.tonemap.json` sidecar records when the global LUT bake failed and the grade fell back to the 8-bit `eq` chain. `drive.strict_grade` (default off) refuses that fallback instead.

### Changed

- Changelog entries are now fragments in `changelog.d/<name>.md` (one or more `### <Section>` blocks), not edits to `CHANGELOG.md`, so parallel PRs no longer conflict on it. `tools/release.py` folds the fragments into the release by section (Keep a Changelog order, entries already under `[Unreleased]` first), then deletes them. A fragment with no heading, an unknown section or no entries is refused by the release tool and by `tests/test_changelog.py`.
- The shorts edited-to-source mapper, the chapter source-to-edited mapper and the assembler's chapter trim adjustment now share `EditTimeline`; their signatures and answers are unchanged. `MarkerMerger.map_to_edited_time` takes an optional `crossfade` (default 0, which is all the chapter generator can pass).
- The stabilise step is judged on camera motion instead of luma jitter, which cannot see a translating image. It keeps the vidstab output only when the shake residual (the high-frequency part of the phase-correlation camera trajectory, so a deliberate pan is not shake) drops to at most 75% of the input's and black-border coverage stays at or under 5%; otherwise it falls back to the input and logs why. Deflicker keeps the luma-jitter gate. The picture-steps record gains `input_residual`, `output_residual` and `border_coverage` for the stabilise step.

### Fixed

- HDR tonemap and wide-gamut conversion chains now end in 10-bit `yuv420p10le` (one shared `TONEMAP_OUTPUT_FORMAT`) instead of 8-bit `yuv420p`, which put an 8-bit hop (at most 256 levels per channel) ahead of the 16-bit grade. A 10-bit ramp through the chain keeps its levels.
- keycut is pinned at v0.1.2. It sizes its ffmpeg and ffprobe timeouts to the file they read, at a 20 MB/s floor, instead of using flat windows. A 110 GB day master on `/mnt/c` overran the 600s keyframe probe. Keyframes came back unknown, so every long-form range fell back to a CPU re-encode that decoded the master from the start, and a range late in the day could not finish inside its flat 3600s. keycut's re-encode fallback now seeks on the input side, decoding at most one GOP before the cut.
- `montage_builder.reencode_range_argv`, the trail opening's seam re-encode, follows keycut's new argv (`-ss S -i src -ss 0 -t D`). A seam late in a long-form video no longer decodes the whole film before it.
- Drive LF description chapters sit after the montage: GPS chapters were timed against the LF body, so every one after 0:00 was early by the montage length and disagreed with the photo markers. Phase D now shifts them by the lead the file carries (the recorded lead, or the trail opening's length); no lead record leaves them unchanged.
- Deflicker and vidstab-transform intermediates no longer inherit ffmpeg's defaults (h264, crf 23 for an `.mp4`). They are written as x265 at the long-form CRF and preset, keep the input's pixel format (10-bit stays 10-bit), and copy the input's colour range, matrix, primaries and transfer from an ffprobe, with primaries and transfer also set through `-x265-params`.
- The deflicker and vidstab-transform intermediates now pick their encoder like the day master: `hevc_nvenc` (long-form NVENC preset and CQ, main10 `p010le`, no `-r`) when a GPU works, libx265 only as the fallback. The choice is injected as `has_nvenc` from `stabilize.nvenc_available()`, so tests need no GPU.

## [0.3.11] - 2026-10-27

### Added

- Windows bundle builder (`packaging/windows/build_release.py`), launcher and per-user Inno Setup script: the embeddable Python 3.12 with the win_amd64 GPU wheels, a privacy audit and a 2 GiB per-asset size gate. Steps in `docs/install.md`.
- Windows release workflow (`.github/workflows/windows-release.yml`): a `vX.Y.Z` tag push gates the tag (on `main`, matches `pyproject.toml`, `ci` passed that commit), builds the installer on a hosted `windows-2025` runner, proves the installed app (silent per-user install as a standard user, Defender scan, `/healthz`, `gomoveshift scan` on a synthetic clip, uninstall) and creates a draft release with the installer and `SHA256SUMS`. A manual run is a dry run that creates no release. Steps in `packaging/windows/README.md`.
- The Windows bundle carries the `route-story-3d` extra (Playwright) and its pinned Chromium under `app\browsers`, and `framegrade.cmd` points `PLAYWRIGHT_BROWSERS_PATH` there, so Route Story 3-D runs offline. `build_release.py --skip-browser` leaves it out.

### Fixed

- An encode's moov atom rewrite (`+faststart`) now gets a stall window sized to the output, at a 20 MB/s floor (at least 1800s), instead of a flat 1800s. ffmpeg prints nothing while it shifts the whole file, and a 118 GB day master on `/mnt/c` was killed ~30 min into a healthy rewrite at ~52 MB/s, losing an 11-hour render. The window for that file is now 5900s.
- The Windows installer proof bounds every wait: the install gets 20 minutes, the Defender scan 20, each other command 5-10. A step that runs out prints what is still running as the proof user (with its command line) and the tail of the installer's log, instead of hanging the job until its 90-minute limit with no output, as the first dry run did. The proof runs the installer, the console and each command as the standard user through a scheduled task: started with `Start-Process -Credential` on a hosted runner, the installer got no desktop and exited 1 without writing its log. The helper no longer names a parameter `$Args`, PowerShell's automatic variable.
- The Windows installer proof grants its standard user the "log on as a batch job" right through the LSA API and reads it back by SID, then revokes it before removing the user. The second dry run stopped before installing because an exported security template names a local account rather than its SID, so the grant could not be confirmed from it.

### Changed

- CI runs the test suite on every logical CPU of the hosted runner (`-n logical`, 4 workers) instead of `-n auto`, which counted physical cores and ran 2 workers (the suite took ~20 min). Local runs keep `-n auto`.
- CI splits the test suite across four hosted runners (`FRAMEGRADE_TEST_SHARD=i/4`; each test lands in exactly one shard by a CRC32 of its node id). The required `test` check now waits for every shard, combines their coverage and holds it to the same 91% floor. Shard 1 also runs the secret scan, lint and type check.

## [0.3.10] - 2026-10-25

### Added

- `metrics export-evidence` emits EvidenceBundle schema 2 and carries Facebook shorts: each row has `platform`, `post_format` and a `day7` counter map (`views`, `reel_total_plays`, `reel_initial_plays`, `watch_time_seconds`, null when unreported) taken from the measured Facebook observation nearest age 7 days within 12h. Reel plays are never written into `views`. A Facebook short with no such observation is listed under `health.excluded` with `platform: facebook`. YouTube rows move `day7_views` to `day7: {views}`. Needs Waypost that accepts bundle schema 2.

### Changed

- Waypost pinned to v0.3.0 (`58168f5`, Back-Road-Creative/waypost#57, #58, #59): reads EvidenceBundle schema 2, keeps Facebook plays out of the YouTube cadence decisions and prints them as a per-platform reach line, and reports a metric nobody measured as unknown instead of 0.

## [0.3.9] - 2026-10-25

### Changed

- redactcam pinned to `2150869` (redactcam #17): when the blur-mask ffmpeg dies mid-write, the error carries ffmpeg's exit code, its stderr and how many frames were sent, instead of a bare `[Errno 32] Broken pipe` (the Ace Basin cabin repair, 2026-10-25). Masks and verdicts are unchanged.

## [0.3.8] - 2026-10-25

### Added

- `health backlog` launches a row that names a whole-day GPX with `--day-gpx` and `--camera-utc-offset-hours` from the row, so footage with no GPS of its own (Insta360) launches from the backlog; a row whose track is missing is skipped with that reason. Needs Waypost with the backlog `day_gpx` fields (Back-Road-Creative/waypost#46).

### Fixed

- A day's blur mask survives a rebuilt raw concat: when the episodes it was built from are unchanged and the raw is the same size, the finished detection is reused instead of re-run (a rebuilt 132 GB raw restarted a ~16 h pass). Masks made by earlier versions still match while their raw is untouched.
- A long-form's "Heading" line is worked out from the first fix of its first episode to the last fix of its last episode, not from one episode alone. An out-and-back drive that ends where it started now gets no heading instead of the direction of its return leg (Beaufort, 2026-02-20, said "Northeastbound"). The publish pass recomputes it from the episode tracks, so a drive encoded on an earlier version is corrected when it is published.
- The chapter list no longer opens with "Start - Beaufort County, South Carolina" followed by "Entering Beaufort County": an "Entering" chapter that names the start place is dropped. The chapter minimum still holds, because padding runs after the merge.

## [0.3.7] - 2026-10-24

### Added

- Preflight prints a non-blocking note beside the plan's ETA when the machine has fewer than 4 logical CPU cores or less than 8 GB total RAM (a machine sold as 8 GB is not flagged). The run is never refused for it.
- `gomoveshift --help` ends with the commands grouped by what they do, and its usage line no longer lists every command.

### Fixed

- A raw stream-copy concat (the episode ingest concat and the day-master raw concat) retries once when ffmpeg fails with `Cannot allocate memory` (`error code: -12`), the transient write failure a WSL `/mnt/c` mount returns on a ~25 GB file with RAM free. The partial output is removed first and the retry is logged at WARNING; any other failure, or a second ENOMEM, still fails at once.

### Changed

- `--publish` is checked when the command line is parsed: it takes a weekday (`mon`..`sun`, any case, or the full name) and anything else, such as a date, exits 2 with the accepted values instead of failing minutes into a run.
- `gomoveshift --version` appends the short commit when it is known, for example `gomoveshift 0.3.4 (abc1234)`.

## [0.3.6] - 2026-10-24

### Fixed

- A failed day-master encode keeps the raw concat instead of deleting it. The blur mask and the cabin verdict are keyed on its size and mtime, so the resume now reuses all three instead of re-concatenating and redoing blur detection and the cabin check (hours on a 4K day).

## [0.3.5] - 2026-10-24

### Changed

- redactcam pinned to `0ddd0c1` (redactcam #16): the blur-mask length check counts FFV1 packets instead of decoding every frame (52 min → seconds on a full 4K day), and the cabin verifier decodes only the frames it samples. Counts and verdicts are unchanged.

## [0.3.4] - 2026-10-23

### Fixed

- The import-time seed cadence (`schedule_constants`) reads the `gomoveshift` brand template through the same resolver as every other brand read, so a `GMS_BRANDS_DIR` template supplies the seed; it used to read the packaged file only.

## [0.3.3] - 2026-10-22

### Fixed

- The day-master disk check requires only what is still to be written: the raw concat's missing bytes plus a master up to the source's size (it was 1.5× the source, which counted an on-disk raw twice and under-sized the master), so a crashed day on a tight volume can resume.

## [0.3.2] - 2026-10-22

### Added

- `GMS_BRANDS_DIR`: a host brand-templates directory. `<GMS_BRANDS_DIR>/<slug>/` replaces the packaged template for that slug (whole directory, no per-file merge) for every `brand.yaml` / `settings.yaml` read, and `known_brands()` lists host-only slugs.

### Fixed

- Day-master privacy-blur detection checkpoints every 5 minutes of video, so a GPU crash mid-pass resumes instead of restarting a multi-hour detection; a checkpoint made under different evidence or detector settings is discarded.
- The resume recipe and the provenance sidecars now record the product's own commit: HEAD of a source checkout, or the commit `pip` recorded for a git-ref install. An installed package inside an unrelated work tree no longer records that tree's HEAD (it records null). `health check`/`heal` keep `--code-root` as the operator's directory.

## [0.3.1] - 2026-10-22

### Fixed

- A resume recipe now records `GMS_INTERMEDIATE_DIR`, and `health heal` relaunches with it, so a resumed drive stages where its saved state is instead of the home cache.

## [0.3.0] - 2026-10-21

### Added

- Console: GPU jobs (`drive`, `route-story-film`) run on a host worker through a
  file spool when `GMS_HOST_SPOOL_DIR` is set; the console queues, shows
  progress, cancels, refuses duplicates and answers 409 when the worker is not
  alive. The worker applies a request's registry-named `env` overrides and stops
  a job's process group on SIGTERM/SIGINT.
- Console: backend runs it did not launch (CLI or cron drives) list under
  `/drives/gates/pending` with their origin; `gomoveshift drive` and
  `gomoveshift pipeline` write `run_origin.json` (`GMS_RUN_ORIGIN`).
- Console: read-only `/publishing/reconcile` view, a `/schedule` read model that
  shows target, capacity, feasible output and scheduled Shorts separately, and
  `docs/console-inventory.md` with a test that fails when a route has no row.
- `--ost-system-review` (default off) lets the long-form, Shorts and superlong
  burn gates pass on a sealed per-surface route-check record instead of a person.
- Route claims: pinned Census county/state boundary proofs for retained-route
  labels; unresolved samples with a proof are reported as `samples_unresolved`.
- `gomoveshift health backlog --region R` launches the next backlog day as an
  audit-only drive, through the job runner (`$GMS_JOB_RUNNER`) under the lease
  script named by `$GMS_LEASE_SCRIPT` (default `bin/lease.sh`).

- `<GMS_INPUTS_DIR>/known_ids.txt` lists the channel's private and unlisted
  video ids that no uploads listing returns, one per line (`#` comments,
  repeats dropped). `site_sync/known_ids.py` `load_known_ids()` reads it;
  `build_channel_manifest` and `library-hygiene`'s `list_uploads` default to
  it instead of the package seed `FROZEN_KNOWN_IDS`, which stays empty. An
  absent file means no extra ids.
- `GMS_INPUTS_DIR` names the operator's own inputs directory
  (`paths.gms_inputs_dir()`), default `operator-inputs/` under the data home.
  The console reads Route Story launch profiles from its `route_story_profiles/`
  subdirectory, ahead of the existing locations. An empty value counts as unset;
  a value that is not an existing directory raises `InputsDirError` (a
  `ValueError`) naming the variable and the path, and the console answers 422 on
  `GET /route-stories/profiles` and 400 on a film launch with that message.
- CI runs a real clip through the grading path, from frame sampling through the
  LUT-baked libx265 encode, and checks the encoded pixels
  (`tests/test_grading_path_real_clip.py`). A missing `ffmpeg` or `libx265`
  fails it in CI instead of skipping it.

### Changed

- Zero-tolerance place gate extended to a Short's title, description and tags: each Short's measured route record is persisted (audit sidecar, plan) and checked before upload; a GPS-bearing Short with no persisted bins is refused as unverifiable.
- Waypost (`seo-manager`) and `gps-journey` pins move to their first release tags' commits: Waypost `v0.1.0` (`f879603d0275`, adds the `SEO_INPUTS_DIR` inputs layer) and gps-journey `v0.2.0` (`10f9f96ae2fd`, adds `GPS_INPUTS_DIR`), in `pyproject.toml` and `ci/siblings.txt`. Docs: `docs/pipeline/82-waypost-region-pin.md`.
- Zero-tolerance place gate: a place named in an LF title, OST, description or
  tags must hold every retained second; `route_claims.recheck_place_text`
  re-checks a persisted record offline.
- Long-form titles are approved against retained route evidence before encoding,
  and the decision is reused by metadata and upload checks.
- Locked pillow 13.0.0 and certifi 2026.10.19.

- Default locations are per-user instead of host-specific. Inputs, outputs,
  photo state and exports default to `input/`, `output/` and `photos/` under
  the data home (`GMS_DATA_HOME`, else `$XDG_DATA_HOME/gomoveshift`, else
  `~/.local/share/gomoveshift`). Caches and lock files live under the cache
  home (`GMS_CACHE_HOME`, else `$XDG_CACHE_HOME/gomoveshift`, else
  `~/.cache/gomoveshift`). Credential, token, key and SOPS-config files live
  under the config home (`GMS_CONFIG_HOME`, else `$XDG_CONFIG_HOME/gomoveshift`,
  else `~/.config/gomoveshift`). Every existing `GMS_*` variable keeps its name
  and still overrides its default; set the ones you relied on explicitly.
- Features that depend on a separate checkout or service (a site checkout, the
  SEO data directory, an intel database) have no default location. Unset means
  "not configured", and the message names the variable to set.
- Console container packaging: host paths in `docker/console-compose.yaml` are
  required variables with no defaults, and `GMS_SITE_HOST_PATH`,
  `GMS_NIGHT_HOST_ROOT` and `GMS_VID_OUTPUT_DIR` take effect only with their
  overlay file (`docker/console-site.yaml`, `docker/console-extra-output.yaml`).
  `docker/console-backup.sh` names extra volumes through
  `CONSOLE_OPTIONAL_VOLUME_NAMES`. The `GMS_HOST_GID` default is now `1000`; a
  deployment whose mounted directories are owned by another gid must set
  `GMS_HOST_GID` explicitly. See `docker/README.md`.
- The Waypost showcase bundle is stamped with the nearest tag of the checkout
  (`git describe --tags`).

### Removed

- Monorepo-specific helper scripts, the multi-tenant console overlay and its
  reprovisioning script, and internal governance documents.

### Fixed

- A resumed drive now rebuilds a `drive_plan.json` written by older phase B+C code instead of reusing it, so planner fixes such as the GPS-gap mask apply to drives that were already planned.
- Viewer-facing place names go through Waypost's locality rule (`seo_manager.core.locality`): a tract/vicinity name such as Longwood is dropped from `towns_traversed`, the per-LF place sequence (Route line, per-LF arc) and the description, an island named in its own geocode address stands in (else it is just dropped), and the "Driving through" sentence names each place once.
- Superlong parts are route-checked over their own retained ranges.
- Drive-text chapters bind to the first evidenced place in each title.
- Phase D no longer crashes on a claim id with no matching evidence place.
- `full_drive` planning and the Shorts planner cut GPS gaps with the route-claim
  gap rule, so planned clips are not refused later.
- Console alerts stay pending until delivered and retry with bounded backoff;
  duplicate schedules are refused; delivery state reads the same on every page.
- The review gate's placeholder check scans metadata values only.
- `queue reslot-overdue-manual` dry run reserves each proposed slot.
- `health heal` honours a `no-auto-resume` file beside a drive's `resume.json`.

- `docs/pipeline/96-packaging.md` linked the install guide as `install.md`,
  which resolves to `docs/pipeline/install.md` and 404s; it now links
  `../install.md`. `tests/test_docs_relative_links.py` fails any relative
  Markdown link in the repo root or `docs/` that does not resolve.

[Unreleased]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.15...HEAD
[0.3.15]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.14...v0.3.15
[0.3.14]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.13...v0.3.14
[0.3.13]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.12...v0.3.13
[0.3.12]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.11...v0.3.12
[0.3.11]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.10...v0.3.11
[0.3.10]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.9...v0.3.10
[0.3.9]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Back-Road-Creative/framegrade/releases/tag/v0.3.0
