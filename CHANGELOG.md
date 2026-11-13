# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.83] - 2026-11-13

### Added

- `gomoveshift queue edit-facebook-caption --id N [--text "..."] [--apply]` rewrites the description of a published Facebook Reel (`FacebookReelsAdapter.update_description`), reads it back, and records it in the publish-plan sidecar. The caption is built from the Short's own measured county (`sampled_regions`) and recording month; typed `--text` is refused for such a Short, and for a Short with no measured county it must claim nothing (`placeless_label_claims_nothing`). Dry run by default; never changes the queue row or deletes the post.

### Fixed

- `--placeless-label` no longer overwrites the measured label of a long-form that has GPS; it labels only the GPS-less ones, and one warning names the long-forms where it was withheld.
- The unattended backlog launcher skips every row carrying `placeless_label` (claim-free or not) and tells the operator to give `bracket_gpx` instead; `gomoveshift health check` rule 5 reports each such row. A placeless label stays a hand-run last resort.

## [0.3.82] - 2026-11-13

### Added

- `gomoveshift queue remeasure-short-route --id N --stages-dir <run>/_stages` re-derives the
  route `bins` a Short planned before the place gate lacks, from the day's `gps_points_<date>.json`
  and the clip range in `shorts_ost_audit_<date>.json`, and writes it to the sidecar's
  `route_record`. Dry run unless `--apply`; refuses (writing nothing) when a source is gone, the
  clip does not match the Short's media, or the clip holds no GPS fix. It never writes to YouTube
  and never touches the place gate; `recover-terminal`'s gates then decide.

### Changed

- The place gate's "route record has no 'bins'" refusal now names `queue remeasure-short-route`.

### Fixed

- The per-day master drop's shorts integrity check compared against a contiguous `short_01..short_N` range, so a day whose shorts were dropped and refilled (gapped file numbers) reported "missing 5 short(s) (expected 21, found 21)" and kept a ~59 GB master. It now checks the day's own planned short file names; a genuinely absent planned short still fails closed.
- A GPS-less drive's shorts no longer fail the `places_verified` gate (check 20) and the drive-text review no longer refuses a bracket label. A short records `place_source` `bracket_route` only when its text is exactly the label of a `--bracket-gpx` record within its gap cap, and `placeless_label` only when the text is a claim-free placeless label. The gate re-checks that evidence instead of trusting the string; `lf_label` and a short with clip regions stay unverified, and any other unbacked name is still refused.
- `gomoveshift queue sync-plan-tags` now also rewrites the sidecar tags of a `failed_permanent` YouTube row the final-text review refused before any upload (no upload id), so `queue recover-terminal` can re-queue it. Dry run stays the default, nothing is sent to YouTube, and the Short tag rule is unchanged.

## [0.3.81] - 2026-11-12

### Added

- `gomoveshift queue reslot-overdue-youtube` moves a `queued` YouTube Short whose slot has
  passed to the region's next free Waypost shorts slot (the allocator `reslot-overdue-manual`
  uses), in the queue row and in the sidecar's `publishAt`, never to a time already past and
  never onto a slot another YouTube row holds. Dry run unless `--apply`.

### Fixed

- A YouTube Short no longer fails the pre-upload final-text review on its own
  generated title. The stamp ("Feb 24, 2nd stretch"), a weekday, "seconds" and
  the closing hashtags are now backed by the record (`recording_date`, the
  generator's tag lists, the clip's own region votes); any other word is still
  refused. The plan builder fills `reviewed_plan_hash` with the content hash of
  what the Short sends, and the review uses the same hash for plans written
  without one (`missing_field:approval_hash`).
- `gomoveshift queue recover-terminal` re-queues a YouTube `failed_permanent` row
  the final-text review refused (it never uploaded), once the review agrees with
  the Short as the sidecar would send it now, behind the same duplicate-scan,
  slot and privacy gates.
- `gomoveshift queue recover-terminal` no longer refuses a final-text-review `failed_permanent`
  YouTube row for a passed slot: it assigns the next free Waypost slot through the same
  allocator. Every other gate, and gate (f) for every other row kind, is unchanged.

## [0.3.80] - 2026-11-12

### Fixed

- `gomoveshift cleanup <project> --abandoned` now reclaims a run that `health backlog` gave up. Its `no-auto-resume` hold was read as an operator's "waiting" hold, and the age gate refused it too, so the 57 GB of a given-up drive stayed on the stage volume while the next drive failed the work-space gate 27.6 GB short. An operator's own hold still refuses.
- The LF work-space gate's "free space" hint now names the stage root it reports (`GMS_INTERMEDIATE_DIR=<root> gomoveshift cleanup …`) and the `--abandoned` form for a killed or given-up run. The bare `gomoveshift cleanup <project>` it printed searched the default stage cache, so a drive staged by `health backlog` under another root answered `Project not found`.

## [0.3.79] - 2026-11-12

### Fixed

- Every Facebook/Instagram Graph request now carries a `(10, 120)` s timeout unless its caller sets one; a dead connection hung `dispatch-queue --apply` for 21 h.

## [0.3.78] - 2026-11-12

### Fixed

- A GPS-less drive whose bracket route only proves the country no longer fails
  at the long-form burn. The OST-burn contract check refused any per-LF label
  equal to the drive-wide `--location`, so a bracket label of "United States"
  on a `--region usa` drive stopped the day after its master encode. The check
  now lets through an LF whose `per_lf_tier_evidence` records the label as the
  ladder's own award (bracket route or placeless operator label), and still
  refuses every other LF.

## [0.3.77] - 2026-11-11

### Changed

- redactcam pinned to `1937312`: each sampled frame now letterboxes its detection
  tiles once for the face and plate models, and runs the COCO model once for the
  vehicle and person detectors (both specs resolve to the same model, sha
  `5ad2aa77f7f6`). Detected boxes are byte-identical. Measured on a 20 s 4K clip
  under load: 4.82/3.64 fps before, 4.94/3.95 fps after.

## [0.3.76] - 2026-11-11

### Fixed

- Confirming an approved Page photo no longer opens or updates driving-short
  tracking records. Its platform identity remains in the publication queue.

## [0.3.75] - 2026-11-11

### Added

- Add an unwired single-Page-photo delivery consumer with a required trusted
  authorization port, atomic write-ahead attempts and retained-ID reconciliation.
  Uncertain writes remain held across crashes, retry exhaustion and ordinary
  cancellation/recovery; exact captions, bytes and Page identity stay bound to
  the approved item. Production publishing remains disabled pending the real
  approval consumer and explicit route activation.
- Add an explicitly injected general-content Reel consumer with exact approved
  bytes/caption/account binding, fresh authorization before finish, durable
  attempts and read-only recovery after uncertain writes. Production wiring
  remains separate.

## [0.3.74] - 2026-11-10

### Added

- Add an unwired Facebook Page-photo byte client with explicit publication intent,
  immutable JPEG hash binding and exact Photo/story/Page readback. Unknown writes
  are reported without retrying; readback failures retain known IDs. Public
  requests are tested offline only; production approval and durable queue wiring
  remain separate prerequisites.

### Fixed

- Checkpoint Facebook Reels finish before sending it, require literal confirmed
  success, and keep ambiguous results for read-only status reconciliation of the
  retained video ID. Queue recovery preserves unknown outcomes with backoff beyond
  normal retry limits instead of falsely declaring publication or permanent failure.
- Bind new uploads to their original Facebook Page before sending them. Legacy
  unknown accounts and changed credentials remain held without rewriting provenance.
- Facebook queued publishing now refuses unsupported drafts and incompatible
  state/time requests before acquiring an upload lease or contacting Meta.
  Future slots within the minimum scheduling lead wait until due instead of
  publishing early; at/after-slot drains keep their existing behavior.

## [0.3.73] - 2026-11-10

### Fixed

- `cleanup <project> --abandoned` now reclaims a run that `health backlog` superseded even when it is younger than `--min-age`. The supersede record (`superseded.json`) already says that run is dead, so the 7-day gate only held its space while the live run waited. The lock, hold and running-drive checks still apply.
- Facebook photo and carousel queue entries no longer use the Instagram media
  client or fall through to the Facebook Reels uploader. Unsupported Facebook
  photo delivery fails before reading media or calling an upload adapter;
  Instagram photo and Facebook video routes remain available.
- `photos sort` groups an intervalometer run outside the night sky, such as a dawn sequence, as one `sequence-<n>` with its own time-lapse. Before, look-alike bursts split it into several groups and loose shots. A run needs 24 or more frames of one category, at most 60 s apart, at a steady cadence.
- The upfront staging estimate (preflight and the hourly backlog) sized a drive at 1.5x its
  source while the day-master encode asks for the raw concat plus the master on top of the
  episode concats it keeps, 3x in all. A drive could launch, spend hours on scoring, then stop
  at the encode for lack of space. Both now read one `PEAK_CACHE_FACTOR` (3.0), so a drive that
  cannot fit is refused before it starts.

## [0.3.72] - 2026-11-10

### Fixed

- `photos sort` no longer files a dusk sky run as landscape. Without a solar track, "daylight" came from the clock alone (05:00-20:59), so 111 frames shot at 20:26-20:59 at 15 s / ISO 3200 / f/3.8 (EV100 -5.1, the night frames' own exposure) were far-focus "landscape". Subject recognition now reads the exposure: EV100 at or below -2 is night evidence, and daylight needs EV100 5 or more.
- `photos sort` no longer ranks lens-cap darks as night-sky shots. The night frames go through the stack's light/dark triage once; the darks move to a `lens-cap-darks/` folder, join no group, and show on the page as a count only. Darks shot within 15 minutes of a night session still calibrate its star trail and time-lapse.

## [0.3.71] - 2026-11-10

### Added

- `health backlog` publishes its region's audited drive: when the `auditing` drive's newest audit succeeded, the hourly tick runs `health publish` for it. A slot still outside Waypost's lead window is a noop ("audit passed: ... waits for its Waypost slot"), retried next tick; any other publish refusal is a refusal. Before, an audited drive held its region's slot ("audit in progress") until a person ran `health publish` inside the window.
- `photos sort` now makes one star-trail picture for every night session of 24 or more usable frames, into `trails/`, shown on the session's card between the stacks and the time-lapse. The session's frames are sorted into lights and lens-cap darks once for both outputs, and a resumed run builds only the trails that are missing.

### Fixed

- `gomoveshift cleanup` finds a drive `health backlog` staged (`<stages root>/<project>/<project>/_stages`), named or discovered. Before, only `<base>/<project>/_stages` was looked up, so cleaning a published backlog drive with `GMS_INTERMEDIATE_DIR=<stages root>` skipped it ("pipeline not complete") and freed nothing.

## [0.3.70] - 2026-11-10

### Added

- `photos sort` now makes a time-lapse video (with a poster picture) for every night session and for every burst of 24 or more look-alike shots, into `timelapse/`. Before, a night with hundreds of frames gave one stack per 19 frames and no video. Lens-cap dark frames are never used as lights, and a resumed run builds only the time-lapses that are missing.

### Changed

- The `photos sort` index page now shows one card per sequence (a night session, a long burst) instead of one thumbnail per frame. The card holds the stack or time-lapse, the frame count, the time span and the best frame, with the individual frames folded away. Each section shows its top 6 single shots, the page has a summary strip, and it uses a dark layout.

### Fixed

- `health backlog` reads a `health heal` resume of a backlog drive (`gms-heal-gms-backlog-<project>-<sha7>`) as that drive's newest audit run. Before, only the original `gms-backlog-<project>` job counted, so a drive that heal resumed to success still refused its region every tick ("audit failed", exit 2), naming the old failed run.
- `photos sort` filed daylight shots focused far away (15 m or more) in `other` when they were shot wide open or with a long lens. They now go in `landscape`, as a weak guess from the camera data alone.

## [0.3.69] - 2026-11-10

### Changed

- Waypost is pinned to its 0.6.0 release commit. The previous pin (the `bracket_gpx` commit) still carried version 0.5.0, so an existing install kept the old 0.5.0 code and the backlog refused `bracket_gpx` rows.

## [0.3.68] - 2026-11-10

### Fixed

- The final-text review backs a stateless county's country ("County Cork, Ireland") with the same measured region claim, so an Irish long-form title is no longer refused as `name_unbacked:title:ireland`.

## [0.3.67] - 2026-11-10

### Fixed

- Night stacks now tell real dark-sky frames from lens-cap-on darks. On the 2026-11-09 card the cap-on test had it backwards: every real frame of seven night groups was refused as "lens cap on" ("0 usable of 19"), while the end-of-run darks shot in the same sequences passed as usable. A frame at the sensor's black floor is now judged by how many star-sized bright spots survive an erosion that removes hot pixels: 20-383 on the card's real frames, 0-5 on its darks.
- Night stacks on sparse star fields now keep frames the old fit threw away. On the 2026-11-09 card, night groups with about 90 stars a frame stacked only 3-4 of their 15-19 good frames: the old 4-number fit (shift, turn, scale) could not follow the lens bending stars near the corners as the sky drifted, so the outer-corner check refused it. Those frames are now fitted with the lens-aware model chunk combining already uses, against the same checks, which are not loosened.

## [0.3.66] - 2026-11-10

### Fixed

- `photos sort` panoramas no longer fail on RAW cards with "could not be read as an image":
  the stitcher is now handed the developed JPEGs the run wrote instead of the camera RAWs,
  which `cv2.imread` cannot open.
- A Short whose last bin holds no fix of its own (the next fix lands just past
  the cut, inside `GAP_S`) no longer refuses its county label: a county-less
  bin with no county-bearing bin on one side takes the nearest source fix
  outside the cut on that side, the same stand-in a head or tail gap already
  uses, and only for a stateless county in one country. Short windows now
  carry those outside fixes (`short_route_window`).

## [0.3.65] - 2026-11-10

### Fixed

- A route-claim reverse lookup that fails (a timeout, a dropped connection) is retried after
  5, 15 and 45 s before its sample counts as unresolved. One timed-out lookup during a network
  blip had refused a Short label the GPS supports and failed the whole day after its encode.
  A lookup that keeps failing is still unresolved, and a lookup that answers with nothing is
  not retried.

## [0.3.64] - 2026-11-09

### Fixed

- `photos sort` no longer hangs with nothing developed: its develop workers are now spawned
  instead of forked, so they do not inherit the OpenMP lock state rawpy left in the parent
  while measuring corrections.

## [0.3.63] - 2026-11-09

### Added

- `gomoveshift.pipeline.photos.aesthetic`: a local, CPU-by-default aesthetic scorer for judging the most
  striking photos. CLIP ViT-L/14 (int8 ONNX) plus the LAION aesthetic head, both pinned by size and sha256 and
  downloaded only by an explicit `fetch_models()`; the head is read without torch through a restricted
  unpickler. `AestheticScorer.score` returns the raw ~1-10 value, `score_unit` a 0-1 clamp.
- Photo subject recognition gains `close_up`: a frame focused within 3 m (about 10 ft), or shot on a macro lens or at 0.25x or more implied magnification, reads as a close-up (0.65, 0.75 with a macro signal). It ranks after the astro rules and before wildlife, so a long lens focused near is a close-up, not wildlife. `CaptureOptics` carries `focus_distance_m` and `lens_model`, and `recognise_photos` accepts an `optics` mapping to supply them. Operator subject sidecars accept `close_up`; look routing is unchanged.
- `gomoveshift/pipeline/photos/quality_order.py`: a per-category quality score for
  ordering shots inside a category (never for excluding one). It scores the developed
  image: sharpness and clipping, star count, star sharpness and trailing for a night
  sky, and a category signal (landscape colour and sky, night-sky darkness and
  foreground separation, close-up subject isolation). An optional aesthetic score
  blends in at weight 0.5. The result carries every named component for `ranked.csv`.
- `gomoveshift photos sort <folder>`: dedupe, classify and develop a whole folder of photos into the fixed `photos.sort_root` (default `/mnt/d/camPics_output`). Renamed or re-offloaded copies and shots seen in an earlier run are listed in `duplicates.csv` instead of being processed again (`--reprocess` overrides); each remaining photo is developed into a category folder (`night-sky`, `landscape`, `close-up`, `wildlife`, `portrait`, `long-exposure`, `hdr-source`, `other`) in capture order, with `run.json`, `ranked.csv`, `skipped.csv` and a static `index.html`. `--dry-run` prints counts and a time estimate and writes nothing; `--index-only` seeds the ledger from folders already sorted by hand. Refuses (exit 2, nothing decoded) when the folder is missing, the output drive is not mounted, or the two paths nest. Takes the shared media admission fence and runs at low priority.
- Photo duplicate detection building blocks (not yet wired to a command): an exiftool capture reader (body serial, shutter count, focus distance, lens; Pillow fallback with a warning when exiftool is missing) and a dedupe module that collapses renamed or cross-folder copies of one shot by capture identity (RAW kept over JPEG), hashes only frames without an identity, and remembers processed shots in a sqlite ledger across runs.
- `photos sort` tags scene groups in `ranked.csv` (`group_id`, `group_best_rank`): night
  sessions, exposure brackets, panorama runs (new detector: same settings, at most 8 s
  apart, three or more frames) and look-alike bursts inside a category. It then builds
  `hdr/`, `pano/` and `stacks/` outputs one at a time, best night session first, rewrites
  `index.html` as each lands, and skips night stacks with `--no-stacks`. A failed group is a
  `skipped.csv` row. `run_sort(resume=...)` finishes only the unfinished groups.
- `photos sort --resume <run folder>` finishes an earlier run's unfinished stacks, HDR
  fuses and panoramas without re-developing or re-scoring.
- `gomoveshift photos sort` now scores every developed shot and orders each category by quality: technical
  sharpness and clipping (plus stars for a night sky), a per-category signal, and the local aesthetic model.
  Files are renamed `NNN-<stem>.jpg` with NNN the quality rank (1 = best); `ranked.csv` carries every
  component, and `index.html` shows each section in rank order with the total under each thumbnail. The
  aesthetic pass runs once per run on the CPU (2 threads), caches by source sha256 in the ledger, and
  never takes the shared GPU lock, so it cannot hold up a GoMoveShift render.
- `gomoveshift photos fetch-models`: the explicit, hash-checked download of the aesthetic weights.

### Changed

- The bracket route (`--bracket-gpx`) caps the gap between its two fixes per level (`BRACKET_MAX_GAP_H_BY_LEVEL`): a county or state still needs fixes at most 36 h apart, a country at most 48 h. A gap past a level's cap drops the label to the next broader level both fixes agree on, so two day tracks a day apart (≈ 44 h) inside one county name the country instead of being refused. Over 48 h is still refused.
- Pinned Waypost (`seo-manager`) to `d74f50a` (Back-Road-Creative/waypost#109). A drive backlog row may carry `bracket_gpx: [before, after]`, which the backlog launcher hands to `--bracket-gpx` (#269); before this pin the loader refused the key.
- `photos sort` drops to the lowest CPU priority (`nice 19`, was 10) so it never slows a
  GoMoveShift render running on the same machine.
- `photos sort` no longer has a `--gpu` flag: the aesthetic pass always runs on the CPU and the
  sort never takes the shared GPU lock, which the GoMoveShift drives hold.

## [0.3.62] - 2026-11-09

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

## [0.3.61] - 2026-11-09

### Changed

- redactcam pinned to `ce6a1c2`: stage-1 detection decodes the day master through
  ffmpeg NVDEC (`-hwaccel cuda`, `hevc_cuvid`/`h264_cuvid`/`av1_cuvid`) piped as
  bgr24, falling back to `cv2.VideoCapture` when ffmpeg, the cuda hwaccel or the
  cuvid decoder is missing, the probe fails, the clip is rotated, or the run resumes
  from a checkpoint (`start_frame > 0`). Measured on a 4K60 HEVC master: 18.1 fps
  cv2 vs 31.8 fps NVDEC. `REDACTCAM_DECODE=cv2|nvdec|auto` overrides.
  `tests/test_redactcam_parity.py` pins the new `decode="auto"` default.

## [0.3.60] - 2026-11-09

### Fixed

- A placeless label may say nothing the footage cannot prove, so `--placeless-label` now refuses any word outside `PLACELESS_NO_CLAIM_WORDS` (scenery, weather, time of day, road kind) at argparse, naming the claiming words, instead of burning it into the OST for a person to catch at the audit gate. A `drive_backlog.yaml` row carrying such a label was skipped silently; `gomoveshift health check` now opens a `placeless_label_unverifiable` incident (row, label, claiming words) for it, open while the yaml carries the label and closed once it does not. The other rows still launch.
- A corrected `--placeless-label` now replaces the old text on resume. The new label was applied to `state.placeless_label` only, while the checkpoint brought back each LF's old label in `per_lf_locations`, `per_lf_ost_text` and `per_lf_titles` and a Phase-D-only resume reused finals already burned with it (a claiming label such as "Through The Valleys And Ridges" survived its correction to "Somewhere On The Road"). A resume whose label differs from the text any LF carries now rewrites every LF's label, drops `phase_bc_day_first` and `phase_d` so the LFs re-burn and the thumbnails and titles re-render, clears `per_day_completed` and the OST audit confirmations (state pass and hash, system-review records, `ost_audit_<date>.json` and `*.confirmed`), so a confirmation of the old text cannot carry over. An unchanged label is untouched.

## [0.3.59] - 2026-11-09

### Added

- `health backlog` reads a row's `per_lf_locations` (1-based LF index to label) and `location` (Waypost 0.5.x `BacklogRow`) and hands them to the drive as `--per-lf-location <i>=<label>` / `--location <label>`, so a GPS-less recording with no day track (`nctopa-2026-03-05-rec0063`) launches from the backlog with the operator's labels typed once into the row instead of a hand launch. The row is checked against the drive's own label rules before any render (`location_labels_skip_reason`): a label the parser would refuse skips the row at the backlog with the parser's reason, and the row behind it still launches. The drive still waits for a person at the OST audit gate.

## [0.3.58] - 2026-11-09

### Fixed

- A stateless-county long-form label with its country tail ("County Cork, Ireland", the label the pipeline itself builds) now gets region facts at the pre-render title gate instead of refusing the encode with `facts_unavailable`; `region_label` appends the country for a single Irish county as it already did for a pair.

## [0.3.57] - 2026-11-09

### Added

- `gomoveshift cleanup <project> --abandoned` reclaims a killed or superseded run's
  whole `_stages/` dir, which the plain `cleanup` could only skip as "Pipeline not
  complete". The dir qualifies only when every rule holds — not a symlink, no
  `no-auto-resume` hold, no run lock held by a live pid, no `gomoveshift drive`
  running anywhere, old enough for `--min-age` — and the first failing rule is the
  `SKIP` reason. Dry run by default; `--confirm` re-checks and removes that one dir.
  A project name is required. Without the flag the incomplete-project `SKIP` line
  now names `--abandoned`.
- `gomoveshift drive --from-recipe <path-or-project>` relaunches a drive from its own
  `<stages>/resume.json`: the recipe's argv and `GMS_INTERMEDIATE_DIR` replace the command
  line before parsing, so the relaunch reaches the stage dir the run used instead of a fresh
  one. A bare project name resolves to the one live recipe under the stages root (none or
  several refuse). Only `--phase`, `--publish`, `--lf-day`, `--lf-publish-date`,
  `--skip-upload`, `--no-youtube-upload`, `--force` and `--no-tmux` may be added; `--fresh`, a
  directory, any other flag, a conflicting `GMS_INTERMEDIATE_DIR` in the environment or an
  unreadable recipe refuses with exit 2.
- Guide/destination signs ("EXIT 283 Woodstock 1 MILE") are now an advisory `destination` sign claim with the stated distance. The verifier forward-geocodes the place (with an N/W initial variant for OCR misreads) within `sign_verify_destination_max_km` (default 80) of where it was read and may corroborate; it never conflicts and never feeds a label.
- `--label-dry-run` suggests a sign-derived label for an LF with no GPS county vote: beneath `<no GPS — --per-lf-location override required>` it prints the county label `sign_label.explain_sign_label` allows from the LF's own road signs, with a copyable `--per-lf-location 'INDEX=LABEL'` form, or every rule that refused one (`none (N sign hit(s))`; `unavailable (<error>)` when bucketing failed). Suggest only: nothing is written into the override map or the state, and a GPS-bearing LF keeps its `sign evidence` verifier block instead.
- The Phase A sign pass logs progress every 50 sampled seconds per day (done/total, hits, elapsed, ETA) and counts the stage each sample reached (sampled / grabbed / region / ocr_text / claims); the counters ride on the day summary and the A6 line and persist as `state.sign_pass_stats`, so a day's `0 hit(s)` can be read as "0 frames decoded" or "0 sign regions" without re-reading the footage.

## [0.3.56] - 2026-11-08

### Added

- `--day-gpx` now covers GoPro footage. A GoPro whose GPMF never locked (`nctopa-2026-03-05-rec0063`: 13 chapters, zero fixes) had no GPS at all and could only be labeled by hand; the day-GPX slicer now cuts a phone track into one sidecar per `GXccNNNN.MP4` chapter, starting a recording at its chapter 01 `creation_time` (UTC, so `--camera-utc-offset-hours` is refused) and each later chapter after the earlier chapters' durations. GoPro ingest reads those sidecars for an episode whose own GPMF does not cover every chapter, merged into one track and tagged `external` so Phase A runs the GPX-video correspondence gate; an episode with GPMF on every chapter keeps its GPMF. Sidecars beside only some chapters, a missing chapter 01 or `creation_time`, and a directory mixing GoPro and `VID_` names are refused.

### Fixed

- A day-GPX-sliced track is judged at a 30 s fix gap (`route_claims.DAY_TRACK_GAP_S`, one evidence bin), not the 5 s GPMF rule. The slicer's sidecar is a phone logger's track with holes of a few seconds while the phone sleeps; judged at 5 s, `ireland-2025-09-17-baltimore-clonakilty` lost 1085 s to the GPS-gap cut and fell under the 3600 s part floor on footage with GPS throughout (longest hole in the day track: 20 s). The tolerance follows the GPX `creator` (`GPSData.creator`, kept by the parser and the checkpoint; the GoPro sidecar merge now writes the slicer's marker too) and is applied on both sides of the contract: the planner mask and the Phase-A floor check (`day_fix_gap_s`: the stricter value when a day mixes GPMF and day-track episodes), and the route-claim gate (`RouteWindow.gap_s`; `gap_threshold_s` records every value used). GPMF tracks and tracks of unknown origin keep 5 s; Shorts and superlong parts keep 5 s.

## [0.3.55] - 2026-11-08

### Fixed

- The resumed-Phase-A sign back-fill now reads the footage. The checkpointed ingest plan is a shallow rehydrate whose `episodes` are the serialized dicts, so `group_episodes_by_day` dropped every one and the pass logged "0 hit(s) across 0 day(s)" in under a second without decoding a frame (seen on `nctopa-2026-03-05-rec0063`). The back-fill resolves the plan the way the day phase does, from the parallel state arrays, before the pass samples it. A pass that groups zero days is no longer recorded as evidence, and a checkpoint already holding `{}` is read again on its next resume.

## [0.3.54] - 2026-11-08

### Fixed

- A resume with Phase A checkpointed now reads a GPS-less drive's road signs. `PhaseA.run` is never re-entered once `phase_a` is in `phases_completed`, so the A6 sign pass (on by default for a GPS-less drive since the previous release) never ran for a drive parked past Phase A, and `--label-dry-run` and the pre-encode sign gate saw no evidence. The orchestrator's resumed-phase branch now calls the same pass body (`sign_pass.run_sign_pass`) from the checkpointed ingest plan, episode paths and rebuilt GPS traces, saves the evidence, and only then prints the dry-run. Footage already read is never decoded again; `--no-sign-verify` still wins.

## [0.3.53] - 2026-11-08

### Added

- `analysis.sign_label.derive_sign_label`: a county-level label ("Colleton County, SC") derived from an LF's road signs, for an LF the camera never located. It is produced only when at least `sign_verify_min_corroborating_hits` (default 2) sign hits from distinct seconds name one county, no other county is named, exactly one state is named, and the LF has zero GPS fixes; otherwise it refuses with a reason naming the rule. The label carries its evidence (hit count, seconds, sign texts) for tier evidence and the OST system review. This release only derives; a later change writes it into the per-LF labels and shows it in `--list-lfs`.

### Changed

- `ocr.parse_sign_claims`: a full state name is a state claim only beside a place claim ("Colleton County South Carolina") or a boundary cue ("Welcome to", "State Line", "Entering", "Leaving", "Welcomes You"). A destination on an interstate guide or distance sign ("Winchester / West Virginia / Exit 310") is no longer a claim about where the camera is, so it can neither mislabel a GPS-less LF nor raise a false state conflict.
- A drive with a GPS-less episode now reads its road signs by default. The sign OCR pass was opt-in (`--sign-verify`), so a drive the camera never located (`nctopa-2026-03-05-rec0063`: 13 chapters, zero locked fixes) rendered with no evidence of where it was. The operator's decision still wins: `--no-sign-verify` turns it off, `--sign-verify` turns it on for any drive. A GPS-less day is sampled every `sign_verify_gpsless_sample_period_s` (default 10 s) instead of the flat 120 per day, and EasyOCR runs on the GPU when torch sees one.

### Fixed

- `health backlog` launches a GPS-less row only when its `placeless_label` claims nothing (every word in `PLACELESS_NO_CLAIM_WORDS`, e.g. "Somewhere On The Road"). A descriptive label ("Through The Valleys And Ridges") is skipped at the backlog, before any render, with the words it claims named in the skip reason, and the row behind it still launches. `nctopa-2026-03-05-rec0063` had rendered that label over a drive that is mostly divided interstate and farmland; the label was operator-typed and nothing measured it. A label outside the placeless vocabulary is now also refused at the backlog instead of failing the launch job at argparse.
- `health backlog` now retires and holds a rowless dead drive even while a drive is running. Before this, heal's resume of that same dead drive started just before each hourly backlog run, so the backlog stopped at "a drive is running" every hour. The hold was never written, and the next mirror SHA brought the same resume again, so neither region's backlog ever launched. The text output now also prints the `retired` and `held` lines, so the cron log shows them.
- CI caches the pinned ffmpeg tarball under a key built from its checksum, and retries the download. On 11-08 the release host answered 504 and every test shard went red at "Install ffmpeg". A cached tarball is still checked against the pinned checksum.
- `health check` and `health backlog` no longer treat a drive stopped at the OST audit gate (exit 5, `DRIVE WAITING:`) as failed. No `drive_failed` row is filed, so `health heal` stops re-resuming it onto the same gate on every mirror SHA. The backlog names the sidecar to review and the `.confirmed` marker to create, then resumes the drive from its stages on the next run once the marker exists.
- `--ost-system-review` now passes a GPS-less drive run with a claim-free `--placeless-label`. Before, route claims had nothing to measure, so the long-form, Shorts and superlong gates always waited for a person to create `.confirmed`, even though backlog drives already pass the flag. An unmeasured LF, Short or superlong part (no lower-thirds) whose exact text is the drive's placeless label gets an `agree` review only when every word of that label is in `content_gates.PLACELESS_NO_CLAIM_WORDS` (no scenery, weather, season or time of day, e.g. "Somewhere On The Road"). A descriptive placeless label ("Through The Quiet Hills") and any other unmeasured text still wait for a person.

## [0.3.52] - 2026-11-08

### Fixed

- Phase A now refuses a full_drive day that the GPS-gap cut alone leaves under `full_drive_part_min_sec`, before the GPU scoring pass. The planner refuses that day anyway, since the stop mask can only cut more, but it used to do so only after scoring had run. On the 03-04 drive the new check gives the planner's own 2889 s against the 3600 s floor. A GPS-less day is never refused by this check.

## [0.3.51] - 2026-11-08

### Fixed

- When phase_bc can't build a drive plan, its error now gives `build_drive_plan`'s own reason. One example: "kept (stop-masked) footage 2889s is under full_drive_part_min_sec=3600s". The old message said "despite complete inputs ... Re-run with --fresh", but a fresh run hits the same refusal. The real cause appeared only in an earlier WARNING line. The old wording stays only for a failed build that recorded no reason.

## [0.3.50] - 2026-11-08

### Fixed

- A drive that `health backlog` retires is no longer re-resumed by `health heal` on every new mirror SHA. Before this, the dead 03-04 audit was resumed three times (jobs ba3a9dbf, 15c1c9b0, 1affffde), and each run held the heavy slot for about 35 minutes while the backlog waited. Retiring now writes the `no-auto-resume` hold beside each of the drive's resume recipes, and heal already refuses a held recipe. A drive already `failed` in the store with no backlog row gets the same hold on the next run, and the result lists each held stage dir under `held`.

## [0.3.49] - 2026-11-07

### Fixed

- `health backlog` no longer stalls a region on a failed audit whose row was dropped from `drive_backlog.yaml`. That drive could not be relaunched or superseded: `--supersede` refused because the row was not in the backlog. So every call refused until someone ran `set-status failed` by hand. Now the store row is set `failed`, the result lists it under `retired`, and the next row launches. A dry run only lists the retirement.

## [0.3.48] - 2026-11-07

### Fixed

- `health` no longer reports "a drive is running" because of its own arguments. With no live job record, the process check matched any `gomoveshift` command line containing " drive ", so `health backlog --supersede ... --reason "queue next Ireland drive"` blocked itself. Now it matches only when `drive` is the subcommand.

## [0.3.47] - 2026-11-07

### Fixed

- Shorts YouTube descriptions no longer open "Through the windshield …": the plan grammar gate refused a lead starting with a bare preposition and failed the whole publish plan. The lead is now "Over the hood", and a test checks every lead against the gate.

## [0.3.46] - 2026-11-07

### Fixed

- The `🗺️ Through:` line drops the LF's own label when it is the trailing qualifier of a region name, so an LF whose location was broadened to a state lists `Polk County, Henderson County` instead of repeating `North Carolina` per county and failing the description redundancy gate (US-64, 2026-11-07).

## [0.3.45] - 2026-11-07

### Fixed

- A Phase-D-only resume of a checkpoint whose burn label the route-claim check broadened now gives that LF the broadened text as its location, read from the sealed review in state. Before, the saved county reached Phase D and failed the location invariant after the encode (US-64, 2026-11-07).

## [0.3.44] - 2026-11-07

### Fixed

- `health heal` resumes a drive at most once per recipe per mirror SHA. Before, `health check` filed each failed heal job as a new incident row and heal resumed that row again on the same code, so one deterministic failure re-ran every hour and its live job kept deferring the staging sync that carried the fix. A heal of a heal no longer repeats `--skip-upload --no-youtube-upload`.

## [0.3.43] - 2026-11-06

### Fixed

- When the route-claim check broadens a long-form's burned label (for example to the state), the LF's location label now broadens with it. Phase D used to keep the narrower county on the `📍 Location:` line, so the location invariant failed after the full encode (US-64 LF 1: burn "North Carolina", location "Transylvania County, North Carolina", Henderson County 37% of the footage).

## [0.3.42] - 2026-11-06

### Fixed

- `health publish` books a Waypost slot for every long-form a drive rendered. It refused any drive the store did not list with exactly one long-form, and `health backlog` lists every drive with one, so a two-long-form drive would have got one booked slot while Phase D dated its second long-form a week later with nothing booked. The count now comes from the audit's `drive_plan.json` (`regular_lfs`); a count that differs from the store is written back before the claim, and a drive with no readable plan is refused before any write.

## [0.3.41] - 2026-11-06

### Fixed

- A second drive at the same place no longer has its shorts refused by `enqueue-plan`'s `cross-batch-repeat` rule. The YouTube description led every short with the fixed "POV driving through {place} Pt.N", whose first six words are the same for every drive there, and the caption openers could repeat one another project had already spooled. Caption openers now skip any opener whose first six words match text spooled in the last 60 days, read-only from the queue, and the description lead rotates through 14 phrases in the same way.

## [0.3.40] - 2026-11-05

### Fixed

- The day-GPX slicer no longer refuses a drive whose recording ends in a short chapter. Its 15 s + 120 s padding was over the span gate's 120 s floor for any clip under 540 s, so a fully covered 198 s last chapter failed as "GPX likely belongs to a different drive". The tail now shrinks (down to 105 s) to fit the unchanged gate.
- A YouTube long-form upload now gets its 5 retries for each 256 MB chunk. Before, every chunk that uploaded used up one of the 5, so an LF over 1.25 GB failed on its first network timeout with no retry. Kenmare (about 8 GB) failed this way twice.

## [0.3.39] - 2026-11-05

### Fixed

- `health heal --recipe` on a drive whose recipe row `health check` had already resolved did
  nothing and said nothing. Registering the recipe now reopens that row (stamped `reopened_at`,
  attempt history kept), so the heal runs.
- An open `drive_failed` row filed before `health check` read a job's own `output.log` kept a null
  `resume_recipe` forever, so `health heal` refused it ("no readable resume.json") on every pass.
  `health check` now fills a missing `resume_recipe` or `log_path` from the current detection.
- `health publish` approved and claimed a drive whose checkpoint the current code re-runs (Kenmare: `phase_bc_day_first` 3 → 4 after its audit), so Phase D refused its prerequisite after the slot was booked. It now refuses before any write, naming the stale phases, so the drive gets re-audited first.
- A drive whose `health publish` job ended without uploading was left `failed` in the schedule
  store, and `health publish` refused anything not `auditing`, so nothing could publish it again.
  `health publish` now reopens a `failed` drive with no upload receipt to `auditing` and claims a
  slot again; every other check still applies.
- A `ready` drive already uploaded outside a claim (Beaufort: its receipt under the run dir, its booking under `src`) sat first in Waypost's claim line, so `health publish` refused every newer drive in the region. `health publish` and the hourly `health check` reconcile now retire such a drive from its receipts through Waypost's `retire_published` (booking adopted, drive `complete`); a receipt Waypost cannot book leaves it `ready`.

## [0.3.38] - 2026-11-05

### Fixed

- The backlog launcher now passes a row's `placeless_label` to `drive` as `--placeless-label`. It used to drop the field, so a no-GPS row launched with no label and the place gates refused it.

## [0.3.37] - 2026-11-05

### Fixed

- `publish-journal --allow-day-create` left `trip.end_date` and `trip.duration_days` behind when it added a day after the trip's end, so the site's `journey-integrity` check failed with `day-count-mismatch`. It now moves `end_date` to the new day and recomputes `duration_days` (end - start + 1), the same rule `extend_adjacent_journey` uses (now one shared `set_trip_end`). A day created before `trip.start_date` is refused, since moving the start would renumber published days. Re-running the same `--apply` repairs a trip block already out of range with its days, even when every LF is already present; a dry run reports the repair without writing.
- `health heal` no longer relaunches a failed drive with `--phase <letter>` (the drive CLI's single-phase mode). A drive that failed in phase A or B/C used to pass that one phase and then fail the DoD gate (`lf_count 0 < 1`) because the later phases never ran. Heal now runs the plain audit-only resume, which runs every phase the checkpoint does not hold, and no longer refuses a failure whose phase it cannot map.
- A backlog drive job's `drive_failed` incident now gets its resume recipe, phase, gate and error line. The drive check reads the job's own `output.log` beside its `job.json` as well as the `expect` list, which for a backlog job is an artifact glob rather than a log, so heal no longer refuses "no readable resume.json for this incident".

## [0.3.36] - 2026-11-04

### Fixed

- `gomoveshift health heal` re-audits a drive whose checkpoint was stamped by an older phase code version (for example after `PHASE_BC_CODE_VERSION` moved to 4) as a full audit-only resume instead of `--phase D`, which stopped at the prerequisite check. Renders are reused by fingerprint; the attempt records `stale_phases`.

## [0.3.35] - 2026-11-04

### Added

- An Irish county claim ("County Cork") on a route with GPS gaps is proven on Tailte Éireann's pinned 2026 statutory county map (CC BY 4.0, © Tailte Éireann) instead of refused for want of a US Census boundary. The map is downloaded only over https from the three ArcGIS hosts in `docs/pipeline/84-network-allowlist-credential-stores.md`, checked against its SHA-256 and cached. Every fix must sit at least 250 m inside the county beyond the speed and error bound, which covers a recording made before the 2026 map. `BOUNDARY_REVIEW_VERSION` is `tiger2026-tailte2026-bounded-v2` and `PHASE_BC_CODE_VERSION` is 4, so a resumed drive rebuilds its route record with the proof.

### Changed

- Pinned Waypost (`seo-manager`) to v0.5.0 (`2ce9f47`, Back-Road-Creative/waypost#106, #107), whose final-text review accepts the Tailte Éireann proof.

## [0.3.34] - 2026-11-03

### Fixed

- `health backlog --supersede --dry-run` now prints `would supersede <from> -> <to>`. It used to print `superseded`, though a dry run moves nothing.
- The pre-encode location invariant now accepts a state label, such as `North Carolina`, when that one state holds every material county. That label is the route gate's own fallback when no county can be proven. The invariant used to refuse it as "omits Henderson County", so two gates contradicted each other and the drive stopped before encode.

## [0.3.33] - 2026-11-03

### Fixed

- A region-only long-form title for a country with no states (an Irish county,
  "County Cork in 4K") now carries Waypost title facts. Before, only a US
  "County, ST" label could match, so every Irish region title was refused before
  the OST burn as `facts_unavailable`. The county must still pass the route gate,
  and its retained footage must resolve to exactly one known country.

## [0.3.32] - 2026-11-03

### Added

- `drive --placeless-label TEXT` ships a drive with no GPS fix at all under an on-screen label that names no place (for example "Somewhere Between"). The label is held to a fixed vocabulary at parse time (letters, spaces and apostrophes, 1-40 chars, no place names, digits or "to"), is refused beside `--location`, `--per-lf-location` and `--day-gpx`, and every LF is awarded the ladder's tier 5 with the recorded reason `placeless_operator_label`. The specificity-ladder gate re-checks the vocabulary, so a place word fails even when the placeless flag is set; every label that does name a place is still checked against the trace.

### Fixed

- `health backlog` skips a row the render gate refuses (a `full_drive` day too short to fill a
  part) and launches the next eligible row, instead of refusing the region on every tick. The
  skip is listed with its `render gate:` reason; with no eligible row behind it the region is
  still `refused`.
- Locked-GPS coverage no longer scores a variable-rate phone or enhanced GPX as
  sparse. A gap of up to 10 s between two consecutive fixes now counts as
  covered; longer gaps still count as dead, and a 1 Hz GoPro trace scores as
  before. The fully tracked Ireland day 11 drive went from 41% (below the 60%
  tier-1 bar, so its place-name title was refused) to 88%.

## [0.3.31] - 2026-11-03

### Added

- `health publish <project> --region R --stages-root S`: the single operator command that approves
  an audited drive in Waypost, claims its long-form slot and launches the Phase D upload with that
  slot's date. The hourly `health check` closes the claim when the job ends: complete when an upload
  receipt exists, failed (booking released) otherwise. Refuses while an older ready drive sits ahead,
  since Waypost claims a region's oldest ready drive. (#179)

## [0.3.30] - 2026-11-02

### Fixed

- `health backlog` no longer refuses to relaunch a `day_gpx` row whose earlier drive got as far as
  slicing the day GPX: the slicer's own `<stem>.gpx` sidecars beside the chapters in
  `<sources root>/<project>` match the row. Any other extra file, or a `.gpx` the slicer did not
  write, is still a mismatch.

## [0.3.29] - 2026-11-02

### Fixed

- `health backlog` asks the drive's raw-duration gate before it supersedes, launches or books
  anything: a `full_drive` row whose day of raw footage is under `full_drive_part_min_sec` is
  `refused` (`render gate: …`), in a dry run too. Before, the launch went through (and a
  `--supersede` moved the old run aside) and the drive refused the day only after it started,
  leaving an `auditing` store row for a render that could never happen.
- `health backlog --supersede <project>` also replaces that row's failed audit (store row
  `auditing`, newest job not succeeded): the run is set aside and the row relaunches as the
  manifest now reads it. Before, a failed audit could only be resumed with its old arguments.

## [0.3.28] - 2026-11-02

### Fixed

- `health backlog` no longer counts a drive that is already uploaded (a `youtube_uploads.json`
  with a video id at its output root or one level down) as holding or covering a long-form slot,
  so a published drive whose Waypost store row went stale no longer blocks the next launch. The
  result lists such drives under `published`. (#177)
- `health backlog --supersede` also sets aside a dead sibling run: an earlier run of the same
  footage under another project name. It still refuses a sibling with an operator hold, with upload
  records, or in the schedule store or content ledger. Every move is recorded in each
  `superseded.json`. (#178)

## [0.3.27] - 2026-11-02

### Added

- `health backlog --supersede <project> --reason <text>`: a row whose own earlier run is stale
  (for example rendered on an older release) moves that run's stage root and output dir to
  `_superseded/` with a `superseded.json` record and launches fresh, instead of being skipped as
  already staged. Refused for a published or booked project, an output with upload records, or
  any row but the region's next.

## [0.3.26] - 2026-11-02

### Added

- `gomoveshift clip-claims backfill OUTPUT_ROOT... [--apply]` rebuilds the clip-claims registry from past outputs: every project folder with an `lf_provenance.json` whose long-form has a `video_id` in `youtube_uploads.json` gets its source clips claimed, so the Phase A reuse gate also sees footage published before the registry shipped. Dry-run by default, idempotent, and a folder without `lf_provenance.json` is reported skipped, never guessed.
- The pre-render review gate now refuses a bare county/region title (for example "County Cork in 4K") when the channel already has a title naming that county/region; a title naming a town or a from-to route is still allowed.

### Changed

- `clip-claims backfill` now also covers outputs written before `lf_provenance.json` named source clips: with no per-LF clips (or no provenance file) it claims the camera files named in the folder's `drive.log` for every uploaded long-form, tagged `basis: drive_log_project`. The reuse gate matches clip names by case-folded stem (a `GX010064` claim catches `GX010064.MP4`) and its refusal marks a whole-project claim as coarse.
- `--day-gpx` no longer needs `--camera-utc-offset-hours`: the slicer measures the offset from each recording's first-chapter ffprobe `creation_time` (filename local time minus UTC). It refuses, naming the clips and values, when a clip has no `creation_time`, recordings disagree, the value is not a whole multiple of 15 min (within 60 s) or lies outside -12..+14 h, or the slice fails the Phase-A gate. A given offset that disagrees with `creation_time` is now refused instead of warned about. `health backlog` passes the flag only when the manifest row sets it.

### Fixed

- `drive` now refuses a real render unless `health backlog` started it (`GMS_LAUNCHED_BY=health-backlog`); `--dry-run`, `--list-lfs`, `--label-dry-run` and `--preview` are exempt, and `GMS_ALLOW_MANUAL_DRIVE=1` is the logged operator override. `health backlog` also refuses to launch when the render interpreter's framegrade is older than the newest `vX.Y.Z` release tag (`GMS_RELEASE_REMOTE` overrides the remote), or when either version cannot be read.
- The same gate accepts every tracked launcher, not only the backlog: `GMS_LAUNCHED_BY` may be `health-backlog`, `health-heal` (a heal resume), `console` (the console job store) or `hostworker` (set by the worker itself on spooled drives, never taken from the request). An unknown value is refused and named.
- A long-form whose Waypost title decision is `facts_unavailable`, or whose locked-GPS fix density is under the 60% tier-1 floor, now stops before the OST burn instead of falling back and burning an unproven place name. `GMS_ALLOW_UNPROVEN_TITLE=1` overrides.

## [0.3.25] - 2026-11-01

### Added

- Phase A now refuses a drive whose camera files another project already published, before the concat/encode. Claims are recorded in `clip_claims.jsonl` beside the SEO events ledger when a long-form gets its YouTube id; `GMS_ALLOW_CLIP_REUSE=1` overrides with a warning.
- `lf_provenance.json` now names the camera recordings each long-form was cut from: `source.clips` per LF (name, path, size, episode, offset, duration) and a top-level sorted `source_clips` list, so footage reuse across projects can be checked.

### Fixed

- The duplicate-title gate (review gate's exact-duplicate check, Phase D title dedupe and the title kit) now reads the live-catalog snapshot as well as the analytics ledger, so a title only the catalog knows no longer slips through.

## [0.3.24] - 2026-11-01

### Added

- The shipped GoMoveShift brand runs the `fb_hashtags` Facebook caption experiment from 2026-11-02 for 45 days: arm A composed caption with no generic hashtags, B adds 2 location hashtags, C the June-style set. Each arm reports a median reach once it has 20 settled, measured shorts.

## [0.3.23] - 2026-10-31

### Fixed

- A GPS gap that runs to the first or last retained second of a cut is bridged to the county of the nearest source fix just outside the cut when the bin on its other side agrees, so a best-moments cut that ends inside a logger pause no longer refuses a county label it otherwise proves; the bridge is recorded in boundary_notes.

## [0.3.22] - 2026-10-31

### Added

- A brand policy can declare caption experiments (`policy.experiments`): arms with a `hashtag_mode` (`none`, `location`, `june`) are assigned by a stable hash, stop `stop_after_days` after `start`, and are tagged in the publish-plan sidecar (`experiment_arms`). `metrics collect` prints `EXPERIMENT <name> platform=facebook arm=<id> n=<settled> median_reach=<value|unknown>` per arm; the median waits until every arm has `min_n` settled, measured shorts.
- `metrics collect` builds a one-time ledger of Page Reels posted before first-party publishing (2026-09-10) at `<brand>/tracking/facebook_history.json`, tagged `source=metricool` with lifetime metrics, and prints `FB_HISTORY listed=<n|unavailable> stored=<n>`. Reels whose captions were hand-edited on 2026-10-02 carry `caption_edited_on`, and `in_matched_cohort` excludes them.

### Changed

- The Facebook suppression detector now leaves `SUPPRESSED`: a 7-day median back to 0.5x the baseline frozen at the flip is `RECOVERING`, 7 straight days at 0.7x is `HEALTHY`, and a fall under 0.3x relapses to `SUPPRESSED`. Same function signature and return shape.

## [0.3.21] - 2026-10-31

### Fixed

- The before-ingest raw-footage floor (`full_drive_part_min_sec`) now applies only when the resolved long-form mode is `full_drive`. An operator-chosen `--lf-mode best_moments` shadows the part bounds, so a short day in that mode is no longer refused for a planner that never runs.

## [0.3.20] - 2026-10-31

### Changed

- The Facebook suppression detector reads the Page daily series from `channel_daily` once one has more than 28 days, trying `page_total_media_view_unique`, `page_media_view`, `page_video_views`, `page_impressions_unique` in that order; otherwise it keeps the reel-plays series. The cron line gains a trailing `source=<metric name|reel_plays>`.

### Fixed

- A route through two counties that carry no state (such as Irish counties) now labels itself "County A & County B, <country>" from the GPS already retained, instead of being refused before encode.
- On such stateless-county routes, a GPS gap or an unresolved bin with the same county on both sides is counted inside that county. A gap or bin next to an unresolved neighbour, or between two different counties, still refuses.

## [0.3.19] - 2026-10-31

### Added

- `metrics collect` pulls Facebook Page daily insights (`page_media_view`, `page_total_media_view_unique`, `page_video_views`, `page_impressions_unique`) into `channel_daily` under `facebook:<metric>`, labelled by the Pacific day, backfilling from 2026-06-01 on the first run. A metric Graph refuses is reported unavailable, never stored as zero; the run prints `FB_PAGE_DAILY <metric>=<rows|unavailable> ...`.
- Facebook Reel observations record `reach` and the metric that answered (`reach_metric`); reach stays unknown when no candidate answers.

### Fixed

- `sync-site` no longer dies on one transient network failure. The four read-only YouTube Data API listing calls behind the channel manifest (`channels.list`, `playlistItems.list`, `videos.list`) now retry up to 4 times with exponential backoff on a read timeout, connection reset, SSL error or HTTP 5xx, then raise the last error. A 4xx still fails at once. The 2026-10-31 daily run was lost to a single `TimeoutError: The read operation timed out`.

## [0.3.18] - 2026-10-30

### Fixed

- The label-quality backstop no longer warns "county-only" on a settlement-led label such as "Goleen, County Cork"; road-led labels ("R591, County Cork", "NC Highway 28, Macon County, NC", "M50 Motorway, County Dublin", or any road the resolver recorded) still warn. The check stays warn-only.
- The shorts CSV grammar gate now checks each YouTube tag on its own, so a tag that starts with a bare preposition is caught anywhere in the cell, not only when it is first.
- Shorts tags strip a leading title word ("To", "Through") from each location component, so "To Goleen, County Cork" no longer yields a "To Goleen" tag.
- A `--per-lf-location` correction on a Phase-D-only resume now reaches shorts whose burned label was inherited from their parent long-form. Phase D read every short's label from the day's shorts OST audit sidecar, which a resume never re-derives, so the corrected text stayed in those shorts' metadata. A short whose own clip GPS named the place is unchanged.

## [0.3.17] - 2026-10-30

### Added

- `metrics collect` now builds a daily Facebook reel-plays series from the nightly observation log (`tracking.facebook_series.facebook_reel_plays_series`) and feeds it to `classify_facebook_reel_reach`, so Facebook is classified shadow-first beside YouTube. The run prints `SUPPRESSION facebook=<status> days=<n>` (`WARMUP` below 28 days) and the equivalent `SUPPRESSION youtube=` line.
- A backtest fixture (`tests/test_suppression_facebook_backtest.py`) shows the detector flags the August 2026 Facebook reel-reach collapse.

### Changed

- Facebook, Instagram, TikTok and YouTube captions are composed from independently rotated parts (an opener, a data-only detail or an engagement prompt for interactive shorts) instead of four fixed templates per platform, so a 12-short batch never repeats an opener.
- Generic hashtags rotate per short from a pool of six instead of every post carrying `#POVDriving #RoadTrip`; the place and region tags stay mandatory.
- `enqueue-plan` refuses a batch whose caption or YouTube title opens with the same six words as text already spooled in the last 60 days (`cross-batch-repeat`).

### Fixed

- A drive recorded the day after a journaled trip ends now joins that trip's journey: `extend_adjacent_journey` extends `trip.end_date` (and `duration_days`) by one day and adds the day stub, instead of the drive hook creating a separate `draft: true` journey. Ace Basin (2026-02-21) had been split off `sc-2026` this way. A trip starting the day after is not extended backwards, and two adjoining trips are refused.
- `publish-journal --replay-pending` now writes each ready entry's title and duration as YouTube shows them at replay time, not the values stored at drive time. A video retitled live before it published (Ace Basin LF 1) was going to reach the journal under its old title. A video still processing at upload reports a zero duration, and `plan_merge` now treats that as missing and falls back to ffprobe instead of journaling `00:00` (Ace Basin LF 2).
- `publish-journal --replay-pending` no longer strands a due entry whose journey file is gone: it flushes it into the journey that holds the entry's day (a trip range holding it, else the trip that ended the day before, extended on `--apply`), records the route there, and consumes the entry. Ace Basin's two long-forms had been deferred into a draft journey that should never have existed; once that draft was set aside, they would have stayed pending every hour. No match or an ambiguous day still keeps the entry pending.

## [0.3.16] - 2026-10-30

### Fixed

- A long-form whose retained GPS runs through exactly two counties of one state is now labelled `"A & B Counties, ST"` instead of falling back to the bare state. The route-claim fallback ladder gains a `counties` level between `county` and `state`; its boundary proof tests the route against the union of the two Census county polygons, the claim reaches Waypost's region claims, and the title and place-text gates accept the pair as naming the route. Ace Basin LF 1 (2026-02-21, Beaufort and Colleton) shipped as "South Carolina in 4K" because no such level existed.

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

[Unreleased]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.83...HEAD
[0.3.83]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.82...v0.3.83
[0.3.82]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.81...v0.3.82
[0.3.81]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.80...v0.3.81
[0.3.80]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.79...v0.3.80
[0.3.79]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.78...v0.3.79
[0.3.78]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.77...v0.3.78
[0.3.77]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.76...v0.3.77
[0.3.76]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.75...v0.3.76
[0.3.75]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.74...v0.3.75
[0.3.74]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.73...v0.3.74
[0.3.73]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.72...v0.3.73
[0.3.72]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.71...v0.3.72
[0.3.71]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.70...v0.3.71
[0.3.70]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.69...v0.3.70
[0.3.69]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.68...v0.3.69
[0.3.68]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.67...v0.3.68
[0.3.67]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.66...v0.3.67
[0.3.66]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.65...v0.3.66
[0.3.65]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.64...v0.3.65
[0.3.64]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.63...v0.3.64
[0.3.63]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.62...v0.3.63
[0.3.62]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.61...v0.3.62
[0.3.61]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.60...v0.3.61
[0.3.60]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.59...v0.3.60
[0.3.59]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.58...v0.3.59
[0.3.58]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.57...v0.3.58
[0.3.57]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.56...v0.3.57
[0.3.56]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.55...v0.3.56
[0.3.55]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.54...v0.3.55
[0.3.54]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.53...v0.3.54
[0.3.53]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.52...v0.3.53
[0.3.52]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.51...v0.3.52
[0.3.51]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.50...v0.3.51
[0.3.50]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.49...v0.3.50
[0.3.49]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.48...v0.3.49
[0.3.48]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.47...v0.3.48
[0.3.47]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.46...v0.3.47
[0.3.46]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.45...v0.3.46
[0.3.45]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.44...v0.3.45
[0.3.44]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.43...v0.3.44
[0.3.43]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.42...v0.3.43
[0.3.42]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.41...v0.3.42
[0.3.41]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.40...v0.3.41
[0.3.40]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.39...v0.3.40
[0.3.39]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.38...v0.3.39
[0.3.38]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.37...v0.3.38
[0.3.37]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.36...v0.3.37
[0.3.36]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.35...v0.3.36
[0.3.35]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.34...v0.3.35
[0.3.34]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.33...v0.3.34
[0.3.33]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.32...v0.3.33
[0.3.32]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.31...v0.3.32
[0.3.31]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.30...v0.3.31
[0.3.30]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.29...v0.3.30
[0.3.29]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.28...v0.3.29
[0.3.28]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.27...v0.3.28
[0.3.27]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.26...v0.3.27
[0.3.26]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.25...v0.3.26
[0.3.25]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.24...v0.3.25
[0.3.24]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.23...v0.3.24
[0.3.23]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.22...v0.3.23
[0.3.22]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.21...v0.3.22
[0.3.21]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.20...v0.3.21
[0.3.20]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.19...v0.3.20
[0.3.19]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.18...v0.3.19
[0.3.18]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.17...v0.3.18
[0.3.17]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.16...v0.3.17
[0.3.16]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.15...v0.3.16
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
