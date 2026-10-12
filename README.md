# GoMoveShift Video Processing Service

Python-based video processing service for travel footage automation.

## Purpose

Automated video processing pipeline for GoMoveShift travel content. Handles frame extraction, sign detection via OCR (EasyOCR), dead spot removal, and intelligent video segmentation. Includes web UI for monitoring and control.

## Status

Active - Development/testing phase with core processing capabilities implemented.

## Exit Condition

Project completes when video processing workflow is fully automated and stable, or replaced by alternative solution.

## Testing

`tests/conftest.py` points `HOME` at an empty temp directory in a module-level
block, before `import gomoveshift`, so the suite is blind by construction to
the developer's real `~/.config/gomoveshift/config.yaml` from the moment
collection starts importing test modules (a fixture only runs at test setup,
after collection has already imported everything, including code paths like
`gomoveshift/pipeline/longform_metadata.py` that read `Config.load()` at
import time). The session-scoped `isolate_home_from_operator_config` fixture
mirrors that same fake home onto `Path.home` for code that reads it directly.
The same block drops `GMS_DATA_HOME`, `GMS_CACHE_HOME`, `GMS_CONFIG_HOME` and
the XDG data/cache/config home variables so the per-user defaults cannot resolve into a real
user directory. The fake home sits under `~/.cache/gms-test-homes/` (removed at
exit), never `/tmp`: defaults such as `paths._DEFAULT_INTERMEDIATE_DIR` resolve
under the home directory and are pinned by tests to stay off tmpfs.
A real config file can carry operational overrides (e.g. a
`publishing_route_youtube` flip) that have nothing to do with the code
under test — without this isolation, a workstation with that file present fails
differently than CI, and a developer's stale override silently changes their
local suite result. See `tests/test_conftest_isolation.py` and
`tests/test_conftest_hermetic_config.py`.

The output root gets the same treatment: the autouse
`isolate_output_root_from_operator` fixture points `GMS_OUTPUT_DIR` inside
each test's own `tmp_path`, so a default-constructed `TrackingStore` (where
`enqueue-plan --apply` registers every enqueued short) never writes the real
output root. A test that needs a specific root still sets
`GMS_OUTPUT_DIR` itself.

**A test never reads the real clock.** A test that calls
`datetime.now()`, `date.today()` or `time.time()` passes today and fails
later — when the calendar week rolls, a slot passes, or a fixture ages out —
and a flaky default branch stops every merge. Freeze the moment
(`freeze_time`, `time_machine`) or pass it in as an argument.
`tests/baselines/test-quality.json` lists the clock reads that predate the
rule; a read you have checked is safe (a timeout loop, a timing measurement)
belongs in that baseline, and fixing a listed test shrinks it.

## Lint Config

`ruff.toml` extends `pyproject.toml`'s `[tool.ruff]` select/ignore and adds `extend-select` for rule families that are already clean across the whole service — a ratchet so a passing family can never quietly regress. Add a family only after fixing every finding it reports today (`ruff check . --select <FAMILY>`); never add one with open findings.

## CLI Flag Semantics

### `--shorts-target N`

**Behavior:** `N` is interpreted as a **per-LF count** — `N` shorts per long-form within the day.

When the flag is supplied:

- The orchestrator persists `N` to `state["shorts_target"]` and mirrors it onto `config.drive.shorts_target_count`.
- In `pipeline/day_pipeline.py::run_day_pipeline`, the read-priority is `state.get("shorts_target")`. When set, the planner is invoked with `target_count = N * num_lfs_in_day` (per-LF semantics, multiplied by the number of LFs assembled from that day's master).
- Net effect: `--shorts-target 14` on a day with 1 LF plans 14 shorts; on a day with 3 LFs plans 42 shorts. Phase D partitions the resulting shorts across each LF's publish window by datetime.

**Default behavior (flag omitted):** when `--shorts-target` is not passed, `state["shorts_target"]` is unset and the planner uses Waypost's decided per-LF count (`pipeline/volume_source.py`) `* num_lfs_in_day`. No config key sets the count (`shorts_per_lf_count` is retired; a leftover key logs `RETIRED_CONFIG_KEY`). `--shorts-target` can only lower the decided value. Full precedence and the capping rules: `docs/pipeline-internals.md`.

### `--shorts-min-duration` / `--shorts-max-duration` / `--shorts-target-duration`

Override the short-clip duration bounds `short_planner.plan_shorts` uses to build candidate windows — otherwise sourced from `config.drive.shorts_{min,max,target}_duration_sec` (defaults 30/45/38s, mirroring `short_planner.DEFAULT_{MIN,MAX,TARGET}_DURATION_SEC`). `shorts_max_duration_sec` may never exceed 60s (`video_types.SHORTS_MAX_DURATION`, the YouTube Shorts hard ceiling), and `min <= target <= max` is enforced at config construction — both raise `ValueError` on violation, whether set via CLI or `config.yaml`.

### `--ingest-mode gopro` (default) needs the `gopro` extra

GoPro ingest reads each clip's GPS from its GPMF telemetry with `gopro-overlay`. Install it with
`pip install -e '.[gopro]' && pip install --no-deps gopro-overlay==0.134.0`: it exact-pins gpxpy==1.6.1,
which the sibling `gps-journey` package cannot take, so the `gopro` extra carries only its other deps. Preflight step 1c refuses a `gopro` run whose venv cannot
import it, instead of letting every clip come out GPS-less hours later. Footage with no GoPro
telemetry runs `--ingest-mode generic`, which skips the check.

### `gomoveshift comments pull [--since 90d] [--out PATH]`

Read-only: pages `commentThreads.list` (`allThreadsRelatedToChannelId`, newest first) and appends new comments to `comments.jsonl` in the data home (override with `GMS_COMMENTS_PATH`), skipping any `comment_id` already on disk. Requires `--channel-id`.

### `gomoveshift comments triage [--in PATH] [--out PATH]`

Reads the `comments.jsonl` ledger written by `comments pull` (default: `GMS_COMMENTS_PATH`, else `comments.jsonl` in the data home; override `--in`) and writes a markdown triage report (default `comments-triage.md` in the data home, override `--out`) grouping flagged comments by video. A comment is flagged when it matches a correction cue (`actually`, `that's not`, `wrong`, `it's called`, `not in`, `you mean`, `this is`, `isn't`, `incorrect`, `mislabel`, etc., case-insensitive) or names a place (2+ Title-case words, or a Title-case word after `in`/`near`/`at`) not already present in the video's own title/description. Titles/descriptions are looked up by `video_id`, first from the SEO ledger, then from the site checkout's `data/videos.yaml`; both are optional — if neither is present, only the cue check runs.

### `gomoveshift comments import --platform tiktok FILE [--video-id ID] [--out PATH]`

No-API import for platforms `comments pull` can't reach (TikTok has none here). `FILE` is either a header-detected CSV (any of `video_id|video|url`, `author|user|username`, `text|comment|content`, `date|created|published_at|time`, case-insensitive; a TikTok URL in the `url`/`video` column yields `video_id` = its trailing numeric id) or a plain `.txt` file where each non-empty line is `author: text` or a bare `text` line — `--video-id` is then required since a `.txt` export carries no per-row video reference. Rows are appended to the same `comments.jsonl` ledger `comments pull` writes (`comment_id` = sha1 of `platform|video_id|author|text`, deduped against existing ids), so `comments triage` runs over TikTok and YouTube rows unchanged.

### `gomoveshift cleanup [project] [--confirm] [--min-age N]`

Dry run unless `--confirm`. Scans the output dir (`GMS_OUTPUT_DIR`) **and** the
drive stage cache (`GMS_INTERMEDIATE_DIR`, default the cache home),
where drive runs keep `_stages/`. For a drive with `phase_d` done it lists each
`day_*/regular_lfs/lf_NN.mp4` whose LF has a YouTube `video_id` in
`youtube_uploads`, and the `episode_*.mp4` concat once every LF is uploaded
and `source_dir` still holds the footage. Once every LF is uploaded it also reclaims the copied sources under the same
cache base, `sources/<project>` (the ext4 copy preflight asks for before a
drive run); the dry run prints it as a `SOURCES DIR` line with its size. Symlinks
inside are unlinked, never followed (the originals are untouched), a symlinked
`sources/<project>` is skipped, and real files go only under `--confirm`.
Symlinked episodes, the blur mask,
`pipeline_state.json` and unfinished drives are never listed. `--min-age`
(default 7 days, by `_stages` mtime) still applies; pass `--min-age 0` to
reclaim a drive published this week.

### `gomoveshift cleanup --archive-base PATH`

Opts a `cleanup` run into archive-backed reclamation: `PATH` is a distinct directory
(e.g. an external archive mount) expected to hold a size-matched copy of each
project's `<name>_FINAL.mp4`. When set, `PipelineCleanup` verifies the copy
before queuing the output deliverable for deletion. **Unset by default**
(`--archive-base` omitted, `d_drive_base=None`) — the deliverable is never
deleted unless you explicitly point at a real, distinct archive
location. There is no environment-variable default; the flag is the only way
to supply it.

## Environment Variables

Every `GMS_*` variable is optional: omitting one means "use the default" rather than "fail", and a variable with no default turns its feature off when unset. The table covers the variables that affect a drive or photo run; `GMS_SITE_PATH` and `GMS_SUPERVISED` are documented at their use sites.

Defaults are per user and follow three roots, each resolved the same way (the first that is set wins):

| Root | Resolution | Holds |
| --- | --- | --- |
| data home | `GMS_DATA_HOME`, else `$XDG_DATA_HOME/gomoveshift`, else `~/.local/share/gomoveshift` | inputs (`input/`), outputs and exports (`output/`), photo state (`photos/`), the comments ledger |
| cache home | `GMS_CACHE_HOME`, else `$XDG_CACHE_HOME/gomoveshift`, else `~/.cache/gomoveshift` | stage caches, geocode, Overpass and Route Story caches, lock files |
| config home | `GMS_CONFIG_HOME`, else `$XDG_CONFIG_HOME/gomoveshift`, else `~/.config/gomoveshift` | `config.yaml`, credential, token, key and SOPS-config files |

The helpers are `gms_data_home`, `gms_cache_home` and `gms_config_home` in `gomoveshift/paths.py`. A location that depends on a separate checkout or service (a site checkout, the SEO data directory, an intel database) has no default: unset means the feature reports "not configured" and names the variable to set. For the four credential and key paths (`GMS_DRIVE_CREDENTIALS_ENC`, `GMS_DRIVE_TOKEN_ENC`, `GMS_AGE_KEY_FILE`, `GMS_SOPS_CONFIG`) an empty or whitespace-only value counts as unset. The container packaging is the one exception to these defaults; see `docker/README.md`.

| Variable | Default | Consequence of omission |
| --- | --- | --- |
| `GMS_STAGING_DIR` | `<data home>/input` | Video staging/input dir (`gomoveshift/paths.py`). `assert_required_paths_exist` raises if the resolved path is missing. |
| `GMS_OUTPUT_DIR` | `<data home>/output` | Pipeline output dir (`paths.py`); also backs the content ledger path (`pipeline/content_ledger.py`). Final deliverables land under the default. |
| `GMS_INTERMEDIATE_DIR` | (none — falls through to the cache home when output is on `/mnt/`) | Absolute override for the ext4-backed `_stages` dir (`paths.py`). Without it, heavy intermediates use the cache home on drvfs output or the output dir itself on ext4. |
| `GMS_ROUTE_STORY_CACHE_DIR` | `<cache home>/route-story` | Route Story map-bundle cache (`paths.py` `gms_route_story_cache_dir`). Kept on ext4 so tile writes are fast; a missing dir is created, a cleared cache just re-fetches. |
| `GMS_OVERPASS_CACHE_DIR` | `<cache home>/overpass` | Overpass response cache (`paths.py` `gms_overpass_cache_dir`). Set to the empty string to never cache. |
| `GMS_GEOCODING_CACHE_DIR` | `<cache home>/geocoding` | Backs the single geocode-cache namespace (`DriveConfig.resolved_geocode_cache_dir`); all consumers — Phase A GPX derivation, Phase BC per-LF labels, Phase D chapters — resolve here so one coordinate resolves identically across a run. A stale/missing path just re-fetches. |
| `GMS_GPS_CACHE_DIR` | (none) | Legacy override for the same unified namespace (takes precedence over `GMS_GEOCODING_CACHE_DIR`); steers **every** geocoder at one dir, not just Phase A. Empty falls back to the default; cache rebuilt on miss. |
| `GMS_WEB_JOBS_PATH` | `<GMS_STAGING_DIR>/_stages/web_jobs.json` | Web-UI job persistence file (`web/app.py`); test override hook. Default snapshot location used. |
| `GMS_OST_ENCODE_TIMEOUT_SEC` | `7200` (2h) | Per-LF OST re-encode timeout (`pipeline/ost_applier.py`). Non-numeric/unset → 2h. Too low would kill a slow NVENC-flap encode. |
| `GMS_WSL_HOST_VOLUME` | `/mnt/c` | Windows host volume the WSL VHDX free-space gate measures (`pipeline/preflight.py`). If the VHDX lives on another drive and this is wrong, the gate skips (logs a warning) or measures the wrong volume. |
| `GMS_DRIVE_CREDENTIALS_ENC` | `<config home>/drive_credentials.json.enc` | SOPS-encrypted Drive OAuth client JSON (`pipeline/uploader.py`, checked by `preflight.py`). A missing/unreadable file fails the Drive-upload preflight. |
| `GMS_DRIVE_TOKEN_ENC` | `<config home>/drive_token.json.enc` | SOPS-encrypted Drive refresh token JSON (`uploader.py`/`preflight.py`). As above. |
| `GMS_AGE_KEY_FILE` | `<config home>/age.key` | age key used to decrypt the two `*.enc` files (`uploader.py`/`preflight.py`). Wrong/missing key fails decryption. |
| `GMS_SOPS_CONFIG` | `<config home>/.sops.yaml` | SOPS config (creation rules + recipients) (`uploader.py`/`preflight.py`). Used for re-encrypt. |
| `GMS_PHOTO_OUTPUT_DIR` | `<data home>/photos` | Photo-pipeline output dir when no explicit `output_dir` is passed (`pipeline/photos/orchestrator.py`). |
| `GMS_EXPORT_DIR` | `<data home>/output` | CSV export target for the photo pipeline (`pipeline/photos/orchestrator.py`). Export is skipped when the dir does not exist. |
| `GMS_COMMENTS_PATH` | `<data home>/comments.jsonl` | Comment ledger written by `comments pull` and read by `comments triage` (`cli/parser.py`). |
| `GMS_COMMENTS_SEO_LEDGER` | (none) | Optional SEO events ledger (JSONL) that `comments triage` reads video titles and descriptions from. Unset: triage prints a note naming the variable and skips that source. |
| `GMS_COMMENTS_VIDEOS_YAML` | (none) | Optional site `videos.yaml` that `comments triage` reads video titles and descriptions from. Unset: triage prints a note naming the variable and skips that source. |
| `GMS_LEASE_BIN` | (none) | Path to a lease helper executable used by the worktree lease guard (`pipeline/orchestrator_lock.py`, `pipeline/drive_orchestrator.py`). Unset: the guard is not configured and runs proceed without a lease. |
| `GMS_DISPOSABLE_WORKTREE_DIRS` | (none) | Directory names, separated by the OS path separator, that mark a code tree as a disposable worktree parent, in addition to the built-in `.worktrees` (`pipeline/orchestrator_lock.py`). Unset: only `.worktrees` counts. |
| `GMS_SOPS_SUDO_USER` | (none) | OS user for a second `sops` decrypt attempt through `sudo -u` when the first one fails (`pipeline/_yt_creds.py`). Unset: the second attempt is off. |
| `GMS_OVERPASS_CONTACT` | (none) | Contact (an e-mail address or URL) appended to the Overpass request User-Agent (`pipeline/chapters/overpass_source.py`). Unset: the User-Agent carries only the project URL. Set it to identify whoever runs the install, as the map-data service's usage policy expects. |
| `GMS_CORRECTIONS_PATH` | `<data home>/place_corrections.yaml` | Place-corrections ledger and place-name denylist (`pipeline/validation/place_corrections.py`, `console/seo.py`). A blank value counts as unset. |
| `GMS_SEO_OWNER_USER` | (none) | Account named in the `chown` repair hint of the SEO state-file ownership check (`pipeline/preflight.py`). Detection and self-repair always run; while this or `GMS_SEO_OWNER_GROUP` is unset the hint shows `<owner-user>:<owner-group>` placeholders. |
| `GMS_SEO_OWNER_GROUP` | (none) | Group named in the same repair hint; see `GMS_SEO_OWNER_USER`. |
| `GMS_GPU_LOCK_PATH` | `<cache home>/gpu.lock` | Shared GPU lock taken by `drive_orchestrator.run` (`pipeline/orchestrator_lock.py`) so an encode and a resident local LLM never hold the same card. Set to the empty string to disable it on a host that does not share the GPU. |
| `GMS_GPU_LOCK_WAIT_S` | `120` | How long a drive run waits for a live holder of that lock before failing. Unparsable values fall back to 120. |
| `GMS_ALLOW_CPU_BLUR` | unset | Preflight step 7c refuses to start a blur run when `onnxruntime-gpu` is installed but CUDA does not actually load: `CUDAExecutionProvider` unavailable (the CPU `onnxruntime` wheel overwrote the GPU build) or listed but a real probe `InferenceSession` binds CPU first (a CUDA 12 build beside CUDA-13 nvidia wheels), caught at minute 0 instead of partway through a run. Set to `1` to accept CPU blur knowingly (much slower); the repair is `bash tools/enable_gpu_detection.sh <venv>`. |
| `GMS_NIGHT_SOURCE_ROOTS` | (none) | `os.pathsep`-separated private roots from which the console may open saved night `.linear` stacks. Unset disables project creation; paths outside these roots are rejected. |
| `GMS_ROUTE_STORY_SOURCE_ROOTS` | (none) | `os.pathsep`-separated private roots from which the Route Story console may read selected GPX, photo, and video sources. Unset disables Route Story draft creation; files outside these roots are rejected. |
| `GMS_ROUTE_STORY_CHROMIUM` | (none — Playwright's bundled Chromium) | Headless Chromium executable for Route Story 3D capture (`pipeline/route_story/scene/capture.py`). Set it when the installed browser build differs from the one Playwright expects; otherwise launch fails. |
| `GMS_ROUTE_STORY_MAPLIBRE_DIR` | (none — the scene integration test skips) | Directory holding a vendored MapLibre 6 `dist/` build (`maplibre-gl.mjs` etc, written by `pipeline/route_story/scene/vendor.py:ensure_maplibre`). Read only by `tests/test_route_story_scene.py::test_scene_renders_deterministically`. |
| `GMS_ROUTE_STORY_RENDER_WORKERS` | (none — `min(6, max(1, cpu_count // 2))`) | Parallel Route Story capture workers (`pipeline/route_story/proof.py:render_proof_clip`); an explicit `workers=` call argument still wins. Unset/unparsable falls back to the default. |
| `GMS_ROUTE_STORY_STILL_WORKERS` | `4` | Spawn processes grading Route Story photo stills in parallel (`pipeline/route_story/still_selection.py:load_stills`); `1` grades serially. Unset/unparsable falls back to the default. |
| `GMS_ROUTE_STORY_FRAME_CACHE` | `1` (enabled) | Opt out of the on-disk captured-frame cache (`pipeline/route_story/scene/frame_cache.py`) with `0` — every pose re-renders through the browser even when a cached frame exists. |
| `GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB` | `40` | Byte cap, in GB, on the captured-frame cache directory; oldest-mtime entries are pruned once the cache exceeds it (`frame_cache.py:FrameCache._prune`). |
| `GMS_NIGHT_ADMISSION_LOCK` | `<system temp dir>/gomoveshift-night-<uid>.lock` | Optional host-mounted lock path shared by participating night workers. Point every console/CLI process at the same regular file when their project workspaces differ; symlinks and non-regular lock files are refused. |
| `GMS_PUBLISHING_ROUTE_YOUTUBE` | `off` | Default for `DriveConfig.publishing_route_youtube` (`config.py`); an explicit `config.yaml`/constructor value still wins. The built-in default is `off`, so a config that names no route publishes nowhere; set `first_party` to opt the platform in. Route is `first_party` (posts via the `youtube_adapter.py` adapter) or `off` (ships nothing); `metricool` is retired and refused. Also settable per brand from the console Settings tab. |
| `GMS_PUBLISHING_ROUTE_INSTAGRAM` | `off` | Same as above for `publishing_route_instagram` / Instagram Reels (`instagram_adapter.py`). |
| `GMS_PUBLISHING_ROUTE_FACEBOOK` | `off` | Same as above for `publishing_route_facebook` / Facebook Reels (`facebook_adapter.py`). |
| `GMS_PUBLISHING_ROUTE_TIKTOK` | `off` | Same as above for `publishing_route_tiktok` / TikTok Direct Post (`tiktok_adapter.py`); independent of `publish_tiktok`, which gates only the shorts CSV's TikTok column. |

(`INTEL_DB_PATH` / `INTEL_SCRIPTS_DIR` in `gomoveshift/intel_db.py` are not `GMS_*`-namespaced and gate an optional, non-fatal IntelDB logging side effect.)

## Console

`gomoveshift console` starts a local FastAPI + Jinja creator console (default `127.0.0.1:8787`).
Rule: **the console fronts the CLIs, it never re-implements pipeline logic** — every action is a
subprocess call to `gomoveshift …` or an import of the same function the CLI calls. The Settings
tab lists the variables in the console registry file
`gomoveshift/console/registry/gms_settings.yaml`. The Settings tab's form persists per-brand
overrides (never the running environment) and every console-launched job applies them; see
`docs/pipeline-internals.md`.

## Route Story telemetry

Route Story derives speed, distance, grade, elevation and trip summaries from the original GPS
recording, never from the speed of an animated map. Its core keeps GPX segments separate: time
between segments is reported as unobserved rather than silently turned into travel or stopped time.
Missing elevation remains unavailable, not a zero grade/ascent. The initial public primitives live
in `gomoveshift/pipeline/route_story/metrics.py`; see `docs/pipeline-internals.md` for metric
definitions before adding a UI or render consumer.

Timed GPX sources are normalized by `gomoveshift/pipeline/route_story/gpx.py`. It preserves
source segment order and timestamp gaps, and rejects timeless, timezone-ambiguous, or malformed track points rather
than creating an animated clock that the recording did not provide.

A Route Story project may select one explicit source calendar day through `select_gpx_day`. That
separate selection seam retains only complete segments for the requested date, removes an exact
duplicate segment only once, and records source/exclusion counts for the private receipt. It never
sorts, bridges, or treats similarly shaped repeat visits as duplicates.

The selection also keeps its explicit UTC source start. `captured_at_route_seconds` converts only
explicitly zoned photo/video capture instants onto that clock; a timezone-free camera value or
filesystem time is rejected. The media anchor policy then decides whether the surrounding observed
GPS edge is short and credible enough to interpolate a location.

Route Story media anchors preserve their evidence: embedded GPS wins; timestamp-only media is
placed only along a short continuous recorded edge; gaps and out-of-range captures remain
unanchored. Private views and exports can retain the timeline decision while suppressing exact
coordinates.

The storyboard is a deterministic mapping from output frames to the original recording clock.
Photo holds freeze journey time; moving clips can advance it. This keeps the map, media cuts, and
telemetry consistent whether a journey is previewed or rendered at final speed.

Renderer frame telemetry resolves only within observed GPS segments. A gap has no fabricated marker,
speed, elevation, or grade, which keeps map transitions truthful.

`route_story.dashboard.dashboard_state_at` is the shared display model for preview and export. It
adds cumulative traveled/remaining/total distance, completion, observed elapsed/moving time, ascent,
descent, and moving average to current source-clock telemetry. It reports an unavailable state at a
segment gap and never creates an edge between recordings.

Route Story satellite openings are frame-addressable camera contracts. An `overview` scene freezes
the recording clock while it moves from a real U.S. geographic overview to the approved visible
route start; preview and export share the same saved camera poses. Imagery remains a separately
prepared local asset and is never fetched per rendered frame.

`SatelliteRaster` accepts only a readable, credited local image with declared geographic bounds.
When supplied with an opening, both frame previews and `render_video` use its deterministic camera
crop. It is an offline source seam: bundle preparation must provide the NASA, Landsat, NAIP, or
other approved derivative and provenance before an export starts. Its private render receipt stores
the source checksum, dimensions, geographic bounds, and credit without exposing a local path.
`prepare` copies an approved image into a content-addressed local cache, so identical imagery is
reused between projects and rendering never re-downloads or rewrites the original.

The film CLI also accepts an optional `--labels-json` manifest containing reviewed public names
(`name`, `latitude`, `longitude`, `kind`, and optional `max_span_lon`). Labels are selected by
viewport scale and priority, deduplicated, and laid out with collision avoidance. This keeps towns,
parks, water, trailheads, cliffs, and other useful context readable without a dense label cloud or
network requests during export; the receipt records the names and kinds included.

`route_story.film.render_route_story_film` builds a private H.264 story from approved local rasters,
the selected source-clock track, and original photos or video (map projection lives in
`route_story.layers`, overlay/typography in `route_story.hud`, and frame encoding/assembly in
`route_story.assembly`). It uses one eased geographic camera
through the U.S. overview, any supplied regional raster levels, and the route follow; local assets are
crossfaded only while that camera continues moving. Its compact stats rail reserves the map field for
the route marker. The export derives speed, elevation, grade, distance, ascent, moving average and
elapsed time from a quality-qualified display route: discontinuities and implausible jumps are excluded
from presentation while the original GPX remains preserved in the receipt. RAW stills use the measured
photo developer, and a crop occurs only for a continuous, near-digital-black perimeter—never by filling
or generating pixels. Photos with a short
credible timestamp interpolation or embedded coordinates appear at that route moment. Unanchored
stills remain visible as explicitly unlocated memories, and an unlocated video is introduced as an
interlude rather than pinned to invented map coordinates. It fetches no tiles or external assets
during rendering.

For a private local export, run `python -m gomoveshift.pipeline.route_story.film --help` to supply
the reviewed GPX day, original media directory, credited national/local rasters and optional credited
regional raster levels, and output paths. The
command writes the optional receipt beside the requested MP4 and does not upload the media.

Route Story projects are durable private drafts. They fingerprint the selected GPX and media under
approved source roots and retain the reviewed storyboard recipe for preview, retries, and export.
The console exposes private draft creation and listing at `/route-stories`; creating a draft does
not start a renderer or expose it to a public surface. A draft with exactly one timed GPX source
also exposes its source-clock telemetry at `/route-stories/{id}/telemetry` for preview inspection.
When a private draft includes one approved GPX, one approved local satellite image, and an opening
recipe with geographic bounds and attribution, `/route-stories/{id}/opening` exposes the opening's
start/middle/end camera poses and imagery receipt without rendering or using the network.
The console rechecks each reviewed source hash before telemetry or opening inspection and refuses
a source changed after draft creation.

## Shared GPU lock

A host can have several consumers of one GPU: this pipeline's `nvenc` encodes,
CI runners, and local LLM inference. A drive run takes an exclusive lock at
`GMS_GPU_LOCK_PATH` right after its run locks and holds it for the whole run —
the encode stages are spread across day masters, shorts and OST passes, and a
lock dropped between them would let a model load into the gap. Set
`GMS_GPU_LOCK_PATH` to the empty string to disable the lock on a host that
shares the GPU with nothing else.

Other programs interoperate through the **file**, never an import:
`fcntl.flock(LOCK_EX)` is the mutex, the JSON body (`owner`, `pid`,
`acquired`) is diagnostics, and **release never unlinks** — a lock file that is
deleted and re-created is two inodes, so a releasing holder could unlink a path
a waiting holder had already opened and both would then "hold" it.

A drive run *waits* for a live holder (an inference turn is tens of seconds;
failing a multi-hour encode over that would be absurd) and then fails rather
than proceeding, because the failure it prevents — an `nvenc` OOM partway
through a day master — is worse than a clear refusal at the start. The error
names the holder and its PID. To clear it by hand, stop the process that holds
the lock (the PID is in the error message and in the lock file's JSON body).

## Future-publish coordination — `publish-journal` cron pattern

The drive pipeline supports `--publish <future-date>` (uploads to YouTube as `private` with `publishAt=<date>` so videos flip public automatically at that time). After upload the run **auto-publishes the journal by default** (resolving the journey from `config.drive.active_journey_by_region` or `--auto-publish-journal SLUG`, site path from `--site-path` or `GMS_SITE_PATH`; `--no-auto-publish-journal` disables). It date-gates each entry on `youtube_uploads[lf].publish_at`: already-public entries are written to the journey immediately; future entries land in a `pending_journal.json` sidecar in the **drive output dir** (`<output_base>/<project>/`), so the live site never embeds a private video. `GMS_SITE_PATH` has no default: with neither `--site-path` nor `GMS_SITE_PATH` the journal step reports "not configured" and names the variable.

The companion `--replay-pending` flushes ready entries; combine with `--auto-commit --auto-push` to also commit + push the change to the site checkout. Designed for cron so you can run the pipeline once with a future publish date and the site updates itself without further intervention.

### Setup — crontab

Edit `crontab -e` as the user that owns the checkout and add:

```cron
# Hourly sweep: flush pending journal entries whose publishAt has fired,
# commit + push the site checkout. Cron sends mail on non-zero exit so
# auto-commit failures (dirty WT, push rejected, etc.) surface immediately.
17 * * * * /path/to/framegrade/.venv/bin/gomoveshift \
    publish-journal --replay-pending \
    --scan-root /path/to/output \
    --apply --auto-commit --auto-push \
    >> /path/to/publish-journal-replay.log 2>&1
```

Point `--scan-root` at the drive **output base** (`GMS_OUTPUT_DIR`, default `<data home>/output`) — the auto-publish hook writes each run's sidecar to `<output_base>/<project>/pending_journal.json`, which the recursive scan finds. The `17 * * * *` schedule (17 past the hour, hourly) gives a ~1h max delay between YouTube's `publishAt` firing and the site update — adjust the cadence to taste.

### Exit codes

- `0` — success (with or without entries flushed)
- `1` — no pending sidecars found at the given paths (usage error)
- `2` — flag combination error (e.g. `--auto-push` without `--auto-commit`)
- `3` — `--auto-commit` was requested AND one or more failures occurred (dirty WT, push rejected, journey outside git, etc.). The sidecar(s) survive so a clean re-run picks up the still-pending entries.

### What gets committed

One commit per repo (not per journey) when multiple journey files in the same repo flush in the same sweep. Commit message follows:

```
chore(journeys): publish-journal flush (N entries)

Files: content/journeys/<slug-1>.md, content/journeys/<slug-2>.md
Triggered by: gomoveshift publish-journal --replay-pending --apply --auto-commit --auto-push
Sweep time (UTC): 2030-01-01T12:00:00.000000+00:00
```

### What the sweep refuses to touch

- A repo with **uncommitted changes to tracked files**: skip-and-log. The sidecar entries belonging to that repo stay pending; a clean re-run flushes them. Untracked files do not count as dirty.
- A journey file **outside any git repo** when `--auto-commit` is set: warn and skip. (Useful guard during ad-hoc testing.)
- A push that fails: no retry, no rebase — exits non-zero so cron alerting fires. Resolve manually (`git pull --rebase && git push` in the site dir) before the next sweep.

## Tools

- `tools/enable_gpu_detection.sh [VENV]` — repair a venv whose GPU onnxruntime build was clobbered by the CPU
  wheel: uninstalls `onnxruntime`, force-reinstalls the pinned `onnxruntime-gpu==1.22.0` (the same pin as
  `pyproject.toml`; CUDA 12) plus the `nvidia-*-cu12` runtime wheels, and verifies `CUDAExecutionProvider`
  loads. A fresh `pip install -e .` already gets those wheels from the pin's `[cuda,cudnn]` extras. Never install the CPU `onnxruntime` wheel into the pipeline venv; `tests/test_dependency_pins.py`
  keeps it out of the dependency spec.
- `tools/retitle_worklist.py` — CSV worklist of live videos needing a retitle:
  duplicate/`Highlights from …`/county-only/`To X` titles. Reads `videos.yaml` + optional
  `--rss PATH` (no network); `suggested_title` uses only structured origin/destination/region/place fields, else `needs-human`.

### Private night projects

Photos → Night projects provides resumable, private redevelopment of saved linear
stacks through the same CLI/console backend. See [setup, commands and current
boundaries](docs/night-projects.md). It does not publish or imply image-quality approval.

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party data sources and their
licences are listed in [docs/route-story-sources.md](docs/route-story-sources.md).
