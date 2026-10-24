# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Back-Road-Creative/framegrade/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Back-Road-Creative/framegrade/releases/tag/v0.3.0
