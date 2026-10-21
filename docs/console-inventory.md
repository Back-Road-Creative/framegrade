# Console inventory

What every enabled console page and action is, what actually backs it, and how much of that was
shown to work. Recorded against the console code and a live console container.
`tests/test_console_inventory.py` fails when a route is added or removed without a row here.

## Status vocabulary

A GET returning 200 shows a page renders. It does not show the action behind it works.

| Status | Meaning |
|---|---|
| proven | A hermetic test exercises the handler **and** the live GET returned 200. Read-only pages only. |
| proven-test | A hermetic test exercises the handler. It was **not** executed against the live container (it would launch a job, write state or post to a platform). |
| blocked | The handler is fine but the container cannot do the work it launches (cause in Note). |
| unverified | No console test references the route, or its resource use is not established. |
| unavailable | Disabled for a brand (`disabled_tabs`). |

The Status column is for the bundled `gomoveshift` brand, which has no disabled tabs. A brand
that lists `drives` or `shorts` in `disabled_tabs` (`gomoveshift/console/brands/<brand>/brand.yaml`)
has those rows **unavailable**; every other row is unchanged.

## Routes

| Method | Path | Tab | Status | Live | Test evidence | Note |
|---|---|---|---|---|---|---|
| GET | `/healthz` | - | proven | 200 | test_console_app, test_console_auth | minimal body unauthenticated |
| GET | `/` | overview | proven | 200 | test_console_app |  |
| GET | `/jobs` | jobs | proven | 200 | test_console_jobs |  |
| POST | `/jobs` | jobs | proven-test | not executed | test_console_jobs | launches an allowlisted subcommand; generic form only |
| POST | `/jobs/drain` | jobs | proven-test | not executed | test_console_jobs |  |
| POST | `/jobs/schedules` | jobs | proven-test | not executed | test_console_jobs |  |
| POST | `/jobs/schedules/{schedule_id}/toggle` | jobs | proven-test | not executed | test_console_jobs |  |
| POST | `/jobs/schedules/{schedule_id}/delete` | jobs | proven-test | not executed | test_console_jobs |  |
| GET | `/jobs/{job_id}` | jobs | proven-test | not executed | test_console_interrupted_evidence |  |
| POST | `/jobs/{job_id}/cancel` | jobs | proven-test | not executed | test_console_host_executor | host-executor jobs only; writes the worker's cancel marker |
| POST | `/jobs/{job_id}/rerun` | jobs | proven-test | not executed | test_console_jobs, test_console_rerun_provenance |  |
| POST | `/route-stories/film` | route-stories | unverified | not executed | test_console_route_story_film | hermetic only; encode path references nvenc (pipeline/route_story/encode.py), so likely blocked in the container -- not run; owner: the console GPU decision (see Execution route) |
| GET | `/route-stories` | route-stories | proven | 200 | test_console_route_story |  |
| POST | `/route-stories` | route-stories | proven-test | not executed | test_console_route_story |  |
| GET | `/route-stories/profiles` | route-stories | proven | 200 | test_console_route_story_film |  |
| GET | `/route-stories/{project}/telemetry` | route-stories | proven-test | not executed | test_console_route_story |  |
| GET | `/route-stories/{project}/opening` | route-stories | proven-test | not executed | test_console_route_story |  |
| GET | `/library` | library | proven | 200 | test_console_library_pagination |  |
| POST | `/library/{catalog_id:path}/tags` | library | proven-test | not executed | test_console_library_search |  |
| GET | `/library/media/{catalog_id:path}` | library | proven-test | 404 for an unknown id (probe x) | test_console_library_asset_media | real media serving not checked live |
| GET | `/library/{catalog_id:path}` | library | proven-test | not executed | test_console_library_asset |  |
| GET | `/drives` | drives | proven | 200 | test_console_drives | hidden when the brand disables the tab (disabled_tabs) |
| POST | `/drives/run` | drives | blocked | not executed | test_console_drives, test_console_drives_form_recovery | launches `gomoveshift drive`; NVENC is required; one probe found it absent in the container (see Execution route) |
| GET | `/drives/preview` | drives | proven-test | not executed | test_console_drives |  |
| GET | `/drives/gates/pending` | drives | proven | 200 | test_console_drives | discovers only console-job-store jobs (jobs.py gate discovery); CLI/cron runs are invisible (see Known gaps) |
| GET | `/drives/gates` | drives | proven-test | 422 without params | test_console_gate_playback |  |
| GET | `/drives/gates/frame` | drives | proven-test | not executed | test_console_gate_playback |  |
| GET | `/drives/gates/video` | drives | proven-test | not executed | test_console_gate_playback |  |
| POST | `/drives/gates` | drives | proven-test | not executed | test_console_drives | writes the gate answer under <out>/_gates |
| GET | `/shorts` | shorts | proven | 200 | test_console_shorts | hidden when the brand disables the tab |
| POST | `/shorts/run` | shorts | blocked | not executed | test_console_shorts | drive-family subcommand; same NVENC dependency as /drives/run |
| GET | `/shorts/csv` | shorts | proven-test | 422 without params | test_console_shorts |  |
| GET | `/shorts/csv/download` | shorts | proven-test | not executed | test_console_shorts |  |
| GET | `/photos` | photos | proven | 200 | test_console_photos_capabilities |  |
| POST | `/photos/run` | photos | unverified | not executed | test_console_photos_develop_path_validation | GPU use not established; not run; owner: the console GPU decision (see Execution route) |
| GET | `/photos/night` | photos | proven | 200 | test_console_night | lists night projects |
| POST | `/photos/night/scan` | photos | proven-test | not executed | test_console_night | reads /night-source (read-only mount) |
| POST | `/photos/night` | photos | proven-test | not executed | test_console_night |  |
| POST | `/photos/night/products` | photos | unverified | not executed | none | no console test references this route; owner: a browser acceptance suite (not yet written) |
| GET | `/photos/night/{project}` | photos | proven-test | not executed | test_console_night |  |
| POST | `/photos/night/{project}/compare/{recipe}` | photos | unverified | not executed | none | no console test references this route; owner: a browser acceptance suite (not yet written) |
| GET | `/photos/night/{project}/product/{task_index}/{output_index}` | photos | unverified | not executed | none | no console test references this route; owner: a browser acceptance suite (not yet written) |
| POST | `/photos/night/{project}/export/{task_index}` | photos | unverified | not executed | none | no console test references this route; owner: a browser acceptance suite (not yet written) |
| GET | `/photos/night/{project}/export/{export_index}` | photos | unverified | not executed | none | no console test references this route; owner: a browser acceptance suite (not yet written) |
| POST | `/photos/night/{project}/pause` | photos | proven-test | not executed | test_console_night |  |
| POST | `/photos/night/{project}/cancel` | photos | proven-test | not executed | test_console_night |  |
| POST | `/photos/night/{project}/resume` | photos | proven-test | not executed | test_console_night |  |
| GET | `/photos/night/{project}/preview/{index}` | photos | proven-test | not executed | test_console_night |  |
| GET | `/seo` | seo | proven | 200 | test_console_seo |  |
| POST | `/seo/corrections` | seo | proven-test | not executed | test_console_last_forms_recovery |  |
| GET | `/comments` | seo | proven | 200 | test_console_seo |  |
| GET | `/journeys` | journeys | proven | 200 | test_console_journeys |  |
| POST | `/journeys/sync-site` | journeys | proven-test | not executed | test_console_journeys_site_path | site mounted read-only at /site in the container: a real sync cannot write |
| POST | `/journeys/publish-journal-replay` | journeys | proven-test | not executed | test_console_journeys | same read-only /site limit |
| GET | `/publishing` | publishing | proven | 200 | test_console_pubqueue |  |
| GET | `/publishing/calendar` | publishing | proven | 200 | test_console_pubqueue_calendar |  |
| GET | `/publishing/reconcile` | publishing | proven | 200 | test_console_reconcile | read-only reconciliation: states, coverage gaps, recovered vs unresolved failures |
| POST | `/publishing/pause` | publishing | proven-test | not executed | test_console_platform_pause |  |
| POST | `/publishing/override-state` | publishing | proven-test | not executed | test_console_override_state |  |
| GET | `/publishing/{delivery_id}` | publishing | proven-test | not executed | test_console_pubqueue |  |
| POST | `/publishing/enqueue` | publishing | proven-test | not executed | test_console_pubqueue |  |
| POST | `/publishing/enqueue-asset` | publishing | proven-test | not executed | test_console_enqueue_asset |  |
| POST | `/publishing/drain` | publishing | proven-test | not executed | test_console_pubqueue | a live drain would deliver to platforms; deliberately not run |
| POST | `/publishing/mark-published/{delivery_id}` | publishing | proven-test | not executed | test_console_pubqueue |  |
| POST | `/publishing/{delivery_id}/reschedule` | publishing | proven-test | not executed | test_console_delivery_reschedule |  |
| POST | `/publishing/{delivery_id}/{action}` | publishing | proven-test | not executed | test_console_pubqueue | cancel/retry-style actions |
| GET | `/settings` | settings | proven | 200 | test_console_settings_tab |  |
| POST | `/settings` | settings | proven-test | not executed | test_console_settings_tab |  |
| GET | `/schedule` | schedule | proven | 200 | test_console_journey_follows_you | page shows the schedule and the four-value read model (`console/schedule_model.py`) |
| GET | `/{slug}` | - | proven-test | 404 for /gomoveshift | test_console_app | catch-all |

Live column: HTTP status from `curl` inside the console container. "not executed" means no
live request was made.

## Execution route

- **Frontend**: FastAPI app `gomoveshift.console.app:create_app`, served by the console
  container, an image built from `docker/console.Dockerfile`. Runs as uid 999 (`console`) with
  group 1001.
- **Job launch**: the console never encodes in-process. `jobs.launch()` Popens
  `sys.executable -m gomoveshift <subcommand>` for an allowlisted subcommand (`jobs.py`
  `ALLOWED_SUBCOMMANDS`) **inside the same container**; the ticker (`GMS_CONSOLE_TICKER=1`) and
  restart policy are the supervisor (`GMS_SUPERVISED=1`).
- **Storage**: `console-data` volume at `/data` (jobs.sqlite3, job logs, night projects).
- **Media paths** (host = container, bind mounts in `docker/console-compose.yaml`):
  `GMS_STAGING_DIR` (the footage input directory), `GMS_OUTPUT_DIR`, `GMS_VID_OUTPUT_DIR` (gate
  answers are written under `<out>/_gates`), `/site` (site checkout, **read-only**) and
  `/night-source` (night photo source, read-only). SOPS/age secrets and SEO credentials are
  read-only mounts.
- **GPU/CPU policy**: the drive pipeline requires NVENC and forbids CPU fallback
  (`color_filters.detect_nvenc`). A host with an NVIDIA GPU still gives the
  container **no GPU device** unless it is passed through: `HostConfig.DeviceRequests` is null,
  runtime `runc`, no `/dev/nvidia*`. ffmpeg lists the `*_nvenc` encoders (compiled in) but
  `detect_nvenc()` run inside the container raised `NVENCUnavailableError` ("NVENC probe returned
  nonzero (returncode 255)"). **A console drive launch therefore cannot encode; it fails at the
  NVENC check.** Compose documents `--gpus all` as the opt-in and does not set it. Passing the GPU
  through, or keeping encodes on the host, is an operator decision; this inventory does not
  change it. The compose file also sets `GMS_GPU_LOCK_PATH` to a host path that does not exist
  inside the container.
- **Host executor** (`console/hostspool.py`, `jobs.launch`): with `GMS_HOST_SPOOL_DIR` set
  (compose binds the host spool at `/host-spool`), `drive` and `route-story-film` are never
  forked in the container. `launch()` writes `pending/<brand>-<job id>.json` (`job_id, brand,
  argv, dedupe_key, env, submitted_at`; `env` holds only the brand's registry overrides, never `os.environ`) and the job shows `queued`; the host worker
  (`docs/host-worker.md`) writes `<id>.status.json` and `heartbeat.json`, which the ticker/reap
  turns into `running` (with progress and phase on /jobs), `done`, `failed` (notifies),
  `cancelled`, or `interrupted` (worker silent past 60 s, or the request vanished; notifies).
  A launch is refused with a 409 and a plain reason when the spool is unwritable, the heartbeat
  is missing or stale, or the same argv for the same brand is already pending or running; it
  never falls back to a local Popen. `POST /jobs/<id>/cancel` writes `cancel/<id>`. The request
  carries no environment: brand setting overrides (`brand_launch_env`) are not passed to the
  worker, which runs with its own environment. The worker unit is installed by the host setup step.

## Known gaps

- Passing the GPU through to the console container stays a pending operator decision. An omitted
  GPU does not mean ffmpeg falls back to CPU encode: the drive fails at the NVENC check.
- The seven `unverified` rows each name an owner in their Note; none has been exercised.
- Gate/job discovery depends on the console job store only (`jobs.py`), so CLI and cron
  drives never appear on `/drives/gates/pending`.
- `/journeys/sync-site` and `publish-journal-replay` need to write the site; `/site` is mounted
  read-only, so a real sync from the container cannot write there.
