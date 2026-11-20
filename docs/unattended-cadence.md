# Unattended cadence

Where GoMoveShift's publish slots come from, and what each region's days and times are.

## One source for slots

Every slot decision (next free slot, is this anchor on the cadence, is this date free)
is made by `seo_manager.scheduling.slots` in the `waypost` project. GMS asks and
executes; it never decides.

- `gomoveshift/pipeline/seo_slots.py` is the adapter. It turns the loaded config into a
  `Cadence` and the content ledger (plus `schedule.sqlite3` beside it, when it exists)
  into the occupancy the slot functions read. It makes no slot decision.
- Thin wrappers keep their old signatures: `content_ledger.next_free_slot`,
  `schedule_report.build_schedule_report`, Rule 8 (`quality_gates`), Rule 2b
  (`state_invariants`), Phase D's LF 2..N guard.
- The dates of LF 2..N come from seo_manager's series rule,
  `seo_manager.scheduling.slots.lf_series_dates` (LF n = LF1 + 7 * (n - 1), same weekday;
  `seo_slots.lf_series` is the GMS call). The drive scheduler and Phase D's LF 2..N guard
  call it and add no days of their own; the series claim books the same dates.
- `cmd_drive` refuses a bad `--lf-publish-date`, `--start-date` (or a saved
  `lf_publish_date` on resume) at launch,
  before any phase: off the run's LF day, past its slot time (today counts once the
  slot time has passed), or booked by another project. The gate creates no directories.
  A saved anchor of a drive that already has a live LF (saved `youtube_uploads` holds a
  video id) gets only the weekday check, like Phase D's resumed run, so a failed LF2
  retry after LF1 aired is not told to re-anchor. An audit-only run (`--skip-upload
  --no-youtube-upload`, a nightly runner's argv) books nothing and gets the same
  weekday-only check, so a past saved anchor does not fail its backlog row every night.
  The message names the next free date over the same allowed days, or says there is no
  LF day when `--publish` is not one of the configured LF days. A date without leading
  zeros is refused as not `YYYY-MM-DD`.
- An LF is never inserted with a publish time that has passed. The launch gate above
  checks the date once; Phase D checks each not-yet-uploaded LF's slot again right
  before its `videos.insert`, because a long encode (or the shorts upload) can outlast
  the date the run started with. When nothing of the run is live yet, Phase D drops the
  passed date and asks seo_manager for the next free slot, the same call a fresh run
  makes, and logs `LF_SLOT_RERESOLVED` with the old and new dates. When an LF of the run
  is already live, it keeps its real date and each pending LF whose slot has passed gets
  seo_manager's next free slot on the run's LF day, behind the LFs before it (the run's
  own bookings are not rivals; its unconfirmed `schedule.sqlite3` rows move to the new
  dates). A later resume keeps those dates. The run stops with `LF_SLOT_PASSED` before
  the insert only when that cannot be done safely: no `--publish` day, a live LF with no
  readable publish time, or a new slot that has passed too.
- A confirmed LF upload is booked in the ledger and mirrored into `schedule.sqlite3`
  when that file exists (never created by GMS), even when the ledger write failed (`LEDGER-UNBOOKED`). A failed mirror logs
  `SCHEDULE-STORE-UNBOOKED` and never fails the upload.
- `tests/test_scheduling_single_source.py` fails on new weekday arithmetic outside an
  allowlist of documented holdouts, and on `timedelta` built from the LF interval (or a
  literal 7 days / weeks) in the wrapper files.

## Documented holdout: the superlong day

`schedule_constants.resolve_superlong_publish_day` and
`content_ledger.next_free_superlong_slot` stay in GMS. seo_manager's `superlong_day` is
one value; GMS resolves the day per region and per related part. They still ask the
`next_free_slot` wrapper, so occupancy is the shared one. Superlong parts are booked in the
ledger only (no `schedule.sqlite3` row).

## Other weekday arithmetic still in GMS

Not slot decisions: `console/pubqueue_calendar.py` (week anchor for the queue view),
`pipeline/social_generator.py` (weekday word in post copy), `tracking/slot_performance.py`
(analytics over past slots).

The single-video stage (`pipeline/stages/schedule.py`) and the ingest batch
(`pipeline/ingest/batch_runner.py`) take their default publish date from
`seo_slots.default_publish_slot` (never today, ledger-booked dates skipped), not from
weekday arithmetic of their own. The drive upload path has no fallback: a missing `lf_slots` stops
the upload (`external_systems_coordinator._upload_lfs_to_youtube`).

## Per-region cadence

- **TikTok** shorts are written to the configured TikTok pack directory (default under the data home) for
  manual scheduling in TikTok Studio (`docs/social-publishing-tiktok.md` § Pack mode).
- **Ireland** — `--lf-day sun --region ireland`: LF ~14:00 EDT; shorts 08:00 + 12:00
  (never 16:00, USA slot).
- **USA** — `--lf-day tue` (LF ~16:00) / `--lf-day fri` (LF ~15:00) `--region usa`;
  shorts 14:00 + 18:00.

**Waypost owns the days.** The tenant `brand.yaml` (`seo: cadence:`) is the SEED; the effective
cadence is that seed overlaid with Waypost's `lf_days` decision rows (the decision store beside the
SEO storage, `seo.storage_path`). The scheduled publisher, `seo_slots.cadence_for(config)`, the launch
gate and `gomoveshift schedule` all read that one cadence (`seo_slots.waypost_cadence`), so a decided
day moves them together and a replaced day stops being offered. A launch with no `--lf-publish-date`
is also checked: `--publish`/`--lf-day` must be one of the region's Waypost days or the launch
refuses and names them. An unreadable decision store halts planning with
`CADENCE_SOURCE_UNREADABLE reason=... (planning halted)` rather than falling back to a guess.
`tests/test_brand_seo_cadence.py` pins the seed's day times to `config.drive.publish_slots`.

## Drive backlog

`drive_backlog.yaml` lists the drives still to audit, in order: one row per recording day of
raw GoPro chapters on D:. It is tenant data, so it lives in the operator's inputs directory (`GMS_INPUTS_DIR`) at
`gomoveshift/console/brands/gomoveshift/drive_backlog.yaml` and is read through
`gomoveshift.paths.drive_backlog_path()`. Its
header names the re-cuts already public and the footage left out. A row for footage with no
GPS and no track carries the operator's labels (`per_lf_locations`, `location`) so the
launcher hands them to the drive; nothing is typed at launch time.
`tests/test_drive_backlog_manifest.py` checks that the consumer resolves to it, that it
parses, and that no chapter is queued twice.

Consumers: `gomoveshift health check` rule 2b (`lf_slot_uncovered`) pages when a region's
earliest free LF slot is within 7 days and the region has no ready drive, naming the
manifest's next row for that region (or saying none is left). `seo-manager schedule
backlog-next --manifest <path>` prints the same row. The hourly `bin/gms-health-cron.sh` tick (workspaces)
runs `gomoveshift health backlog --region R` per region, which launches the next eligible row as
an audit-only drive for any free slot inside the render horizon (the wider of 7 days and
`lead_hours`, 70 days in the packaged `brand.yaml`), so rows render back to back
(`docs/pipeline/156-health-check.md`). Uploading that drive is `gomoveshift health publish <project> --region R` (approve,
claim a Waypost slot, launch Phase D), not a hand-launched `drive --youtube-upload`. The page reaches the operator only while the
`gms-health` cron job is enabled on the host.

The tick reads the framegrade release list (the version-pin check, `git ls-remote --tags`) at most
once per 24 h (`launch.LATEST_VERSION_TTL`): the answer is cached in `<data root>/release-latest.json`
and reused on a network failure while it is under 7 days old; with no usable cache a failure still
refuses the launch ("cannot determine the latest framegrade release"). `RETIRED_CONFIG_KEY` logs
once per process per key. Not built: launching in one region while heal resumes a drive in another.
Each drive's staging fills its own device, so that overlap is unsafe until B6's storage check
(`storage_plan`) is the precondition for it; it stays off until measured.

### Stage roots with no backlog row

A stage root kept on purpose with no `drive_backlog.yaml` row (today: `mizen-remeasure`, held as
evidence for framegrade#398) needs an owner record in `gomoveshift/health/stage_owners.json`
(`gms-stage-owners/1`): `name`, `owner`, `reason`, `issue`, `expires` (an ISO date). A record
missing a field, with a bad date or a repeated name is refused, never guessed past. The day after
`expires`, `health backlog` appends the name to `<data root>/host-cleanup/gms-stages-release.list`
(`stage_owners.STAGES_RELEASE_LIST_NAME`, with a comment naming the issue and owner); removal is
the host-cleanup rule 7c, which keeps every hold it has (running spool job, `.keep`, live lease,
process inside). The file name is the one `_ops/host-cleanup.sh` reads (`GMS_STAGES_RELEASE`); a
test pins the literal. Extending a hold means editing `expires` in the same PR that cites the
issue. A root with no row and no record is printed as `unowned stage root` on every tick.
