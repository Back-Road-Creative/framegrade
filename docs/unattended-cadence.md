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
