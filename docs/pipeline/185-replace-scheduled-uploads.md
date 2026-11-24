## `health backlog --replace-scheduled`: replace a drive's still-scheduled uploads

Part of `health backlog` ([156-health-check.md](156-health-check.md)); `--supersede` is described there.

**Replace scheduled uploads.** `--supersede <project> --reason <text> --replace-scheduled` also
accepts an output whose `youtube_uploads.json` holds records, when every record is either already
public (`publish_at` in the past: its LF is kept) or still scheduled at least 6 h ahead
(`REPLACE_MARGIN_H`) and confirmed live as private with a future `publishAt` through the YouTube
client (`videos.list`). A record that is inside the margin, no longer scheduled, unreadable, or
that the live check cannot decide refuses everything; an output with nothing scheduled is refused
as nothing to replace; without the flag the refusal is unchanged. The project's own store row may
stay while it is not `claimed`; a `claimed` row is refused, and its content-ledger entry no longer
blocks. A `complete` row (a published drive: its LFs are booked with video ids) is reopened by
Waypost's `ScheduleStore.reopen_for_replace`, never `register` (which raises for `complete`):
the launch first calls it with `dry_run=True` (expected video ids per LF, the replaced LFs, the
sha256 of `replace_plan.json`, `--reason`, the new output dir), BEFORE any rename, and a reason it
returns refuses the launch with nothing moved; after the renames and the audit-only launch it
calls it for real, and a refusal there is logged (`REPLACE-REOPEN-REFUSED`) and exits non-zero. A
`complete` row with no replace plan is still refused, and so is an installed `seo_manager` that
lacks `reopen_for_replace` ("repin"); there is no fallback to `register`. The plan is durable: `superseded.json`
becomes `backlog-supersede/2` (`/1` still reads) with a `replace` object (`kept` and `replace`
records: old video id, `publish_at`, title, playlist), and the same plan is written into the fresh
output dir as `replace_plan.json`. Phase D reads it: a kept LF is put back as uploaded and never
re-uploaded; a replaced LF has its old video re-checked live, and if still scheduled the new video
is uploaded with the old `publish_at` and title, and only after the upload succeeds and is booked
are the ledger entry re-pointed at the new id and the old id deleted (`videos.delete`). If the old
video went public meanwhile it is kept, nothing is uploaded, and `replace_plan.json` records
`kept_live`. A delete that fails is logged as `REPLACE-OLD-UNDELETED`. The new video's playlist
comes from the normal metadata path, not the old record's, and shorts follow their normal per-LF
Drive upload; neither is changed by the flag.
Two checks keep the replace consistent. `health publish` refuses a drive whose output dir holds a
`replace_plan.json` when the render's long-form count differs from the plan's total (kept plus
replaced): a different split would upload other content under the old titles and slots, and the
store's `lf_count` is left unchanged. Phase D is launched with `--lf-publish-date` = the first
claimed slot, which for a replace is the kept, already-public LF1; the launch anchor gate
(`_launch_anchor_refusal`) accepts that past date, keeping only the weekday check, but only when
the output dir's `replace_plan.json` keeps an LF1 whose `publish_at` local date (America/New_York)
equals the anchor. Without a plan, or with another date, a past anchor is refused as before.
The upload, swap and delete sequence is documented in `gomoveshift/pipeline/replace_scheduled.py`.
