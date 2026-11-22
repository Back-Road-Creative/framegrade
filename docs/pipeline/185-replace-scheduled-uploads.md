## `health backlog --replace-scheduled`: replace a drive's still-scheduled uploads

Part of `health backlog` ([156-health-check.md](156-health-check.md)); `--supersede` is described there.

**Replace scheduled uploads.** `--supersede <project> --reason <text> --replace-scheduled` also
accepts an output whose `youtube_uploads.json` holds records, when every record is either already
public (`publish_at` in the past: its LF is kept) or still scheduled at least 6 h ahead
(`REPLACE_MARGIN_H`) and confirmed live as private with a future `publishAt` through the YouTube
client (`videos.list`). A record that is inside the margin, no longer scheduled, unreadable, or
that the live check cannot decide refuses everything; an output with nothing scheduled is refused
as nothing to replace; without the flag the refusal is unchanged. The project's own store row must
be `complete` (`ready`, `auditing`, `failed`, `claimed` or absent is refused with the reason), and its
content-ledger entry no longer blocks. A `complete` row (a published drive: its LFs are booked with video ids) is reopened by
Waypost's `ScheduleStore.reopen_for_replace`, never `register` (which raises for `complete`):
the launch first calls it with `dry_run=True` (expected video ids per LF, the replaced LFs, the
sha256 of `replace_plan.json`, `--reason`, the new output dir), BEFORE any rename, and a reason it
returns refuses the launch with nothing moved (a `--dry-run` asks it too, so it reports that
refusal instead of `would_launch`); after the renames and the audit-only launch it calls it for
real with a fresh clock. Every check that can refuse runs before a rename; a blocked preflight or a
failed launcher after the renames puts the stage root and output dir back (a retry plans again
from the old `youtube_uploads.json`). A refusal of the real reopen cannot be undone (the audit is
already running on the fresh dirs): it is logged (`REPLACE-REOPEN-REFUSED`), opens an incident and
exits non-zero. A
`complete` row with no replace plan is still refused, and so is an installed `seo_manager` that
lacks `reopen_for_replace` ("repin"); there is no fallback to `register`. The plan is durable: `superseded.json`
becomes `backlog-supersede/2` (`/1` still reads) with a `replace` object (`kept` and `replace`
records: old video id, `publish_at`, title, playlist), and the same plan is written into the fresh
output dir as `replace_plan.json`. Phase D reads it: a kept LF is put back as uploaded and never
re-uploaded; a replaced LF has its old video re-checked live, and if still scheduled the new video
is uploaded private under the NEW render's gated title (`require_title_decision` unchanged; the
old and new titles are recorded in `replace_plan.json`) and later given the old `publish_at`, and the order is fixed: upload private, recheck the old live, delete the old (`videos.delete`), schedule the
new at the old `publish_at`, swap the ledger entry to the new id (title and slot too), then book the store. If the old
video went public meanwhile it is kept, nothing is uploaded, and `replace_plan.json` records
`kept_live`. A delete that fails is logged as `REPLACE-OLD-UNDELETED`; a 404 on it counts as deleted. The new
id is written to the plan the moment `videos.insert` returns. The recheck before the delete decides
three ways: the old is provably public or too close, so the new is deleted and the LF abandoned;
the lookup errored (including a quota error), so nothing is deleted, an incident opens and the next
entry retries; the old is already gone, so the replace continues. The new video is deleted only by a
call that has just proven the old one exists. After the old is deleted the slot counts as
passed only when it is within `REPLACE_SLOT_PASSED_MARGIN_MIN` (5 min) of now; a slot an hour ahead is
scheduled as planned. A truly passed slot is rebooked automatically at Waypost's next free slot
(`seo_slots.next_free_slot` over the same ledger and `schedule.sqlite3`, no `publishAt` in the past,
never two videos in a slot) with a `slot_rebooked` incident that resolves at `done`; if nothing is free
the new stays private in `slot_passed` with an incident and every later entry retries it. Thumbnails go only on a new video that is scheduled or done, once. A plan run
is exempt from the Drive-manifests rule (it queues no shorts), its quota projection counts only the
plan LFs still to insert, and Phase D fails with `REPLACE-UNFINISHED` plus a `replace_scheduled_partial`
incident while any plan LF is neither `done` nor `abandoned`. Every replace refusal or partial
failure (`REPLACE-REOPEN-REFUSED`, `REPLACE-OLD-UNDELETED`, an abandoned LF, a failed schedule, a
title-gate refusal, a restore that failed, a lookup error on the old video, and a publish claim
closed on a replace with an unfinished LF, a deferred pre-upload check) opens a
`replace_scheduled_partial` incident per LF, shown by `gomoveshift status --json`; each closes itself
when its LF reaches `done` or `abandoned` (`abandoned` and `new_undeleted` stay open on an abandoned
LF), and the drive-level ones when nothing is left. Nothing waits for a person: `health check` closes
a claim whose plan has an unfinished LF `failed` (an abandoned LF is terminal, so that closes
`complete`), and `health backlog` republishes a failed drive, which re-enters Phase D and resumes each LF
from its state, spaced by the heal retry backoff (1 h doubling, capped at 24 h). The new video is filed in
the old record's `playlist_id`, falling back to the metadata-resolved playlist when the old record has none.
`upload_if_needed` flushes the checkpoint right after `videos.insert` on both of its branches, so the new
id is in the plan even though a replace passes no `config`. Shorts follow their normal per-LF Drive
upload, unchanged by the flag. The real reopen of the store row runs after preflight and BEFORE the
launch (a locked database is retried): a refusal or a failed launch puts the old run back (a failed launch
also gives the row back `complete` from the set-aside receipts, else `retire_uploaded` closes it), writes
`<output-root>/.replace-retry/<project>.json` and opens `reopen_refused` / `replace_retry` incidents.
The marker records its region and is written BEFORE the renames (a crash mid-replace still leaves it);
`due_replace_retry` returns only a marker of the tick's region (a marker without one is read through the
backlog row's region). The next `health backlog` tick picks it up (heal backoff) and repeats
`--replace-scheduled` itself through `launch.run_tick`, which decides what happens to the marker: the replace's
OWN launch (the result names the marker's project and carries `supersede`) clears it, while any other launch in
that call (an auditing drive's publish, a heal resume) is returned as it is and leaves the marker, with no bump and
no fall-through, for the next tick; a raise after the replace launched never re-marks it nor runs the plain tick; a PERMANENT refusal (nothing to replace, every record public or kept inside the margin, the row
gone from the region's backlog rows (checked before the next-eligible-row refusal), nothing to supersede, no upload
record) clears it and resolves the `replace_retry` incident with an
`outcome` naming the reason; any other refusal is TRANSIENT and bumps it, so the backoff advances. Either
way the same tick then runs the plain backlog, so one stuck marker never blocks the region's next row, its
failed-drive relaunch or its failed-audit path. A crash between the renames and the plan write leaves no output dir and a set-aside run with the records:
the retry plans from that run (`_superseded_uploads`), never reads it as "nothing to supersede"; a raise in
`_supersede` itself puts the run back and retries. `_supersede` writes each `superseded.json` into the dir
BEFORE renaming it, so a dir under `_superseded` always carries its record whenever the process dies. A raise
before a move leaves the old run untouched: `_unsupersede` finds no set-aside dir, removes only the record it wrote
(`cleanup` reads a record in the project dir, so none is left in a live run) and never renames or removes that
run, so the next tick still plans from its `youtube_uploads.json`; a record left by a death is overwritten by the retry. The set-aside fallback takes only `<project>-<YYYYMMDDTHHMMSSZ>` dirs whose record names that
project, so `day-a` never reads `day-a-long`'s run. Before the launcher runs the marker is stamped
`launching_at`; if the process dies after the launcher returned ok, the next tick finds the
`gms-backlog-<project>` job created since the stamp, clears the marker and never replaces the fresh run a
second time (no such job, or another project's: the stamp is ignored and the replace retries). A launcher that
reports a failure (a timeout) after its job was written counts as launched: the fresh run is never put back. The
`margin_kept` incidents are recorded before the marker is cleared, also by that stamp branch (from the fresh dir's
`replace_plan.json`), one row per LF however often both run. Any launch or publish in the marker
tick is noted as it succeeds, so a raise after it returns that launch and never re-marks or falls through. A
marker with no region whose manifest row is gone can never be claimed: the tick gives it up like a permanent
refusal, recording the `replace_retry` incident (opened if none is) resolved with outcome
`given up: no row in any region` before clearing it, and lists it as `given_up_replace_retry`. `status --json`
lists every incident resolved in the last 7 days as `incidents.recently_resolved` (`project`, `kind`, `code`,
`outcome`, `resolved_at`), so a given-up replace names its project and why; `open` and the `resolved` count stay. Audit lookups
on the replace path (retiring a dropped row's failed audit, publishing a failed row) apply the plan's `since`.
Audit lookups for a reopened row ignore jobs created before
the plan's `created_at`, so the set-aside run's passed audit never publishes the fresh directory. An LF whose slot is inside the 6 h margin is kept (its old
video publishes on its own; `margin_kept` incident) and the rest are replaced. A slot that passed is
rebooked and RESERVED (ledger entry and Waypost row, `SlotTaken` searches again) before any YouTube call and
re-verified on re-entry, so two LFs never share a date. A failed replace drive whose manifest row was removed
is still published again, never given up. The quality gate runs beside the replace check, so a quality failure is not hidden by it.
Two checks keep the replace consistent. `health publish` refuses a drive whose output dir holds a
`replace_plan.json` when the render's long-form count differs from the plan's total (kept plus
replaced): a different split would upload other content under the old titles and slots, and the
store's `lf_count` is left unchanged. Phase D is launched with `--lf-publish-date` = the first
claimed slot, which for a replace is the kept, already-public LF1; the launch anchor gate
(`_launch_anchor_refusal`) accepts that past date, keeping only the weekday check, but only when
the output dir's `replace_plan.json` keeps an LF1 whose `publish_at` local date (America/New_York)
equals the anchor. Without a plan, or with another date, a past anchor is refused as before.
The upload, swap and delete sequence is documented in `gomoveshift/pipeline/replace_scheduled.py`.
