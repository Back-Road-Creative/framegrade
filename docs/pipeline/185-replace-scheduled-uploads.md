## `health backlog --replace-scheduled`: replace a drive's still-scheduled uploads

Part of `health backlog` ([156-health-check.md](156-health-check.md)); `--supersede` is described there.

**Replace scheduled uploads.** `--supersede <project> --reason <text> --replace-scheduled` also
accepts an output whose `youtube_uploads.json` holds records, when every record is either already
public (`publish_at` in the past: its LF is kept) or still scheduled at least 6 h ahead
(`REPLACE_MARGIN_H`) and confirmed live as private with a future `publishAt` through the YouTube
client (`videos.list`). A record that is inside the margin, no longer scheduled, unreadable, or
that the live check cannot decide refuses everything; an output with nothing scheduled is refused
as nothing to replace; without the flag the refusal is unchanged. The project's own store row may
stay only while it is not `claimed`/`complete` (Waypost cannot reopen those, so such a row is
refused), and its content-ledger entry no longer blocks. The plan is durable: `superseded.json`
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
