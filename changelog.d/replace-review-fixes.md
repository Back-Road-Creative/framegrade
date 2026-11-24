### Fixed

- `--replace-scheduled` review fixes: a blocked preflight or failed launcher after the renames
  puts the drive back so a retry can still plan; only a `complete` store row is replaced; the
  replacement goes up under the new render's gated title (old and new titles recorded in
  `replace_plan.json`); `--dry-run` asks Waypost and reports its refusal; the real reopen reads a
  fresh clock; a failed lookup of the old video is a distinct reason. Every replace refusal or
  partial failure, including a publish claim closed on kept receipts while a replaced LF is
  unfinished, opens a `replace_scheduled_partial` incident in `gomoveshift status --json`.
- `--replace-scheduled` second review: a failed or errored recheck of the old video never deletes
  the new one (only a provably public or too-close old abandons; an error defers, a gone old
  continues, a 404 on its delete is success); a slot that passes mid-replace leaves the new video
  private with an incident instead of sending a past `publishAt`; Phase D plan runs skip the
  Drive-manifests rule, thumbnail only scheduled or done new videos, and fail with an incident while
  a plan LF is unfinished; the new id is saved the moment it is inserted; the quota projection counts
  only plan LFs still to insert; the fallback playlist filing is idempotent.
