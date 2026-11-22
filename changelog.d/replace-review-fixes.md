### Fixed

- `--replace-scheduled` review fixes: a blocked preflight or failed launcher after the renames
  puts the drive back so a retry can still plan; only a `complete` store row is replaced; the
  replacement goes up under the new render's gated title (old and new titles recorded in
  `replace_plan.json`); `--dry-run` asks Waypost and reports its refusal; the real reopen reads a
  fresh clock; a failed lookup of the old video is a distinct reason. Every replace refusal or
  partial failure, including a publish claim closed on kept receipts while a replaced LF is
  unfinished, opens a `replace_scheduled_partial` incident in `gomoveshift status --json`.
