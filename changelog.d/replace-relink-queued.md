### Added

- `relink_replaced_lf` rewrites a replaced long-form upload's old id in unposted queue first
  comments, the publish plan, the pending journal and the operator checklist.

### Fixed

- First-comment drains re-read the text before posting and mark it posted only if unchanged.

- A first comment found live with a replaced text (a relink rewrote the row mid-post, or the
  response was lost) is recorded as posted with that text, `first_comment_error` set and the
  drain's `errors` filled, never marked done silently; Meta drains no longer post a new text
  beside a live old one. Relink reports a rewritten plan (re-confirm Gate C/D) and other
  projects' rows naming an old id.
