### Fixed

- A queued second-copy delete is never dropped silently: a Graph "gone" answer on its re-read is
  confirmed by listing the post (still listed keeps it pending); a copy that now says other text,
  or whose author is missing or not our account, is dropped with an error line.
- The comment a Meta drain just posted is read back; an author other than our own id raises a
  `first_comment_own_id_mismatch` incident and an error line, and the comment is never deleted.
  Whether Graph's `from.id` equals `page_id` / `ig_user_id` is unverified.
- `read_comment` returns the author too (`.author`), and the extra delete requires it to equal our
  own account id (`own_id()` on all three clients).
