### Fixed

- The delete a gone reopen queues for the old comment id survives a lagging read: it is kept and
  re-read every drain for 24 hours (`REOPEN_LAG_WINDOW`, an assumed bound) and dropped past that
  with an error line and a `first_comment_lag_delete_dropped` incident. A row that reaches the
  repost cap while copies lag is listed each drain; the lowest id is adopted and every other
  identical own copy is queued for deletion, so exactly one stays live and the row records it.
- The repost-with-gone and own-id-with-gone incident details say what happens; an unroutable
  gone no longer promises a retry for a capped correction; the copy an adopt keeps is the lowest
  id by number for Meta ids (`123_99` before `123_100`).
