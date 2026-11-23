### Fixed

- A delete a gone reopen queued now runs first in its row's correction pass and outside the
  correction cap, so a row at the cap or with a failing sibling item never leaves two copies of
  ours live; a failure there counts no attempt, and a row holding only such deletes is not called
  capped.
- A capped row whose listing fails, is cut short, or shows our copy with other text no longer
  retries forever or says "not listed": past the 24-hour wait the delete is dropped with an error
  line and one `first_comment_lag_delete_dropped` incident naming every id and the reason (a
  copy with other text is never deleted; a copy appearing later needs a hand delete).
- Adopting a capped row resets its correction attempts; a held reopened delete no longer delays
  restoring a removed row.
- A reopened delete past its 24-hour deadline whose read fails or whose delete the platform
  refuses is dropped with an error line and a `first_comment_lag_delete_dropped` incident naming
  the id and the last failure, instead of being retried every drain for ever. Several expired
  held deletes on one row raise one incident naming every id, and the incident detail leads with
  the ids so a real Meta id is never cut off.
