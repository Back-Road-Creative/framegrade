### Fixed

- A comment a Meta read-back answers gone right after the post is recorded as posted and
  re-read on the next drain; only a second confirmed gone reposts it. One lagging read can no
  longer leave a second live copy, and a gone-every-time comment stops at the attempts cap with
  the repost incident open. Reopening a gone row queues the old id as our own extra delete. A
  gone route that cannot move the row (a correction is pending) is said: an error line, a
  WARNING and an incident detail that names why.
- A removed row adopts our current link by video id: identical hand re-adds adopt the lowest
  id and queue the rest as our own extra deletes, comments naming different videos stay
  ambiguous, and when the plan's sidecar names the current video only that link is adopted.
  The replaced-link check compares video ids (a different emoji prefix is the same link), and
  our own copy of a replaced link is queued for deletion after an adopt.
- The removed incident leads with the exact re-add text so the detail cap never cuts it; the
  own-id incident and the `_author_mismatch` docstring no longer claim the drain reposts on one
  gone read.
- The private-relink notes (`replace_relink`) no longer present a hand re-add as the only way
  back: the drain restores the comment by itself once the plan gives the row text.
