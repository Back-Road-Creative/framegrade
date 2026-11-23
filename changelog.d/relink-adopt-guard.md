### Fixed

- A removed first comment adopts only our CURRENT link: a live copy of a replaced link
  (`first_comment_prior_text`), the id of the comment we deleted, or one of several distinct
  candidates is never adopted (the error line names the ids), in the drain and in the adopt
  UPDATE. A refused adopt now says why. The removed incident and the docs give the exact text a
  hand re-add must be, with the real video id.
- A comment a read-back confirms gone (right after a Meta post, or the own-id recheck) is
  reposted instead of leaving zero comments live behind an own-id incident that never closed:
  the row is unposted (the try counts toward the attempts cap, so it cannot repost forever) under
  a repost incident, and an open own-id incident whose comment reads gone is resolved with a
  note. A failed recheck read adds an error line.
