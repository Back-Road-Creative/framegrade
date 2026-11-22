### Fixed

- A removed first comment (deleted, no text to repost) is no longer waiting on a hand step that
  never closed it: the drain lists the post's comments, adopts our own full-drive comment when it
  is live again (a hand re-add by the channel account), records it as the live id and closes the
  `first_comment_removed` incident. A viewer's or authorless copy is never adopted; a failed or
  cut-short listing keeps the incident open with an error line. The incident text, the drain's
  error line and the docs now say the pipeline watches for our comment.
- A restored row gets a full post budget again (`first_comment_attempts` resets to 0).
- A read-back that fails or answers gone right after a Meta post is reported (error line and
  `first_comment_own_id_mismatch` incident, nothing deleted), and that incident closes itself once
  a read-back shows our own id as the author.
