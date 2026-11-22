### Fixed

- A private relink during a post queues a delete of the live comment, never an edit to empty text.
- A Graph 100/33 "gone" answer for a comment that is still there is not trusted, so an Instagram
  replace never posts beside a live dead link.
- A private relink of a published row reports a possibly live OLD comment; a posted row's note says
  to re-add the comment by hand. `id_map` rejects a non-str id or time.
- `drop_private_edit` is a compare-and-set with a retry; a failing Meta/YouTube correction pass is
  printed by `dispatch-queue`; an Instagram delete-then-post failure counts toward the attempt cap
  and keeps a `first_comment_repost_failed` incident open.
- A private-relink delete leaves a `first_comment_removed` incident and an error line open until
  the row has comment text again (re-add by hand; the backfill never refills a posted row).
- A Graph 100/33 "gone" answer is confirmed by listing the post's comments, not by `GET /{id}`;
  still listed, a failed listing or no post to list leaves the delete pending.
- A replace (Instagram, or a refused Facebook edit) records the new comment as the row's live id,
  so a later scan never deletes it as a second copy.
