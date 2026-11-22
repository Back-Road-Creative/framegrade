### Fixed

- A private relink during a post queues a delete of the live comment, never an edit to empty text.
- A Graph 100/33 "gone" answer for a comment that still reads is not trusted: the delete is
  confirmed with a GET, so an Instagram replace never posts beside a live dead link.
- A private relink of a published row reports a possibly live OLD comment; a posted row's note says
  to re-add the comment by hand. `id_map` rejects a non-str id or time.
- `drop_private_edit` is a compare-and-set with a retry; a failing Meta/YouTube correction pass is
  printed by `dispatch-queue`; an Instagram delete-then-post failure counts toward the attempt cap
  and keeps a `first_comment_repost_failed` incident open.
