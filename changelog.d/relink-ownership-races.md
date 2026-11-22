### Fixed

- A viewer's comment that says our first-comment text is never marked as our posted comment and
  never edited or deleted (YouTube and Meta): the scans flag it (`ForeignCommentId`), prefer our
  own match, and the drain posts ours beside it.
- A queued delete of a stale first comment re-reads that one comment first: one that now says an
  intended text is kept, and a failed read leaves the delete pending.
- A second copy of the current text found live beside the recorded comment gets a delete (new
  column `deliveries.first_comment_live_id`); the recorded comment is never deleted.
