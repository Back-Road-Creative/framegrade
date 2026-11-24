### Fixed

- A viewer's comment that says our first-comment text is never marked as our posted comment and
  never edited or deleted (YouTube and Meta): the scans flag it (`ForeignCommentId`), prefer our
  own match, and the drain posts ours beside it.
- A queued delete of a stale first comment re-reads that one comment first: one that now says an
  intended text is kept, and a failed read leaves the delete pending.
- A second copy of the current text found live beside the recorded comment gets a delete (new
  column `deliveries.first_comment_live_id`); the recorded comment is never deleted.
- Only a comment positively ours (author = our channel, Page or IG user id) is ever deleted,
  edited or adopted; a comment with no author is a viewer's for every purpose (logged once).
- A queued second-copy delete re-reads its comment and runs only on one marked ours that is not the
  recorded comment; two own copies on Facebook / Instagram end as one; an old row with no
  recorded comment id reports a second copy instead of ignoring it.
