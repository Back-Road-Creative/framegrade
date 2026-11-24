### Fixed

- A replaced first comment found live on Facebook is edited in place (`POST /{comment-id}`
  `message=`, else deleted and reposted); on Instagram, which has no comment edit, it is deleted
  and the intended text posted only once the delete is confirmed, never two comments and never a
  post before the delete. Both run at the start of the next Meta drain.
- The Graph scan reads every bounded page when a replaced text exists, prefers the new comment
  and reports a replaced one live beside it for deletion, never the new one.
- `relink_replaced_lf`: the lease pre-check uses the process clock, and a failing other-project
  report is recorded, not raised. Public `post_first_comment` raises for a replaced text.
