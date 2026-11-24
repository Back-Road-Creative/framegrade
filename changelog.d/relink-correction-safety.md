### Fixed

- A stale first-comment correction never deletes the intended comment and touches only our own
  comments (fails closed with no own id); a current-text comment counts whoever wrote it.
- Corrections have their own counter and cap (a 403 `forbidden` counts), a 404 delete resolves, a
  404 edit reposts, and a capped one stays on the summary line (`first_comment_correction_capped=N`).
- A second replace rewrites a pending correction's text.
