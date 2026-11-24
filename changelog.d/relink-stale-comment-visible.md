### Fixed

- A first comment found live with a replaced text is counted (`stale`, `first_comment_stale=N`
  on the summary line), filed as a `first_comment_stale` health incident (`gomoveshift status
  --json`) and, on YouTube, corrected in place on the next pass (edit, or delete a duplicate);
  the incident clears itself. Meta drains count and file it too (their correction: see
  `relink-meta-correct.md`).
- The YouTube scan reads every bounded page and reports a replaced text live beside the new one,
  and a concurrent drain's comment is reported. Relink lists rewritten published rows for the
  drain to verify.
