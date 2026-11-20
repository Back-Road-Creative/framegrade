### Fixed

- A Short the final-text review would refuse at dispatch is now refused at enqueue: it is left
  out of the queue (all platforms), raises one `short_review_refused` health incident naming
  the reason, and the plan's other Shorts still enqueue. `enqueue-plan` lists the refusals on a
  dry run and on `--apply`. Before, it sat queued, burnt its attempts and lost its slot.
- The plan builder drops a Short's unreviewable YouTube tags (`reviewable_short_tags`, the
  `sync-plan-tags` filter) before the plan hash is taken, so a Short whose only fault was a tag
  now passes.
