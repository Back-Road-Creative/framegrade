### Fixed

- The Gate C/D confirmation hash covers `media`, `reviewed_plan_hash` and `consent_hash` again for
  `enqueue-plan` and the auto-pass; a confirmation also records a swap-invariant hash that only the
  replace swap accepts, and only for a plan that proves to be a swap of the old one.
- A Short refused for a missing `recording_date` stays in the written plan (null date), so dispatch
  holds its stale bytes instead of burning its rows, and its incident stays open until it moves.
- A finished replace swap is checked against the media on disk: a changed file or a forged sha is
  refused with an incident. A verified attestation beside the plan is kept over an unverified stage
  copy and reported in `RebuildResult.attestations_kept`.
