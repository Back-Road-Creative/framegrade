### Fixed

- `health heal`: a resume killed by a host restart (its job record stays "running" with a dead
  nanny) is now outcome `interrupted`, pushed once, and does not use the retry budget or the
  backoff, so the drive is relaunched on the same fingerprint instead of being refused as
  "1 of 1 attempts used". Interrupted relaunches are capped at
  `retry_policy.HEAL_INTERRUPTED_RETRIES` (3) per fingerprint; past that they count like a failure.
