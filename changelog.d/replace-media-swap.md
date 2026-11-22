### Added

- `rebuild_replaced_sidecar`: a replace re-render rebuilds the shorts plan onto the new media (fresh
  fingerprint, recomputed plan hash moved on queued rows) behind the identity, privacy and mask-frame
  checks, or holds the Short with a `replace_rebuild_refused` incident in `status --json`. Not wired
  into `replace_scheduled` yet; rebuild, then relink.
- A delivery stuck on `TransientMediaError` past the backoff ceiling opens a
  `delivery_media_stalled` incident; it keeps retrying and the incident closes after it uploads.
