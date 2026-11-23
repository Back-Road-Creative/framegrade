### Added

- `rebuild_replaced_sidecar`: a replace re-render rebuilds the shorts plan onto the new media or
  holds the Short with a `replace_rebuild_refused` incident in `status --json`. It swaps only when
  Gate C/D still holds, the old and new audits name the same source segment, and the Short's own
  recording day is attested; the per-day attestations are placed beside the sidecar so a real drain
  uploads it. Uploaded or recovery siblings refuse the Short; a live lease, in-flight row or a
  future slot inside the 24 h lead time defers it; a past or missing slot moves. New required
  argument `old_stages_dir`. Not wired into `replace_scheduled` yet.
- A delivery stuck on `TransientMediaError` past the backoff ceiling opens a
  `delivery_media_stalled` incident; it keeps retrying and the incident closes after it uploads.
