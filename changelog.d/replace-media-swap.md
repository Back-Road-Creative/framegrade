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
- The replace swap keeps a finished swap and a confirmed Gate C/D across reruns: the crash restore
  undoes only an unfinished move (file still old, row queued and unuploaded), the swap's own sidecar
  is left out of the Gate C/D hash it checks, a published sibling refuses the Short for good, and a
  Short held by another day's missing attestation is deferred with that cause. A refusal incident
  resolves once the Short's rows are all cancelled or published, and a sqlite error in the
  media-stall sync is reported in the drain's errors instead of escaping.
- The media-stall threshold follows the drain's `backoff_*` arguments and its incident closes only
  when the row uploaded or went terminal.
