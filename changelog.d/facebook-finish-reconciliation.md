### Fixed

- Checkpoint Facebook Reels finish before sending it, require literal confirmed
  success, and keep ambiguous results for read-only status reconciliation of the
  retained video ID. Queue recovery preserves unknown outcomes with backoff beyond
  normal retry limits instead of falsely declaring publication or permanent failure.
- Bind new uploads to their original Facebook Page before sending them. Legacy
  unknown accounts and changed credentials remain held without rewriting provenance.
