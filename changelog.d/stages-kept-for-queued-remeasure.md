### Fixed

- A project's stages are kept while a queued YouTube Short of it may still need a route re-measure (its route record has no `bins`, or its sidecar cannot be read): the expired owner-record release listing skips that root, and `gomoveshift cleanup` keeps its `gps_points_<date>.json` / `shorts_ost_audit_<date>.json` and never removes it as abandoned. An unreadable queue releases nothing. Before, Early Clonakilty's queued Shorts lost those files on 2026-11-22 and were refused at every tick with no alert.
- `health check` rule 17 files one `queued_short_stages_gone` incident per project whose queued Short needs a re-measure while those files are already gone, saying a supersede re-render is needed.
