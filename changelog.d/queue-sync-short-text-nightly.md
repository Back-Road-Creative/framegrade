### Added

- `gomoveshift health sync-short-text` runs the Short text sync (`queue sync-short-text --apply`) unattended and files a `short_text_sync_refused:<project>` or `short_text_sync_crash` health incident for a refusal or a crash, resolves it when a run is clean, and pushes it once. Schedule it after the nightly `metrics collect`.

### Fixed

- `queue sync-short-text` is idempotent across sidecars: sidecars are planned in earliest-slot order (ties by path), each seeing one pre-run read of the queue's text (published Shorts of a project with a queued Short included) plus the final text of the sidecars before it, and an earlier sidecar's new text is checked against later sidecars' final text, so two queued projects that share a place no longer flip description leads on every apply. The title ledger and nightly title snapshot are read once per run, and a refusal re-checks the neighbour-place and batch rules on the final text. After a failed write the remaining sidecars are refused for that run. The nightly entry point files an incident only for refusals that wait for a person.
