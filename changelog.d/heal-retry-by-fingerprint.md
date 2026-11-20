### Fixed

- `health heal` no longer relaunches a failed drive because the mirror gained an unrelated commit. Each attempt records a failure fingerprint (installed `gomoveshift-video`, `seo_manager` and `redactcam` versions, the drive's manifest row, the drive config, the failed phase) and a class (`deterministic`, `transient`, `publish_only`, `unknown`). Deterministic failures wait for the fingerprint to change, transient ones retry up to 3 times, unknown ones once, and publish-only ones are refused with a recorded reason.
