### Fixed

- A drive whose `health publish` job ended without uploading was left `failed` in the schedule
  store, and `health publish` refused anything not `auditing`, so nothing could publish it again.
  `health publish` now reopens a `failed` drive with no upload receipt to `auditing` and claims a
  slot again; every other check still applies.
