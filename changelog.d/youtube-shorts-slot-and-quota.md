### Fixed

- A YouTube Short drained at or after its slot was uploaded private with a `publishAt` already in the past; it now goes up public at once (a future slot is still scheduled), and a Short still not public an hour after its go-public time opens a `youtube_not_public_after_slot` health incident.
