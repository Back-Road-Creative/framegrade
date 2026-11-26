### Fixed

- A YouTube Short drained at or after its slot was uploaded private with a `publishAt` already in the past; it now goes up public at once (a future slot is still scheduled), and a Short still not public an hour after its go-public time opens a `youtube_not_public_after_slot` health incident.
- A YouTube retry more than two hours after a lost-response upload no longer re-uploads the Short: the channel scan looks back to the row's first submit (`first_submitted_at`, a new queue column), and never adopts a video another queue row already owns.
