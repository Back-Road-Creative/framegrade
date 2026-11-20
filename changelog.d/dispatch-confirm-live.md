### Added

- Each `dispatch-queue --apply` tick now confirms scheduled YouTube Shorts that went public: it runs `confirm_scheduled_live` after dispatch, recording `platform_live` and the real go-public time, and logs `CONFIRMED_LIVE count=<n>`. A failure there only logs a warning and is retried next tick.
