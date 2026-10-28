### Added

- `--short-reframe center|tracked`, `--output-cadence fixed|source` and `--insert-still-motion none|push_in` on the drive command: opt-in switches for the tracked vertical reframe, source-rate output and the insert still push-in. Each defaults to today's behaviour, passes nothing extra to the encode when absent, and is remembered across a resume. `--output-cadence source` refuses a source rate it cannot keep (anything but 29.97, 30, 59.94 or 60 fps) rather than resampling it.
