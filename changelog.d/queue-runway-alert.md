### Fixed

- `health check` now raises one `social_queue_gap` incident per platform naming the first day within the next 42 days with nothing scheduled in the social queue, so a thin stretch weeks ahead alarms while there is still time to render and enqueue Shorts for it, instead of days before.
