### Fixed

- `health backlog` now retires and holds a rowless dead drive even while a drive is running. Before this, heal's resume of that same dead drive started just before each hourly backlog run, so the backlog stopped at "a drive is running" every hour. The hold was never written, and the next mirror SHA brought the same resume again, so neither region's backlog ever launched. The text output now also prints the `retired` and `held` lines, so the cron log shows them.
