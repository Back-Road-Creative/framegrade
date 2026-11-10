### Fixed

- `health backlog` reads a `health heal` resume of a backlog drive (`gms-heal-gms-backlog-<project>-<sha7>`) as that drive's newest audit run. Before, only the original `gms-backlog-<project>` job counted, so a drive that heal resumed to success still refused its region every tick ("audit failed", exit 2), naming the old failed run.
