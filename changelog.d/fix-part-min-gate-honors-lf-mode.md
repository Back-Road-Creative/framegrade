### Fixed

- The before-ingest raw-footage floor (`full_drive_part_min_sec`) now applies only when the resolved long-form mode is `full_drive`. An operator-chosen `--lf-mode best_moments` shadows the part bounds, so a short day in that mode is no longer refused for a planner that never runs.
