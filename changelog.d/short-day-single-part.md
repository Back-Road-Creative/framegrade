### Added

- `full_drive_short_day_min_sec` (drive config, optional, `0 < value <= full_drive_part_min_sec`, inert unless both part bounds are set): a day whose kept footage is in `[short_day_min, part_min)` ships as one long-form instead of being refused. The sample `config.yaml` sets 2700, so a 45-59 min day ships as one part. The planner, the Phase-A GPS-gap pre-gate and the preflight/backlog refusal all use the same lower floor; a day under it is still refused, and footage is never padded, repeated or joined to another day.
