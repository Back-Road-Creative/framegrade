### Changed

- `drive` (`full_drive`): a day whose kept footage is under the full-drive floor
  (`full_drive_part_min_sec`, or `full_drive_short_day_min_sec` when lower) is now planned as a
  shorter `best_moments` long-form from the same scores and stop mask, within that day, instead of
  failing the drive with `InsufficientFootageError`. It is logged
  `day <d>: kept <n>s under <floor_name>=<floor>s -> best_moments long-form` and recorded as
  `lf_mode`/`lf_mode_reason` on that day's LF entry in `drive_plan.json`. The Phase A GPS-gap gate
  and the pre-ingest day check now refuse only a day under the 480 s minimum long-form, and
  `InsufficientFootageError` is raised only when best_moments cannot reach 480 s either. A failed
  drive blocked on this error relaunches from source once this version installs (its failure
  fingerprint changes with the installed version).
