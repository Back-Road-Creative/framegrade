### Fixed

- Phase A now refuses a full_drive day that the GPS-gap cut alone leaves under `full_drive_part_min_sec`, before the GPU scoring pass. The planner refuses that day anyway, since the stop mask can only cut more, but it used to do so only after scoring had run. On the 03-04 drive the new check gives the planner's own 2889 s against the 3600 s floor. A GPS-less day is never refused by this check.
