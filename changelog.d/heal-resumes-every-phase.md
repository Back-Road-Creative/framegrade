### Fixed

- `health heal` no longer relaunches a failed drive with `--phase <letter>` (the drive CLI's single-phase mode). A drive that failed in phase A or B/C used to pass that one phase and then fail the DoD gate (`lf_count 0 < 1`) because the later phases never ran. Heal now runs the plain audit-only resume, which runs every phase the checkpoint does not hold, and no longer refuses a failure whose phase it cannot map.
- A backlog drive job's `drive_failed` incident now gets its resume recipe, phase, gate and error line. The drive check reads the job's own `output.log` beside its `job.json` as well as the `expect` list, which for a backlog job is an artifact glob rather than a log, so heal no longer refuses "no readable resume.json for this incident".
