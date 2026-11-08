### Fixed

- When phase_bc can't build a drive plan, its error now gives `build_drive_plan`'s own reason. One example: "kept (stop-masked) footage 2889s is under full_drive_part_min_sec=3600s". The old message said "despite complete inputs ... Re-run with --fresh", but a fresh run hits the same refusal. The real cause appeared only in an earlier WARNING line. The old wording stays only for a failed build that recorded no reason.
