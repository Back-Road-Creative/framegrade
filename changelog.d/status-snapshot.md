### Added

- `gomoveshift status [--json]`: one read-only snapshot of running and recent jobs, open incidents, the publishing queue by platform, the schedule runway, the drive backlog, free disk space, the GPU lock and the cron logs. Each section says where and when it read; a store it cannot read says `unavailable` with the reason instead of looking empty. New settings: `GMS_JOBS_DIR`, `GMS_CRON_LOGS_DIR`, `GMS_STATUS_VOLUMES`, `GMS_STATUS_GPU_LOCK_PATH` (docs/pipeline/179-status.md).
