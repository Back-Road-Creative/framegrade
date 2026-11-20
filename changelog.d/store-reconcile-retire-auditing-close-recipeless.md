### Fixed

- `retire_uploaded` also retires an `auditing` drive that holds an upload receipt (ace-basin: published, still `auditing`); never without a receipt.
- The backlog publish retry treats a publish job that ended `cancelled`, `stalled` or `killed` with no exit code as failed, so a failed row with a passed audit and no receipt is published again (Kerry-to-Cork). It retries the publish, never the render.
- `health heal` closes a recipe-less incident whose job is a publish or heal job, with a recorded `resolution_reason`, instead of refusing it every tick. A heal of a heal reuses the base drive name instead of nesting `gms-heal-` names.
