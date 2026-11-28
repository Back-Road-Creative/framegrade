### Fixed

- A passed audit whose checkpoint went stale under new code (Old Head to Courtmacsherry after 0.3.185) was refused by `health publish` on every hourly tick with "re-audit it (health heal)", but heal only re-audits drives that have a `drive_failed` row and this one had none, so nothing ever re-audited it. The refusal now files that row itself (`drive_failed:<drive>:recipe`), and the next hourly heal re-audits the drive.
