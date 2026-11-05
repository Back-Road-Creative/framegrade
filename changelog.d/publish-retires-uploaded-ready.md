### Fixed

- A `ready` drive already uploaded outside a claim (Beaufort: its receipt under the run dir, its booking under `src`) sat first in Waypost's claim line, so `health publish` refused every newer drive in the region. `health publish` and the hourly `health check` reconcile now retire such a drive from its receipts through Waypost's `retire_published` (booking adopted, drive `complete`); a receipt Waypost cannot book leaves it `ready`.
