### Fixed

- A released fix no longer sits pinned but uninstalled while renders run back to back: `health backlog` (a new drive or a confirmed OST resume) and `health publish` now refuse, as `health heal` already did, while the service venv lacks the version the code root pins, so the staging sync's deferred install lands between renders.
- `health check` raises one `install_deferred` incident when that pinned version has still not been installed after 6 hours.
