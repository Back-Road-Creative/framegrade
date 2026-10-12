### Tenant storage stamp (`console/tenant.py`)

Isolation between console instances is path separation (own `seo.storage_path`,
`tests/test_publishing_brand_isolation.py`), but the image bakes that path identically for every
brand (`docker/console-config-override.yaml`) -- only the HOST side of each brand's bind mount
(the value each instance gives `GMS_SEO_STORAGE_DIR`) is meant to differ, and two brands sharing one
host directory look identical from inside either container. `_lifespan` (`console/app.py`) calls
`tenant.claim_storage(...)` before anything else, strictly before `jobs.sweep_orphaned_jobs`, so a
refused brand never first mutates the jobs table. It claims BOTH of the instance's storage
directories: the delivery-queue dir above, and `console_data_dir()` -- `jobs.sqlite3`, the job logs
and each brand's saved settings, bound at `/data` for every brand by the same design, so two brands
on one host directory shared every launch, log and setting while the queue stamp reported clean. `claim_storage` writes/checks a
`.waypost-tenant` stamp inside the storage dir: absent -> this brand claims it; matches -> no-op;
a different or unreadable/empty stamp -> `TenantMismatchError` naming both brands and the
host-side env var to fix.

Migration: an existing deployment with populated storage and no stamp is claimed by whichever
instance starts first; if two brands already share one host directory, the second to (re)start
after this lands refuses rather than silently keeping the first brand's queue -- the bug being
caught, not a regression.

**The compose contract is asserted, not just the runtime refusal** (`tests/test_console_compose.py`). The stamp makes a collapsed path a loud startup failure rather than a silent leak, which is the upstream gate. The compose test pins the rest: the queue mount is found by its container target rather than by a substring (the previous check, `"seo" in volumes_text`, was satisfied by the credentials mount alone); that target is a fixed literal; and its host side is a required variable with no default, so compose refuses to render when it is unset. A second tenant is a second instance with its own value for that variable. These read the YAML directly instead of `docker compose config`, so they cannot skip on a host
without docker.
