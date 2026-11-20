### Added

- Stage roots with no backlog row get an owner record with an expiry (`gomoveshift/health/stage_owners.json`; `mizen-remeasure` held for framegrade#398). `health backlog` appends an expired root to the host-cleanup release list, whose file name is now one constant (`stage_owners.STAGES_RELEASE_LIST_NAME`, pinned by a test), and prints roots that have neither a row nor a record.
- A backlog launch refused for lack of staging or output space opens a `backlog_storage_refused` health incident (pushed once); a later tick that launches or finds the space resolves it.
