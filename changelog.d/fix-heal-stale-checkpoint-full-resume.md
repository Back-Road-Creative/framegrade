### Fixed

- `gomoveshift health heal` re-audits a drive whose checkpoint was stamped by an older phase code version (for example after `PHASE_BC_CODE_VERSION` moved to 4) as a full audit-only resume instead of `--phase D`, which stopped at the prerequisite check. Renders are reused by fingerprint; the attempt records `stale_phases`.
