## Shared heavy-photo admission

Supported `photos astro`, `astro-combine` and `astro-inspect` CLI operations hold
`pipeline.media_admission.media_admission` across their entire heavy handler,
including child workers and export. This uses the original night worker lock:
`GMS_NIGHT_ADMISSION_LOCK`, or the OS-user `gomoveshift-night-UID.lock` in the
system temporary directory. Contention exits 2 before decoding; an unsafe lock
path fails closed. Existing RAM/disk checks and decode-worker degradation still
run inside the fence. No lock is unlinked, and process death releases admission.
Trails, timelapses, triage and from-master inspection use the same astro dispatch;
help and independent metadata-only helpers stay outside it. `astro-inspect` is
read-only for its source but decodes full pixels and therefore participates.
The fence follows the CLI handler lifetime. Abrupt parent death releases it;
it does not reserve capacity for any orphaned converter or encoder descendant.
Existing child kill/wait behavior remains unchanged. This is cooperating-process
exclusion, not a RAM reservation or a host-wide CI/drive scheduler. Cross-container workers need the same explicit host-mounted lock
and host-provisioned access. Tests use real locks in a private directory per pytest
worker so parallel test processes cannot contend with operational media jobs.
