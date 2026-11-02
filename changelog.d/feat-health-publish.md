### Added

- `health publish <project> --region R --stages-root S`: the single operator command that approves
  an audited drive in Waypost, claims its long-form slot and launches the Phase D upload with that
  slot's date. The hourly `health check` closes the claim when the job ends: complete when an upload
  receipt exists, failed (booking released) otherwise. Refuses while an older ready drive sits ahead,
  since Waypost claims a region's oldest ready drive. (#179)
