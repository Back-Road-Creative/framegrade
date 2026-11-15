### Added

- Long-form YouTube titles explore 1 upload in 4 (JP 2026-11-15; Waypost `title-policy-3`,
  pinned to #115's merge 12b57c5). The pre-render approval hands Waypost the baseline title as
  a stable `explore_key`; on the uploads Waypost picks, its least-measured renderable
  alternative ships with reason `exploration`, so alternatives finally earn impressions of their
  own. The explored title passes the same retained-route gate before the OST burn, and Phase D
  reuses that decision. Shorts never explore.

### Fixed

- The final-text review approved only the baseline title, so any other chosen long-form title
  (an explored one, or a future measured winner) was refused as `title_changed_after_approval`.
  It now approves the pre-render decision's chosen title.
- The title endpoint parser read `"Ash County Drive: Alder to Dogwood in 4K"` as
  `"Ash County Drive: Alder"` to `"Dogwood in 4K"`, refusing it as `TITLE-EVIDENCE E`. It now
  skips the `"{region} Drive: "` head and the `" in 4K"` tail.
