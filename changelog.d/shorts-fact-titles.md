### Changed

- Drive Short titles name the most specific place the clip measured, such as `Tryon` or
  `Polk County`, instead of a "Sep 14, 3rd stretch: 41 seconds on a Monday" stamp. The final-text
  review is unchanged, so no road appears in a title. A title already published or used in the
  batch takes the next measured place; only a Short with none left keeps its stamped title.
  The `yt_title_body` arm is no longer assigned to a fact title.
- Queue and bracket sync now re-derive fact titles: the ordinal comes from the day sidecar
  (then a verified stamp, then plan position), not from the old title, so a plan the
  generator wrote syncs to no change. Discarded siblings take no measured place, and Phase D
  shares one title ledger across a drive's LFs so a place titles one Short of the drive.
