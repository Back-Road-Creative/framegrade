### Fixed

- A YouTube Short no longer fails the pre-upload final-text review on its own
  generated title. The stamp ("Feb 24, 2nd stretch"), a weekday, "seconds" and
  the closing hashtags are now backed by the record (`recording_date`, the
  generator's tag lists, the clip's own region votes); any other word is still
  refused. The plan builder fills `reviewed_plan_hash` with the content hash of
  what the Short sends, and the review uses the same hash for plans written
  without one (`missing_field:approval_hash`).
- `gomoveshift queue recover-terminal` re-queues a YouTube `failed_permanent` row
  the final-text review refused (it never uploaded), once the review agrees with
  the Short as the sidecar would send it now, behind the same duplicate-scan,
  slot and privacy gates.
