### Fixed

- A row parked at `needs_operator` can now be marked published from the console when a person finds the post live or posts it by hand; before, mark-published took TikTok manual rows only, so such a row could only be cancelled or retried.
- A row parked at `needs_operator` off a dead worker's lease is now counted in the drain summary's `needs_operator=<n>`; before, the interrupted-lease reconcile parked it without counting it.
