### Fixed

- `health publish` books a Waypost slot for every long-form a drive rendered. It refused any drive the store did not list with exactly one long-form, and `health backlog` lists every drive with one, so a two-long-form drive would have got one booked slot while Phase D dated its second long-form a week later with nothing booked. The count now comes from the audit's `drive_plan.json` (`regular_lfs`); a count that differs from the store is written back before the claim, and a drive with no readable plan is refused before any write.
