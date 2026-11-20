### Fixed

- A queued YouTube Short the review still refuses inside 48 h of its slot no longer burns the slot: `dispatch-queue` hands the slot to the next ready Short (printing `QUEUE_SLOT <id> <old> -> <new> <reason>`) and moves the refused row after the last queued slot on the daily grid. It also requeues a `failed_permanent` review refusal that now passes, never JP's kept rows (98, 102, 106, 110, 114).
- The health check gains the `delivery_failed_permanent` rule: a new `failed_permanent` YouTube delivery raises an incident.
