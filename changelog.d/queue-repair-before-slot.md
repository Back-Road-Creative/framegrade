### Fixed

- A queued YouTube Short the final-text review would refuse (a pre-filter tag, a route record with no `bins`) used to fail at its slot and never be refilled. `dispatch-queue` now repairs it first with the existing tag filter and route re-measure, writing the sidecar only when the review then passes, and prints `QUEUE_REPAIR <id> <project> repaired|still_refused:<reason>|ok`.
- The health check gains the `queued_short_refused` rule: a queued Short still refused within 48 h of its slot raises an incident.
