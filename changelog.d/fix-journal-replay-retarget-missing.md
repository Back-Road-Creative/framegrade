### Fixed

- `publish-journal --replay-pending` no longer strands a due entry whose journey file is gone: it flushes it into the journey that holds the entry's day (a trip range holding it, else the trip that ended the day before, extended on `--apply`), records the route there, and consumes the entry. Ace Basin's two long-forms had been deferred into a draft journey that should never have existed; once that draft was set aside, they would have stayed pending every hour. No match or an ambiguous day still keeps the entry pending.
