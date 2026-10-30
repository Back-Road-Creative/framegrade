### Fixed

- A drive recorded the day after a journaled trip ends now joins that trip's journey: `extend_adjacent_journey` extends `trip.end_date` (and `duration_days`) by one day and adds the day stub, instead of the drive hook creating a separate `draft: true` journey. Ace Basin (2026-02-21) had been split off `sc-2026` this way. A trip starting the day after is not extended backwards, and two adjoining trips are refused.
