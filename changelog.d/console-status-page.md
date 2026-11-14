### Added

- Console: a read-only Waypost page at `/status` (first tab) and `/status.json`, drawn from the same snapshot as `gomoveshift status`. It lists what needs attention, what is rendering, the runway, publishing by platform, the backlog and the machine, refreshes in place every 30 s with a Pause updates button, and says `unavailable: <reason>` for any store it cannot read (docs/pipeline/179-status.md).
