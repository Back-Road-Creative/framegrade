### Fixed

- A scheduled YouTube Short now records the time YouTube confirmed it public (new source `platform_live`) and tracking `posted_at` follows it, instead of the upload moment or the slot. A row held past its slot by more than 2 hours raises a `delivery_held_past_slot` incident, and dispatch moves a second Short due in the same clock hour to the next free slot.
