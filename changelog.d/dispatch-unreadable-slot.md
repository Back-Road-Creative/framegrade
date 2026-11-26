### Fixed

- One delivery whose slot was written with no time zone, or is not a time at all, could end a whole drain pass. A zoneless slot is now read as UTC, and an unreadable one is held with an incident while the other deliveries go out.
