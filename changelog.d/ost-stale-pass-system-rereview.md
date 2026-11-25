### Fixed

- A day whose on-screen text changed after it passed the OST gate (including Shorts re-picked onto new moments) is now gated like a new day: a fresh `--ost-system-review` record that agrees on the exact new text, refusing nothing, passes it. Before, the stale pass blocked even a fully agreeing review, so an unattended heal parked at the gate for good. Without the opt-in, or with any refused or disagreeing claim, the day still waits for a person, and nothing from the earlier pass carries over. Applies to the per-day, front-load and Shorts gates; superlong is unchanged.
