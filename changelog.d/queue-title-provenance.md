### Fixed

- The nightly queue text sync no longer rewrites a Short title someone edited by hand in the plan. The plan now records the exact title the generator wrote, and only a title still equal to it is regenerated, whatever the edited title looks like.
- A Short queued with a title-body experiment arm the generator no longer renders is now left as stored, instead of being refused, with an incident filed, every night.
