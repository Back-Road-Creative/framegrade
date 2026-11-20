### Fixed

- A cabin-leak repair now always adds a mask box covering the gate's own leaked box for every leaked frame, so a repair given a leak can no longer add nothing and waste the retry.
- The single repair attempt is now up to `drive.blur_cabin_repair_rounds` rounds (default 2), each followed by a full gate pass. The encode runs only on a pass; after the last failed round the gate error is raised as before.
- A failed repair is recorded in the cabin verdict cache under the same signature. A heal retry on identical inputs raises the recorded failure at once instead of repeating the repair; a changed mask, model, config or renderer repairs again.
