### Fixed

- After the queue had been down, the next drain posted every overdue Facebook and Instagram row at once. A drain pass now sends at most one delivery per platform, on every platform; the rest wait for the next pass.
