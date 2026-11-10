### Fixed

- Night stacks no longer refuse real dark-sky frames as "lens cap on". Stars covering under 0.1% of a frame left the cap-on test reading it as flat noise, so every frame of six night groups on the 2026-11-09 card was refused ("0 usable of 19"). A would-be cap-on frame is now re-checked for stars after a 3x3 median filter that removes hot pixels; any that survive mean it saw sky.
