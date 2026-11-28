### Fixed

- A Facebook or Instagram reel upload that keeps making progress is no longer cut off with "The write operation timed out". The whole video went out in one write that had to finish within the 10 s connect bound, so a reel on a slower uplink failed with the error deliveries 384 (2026-11-22) and 374 (2026-11-27) recorded. The bytes now go out in 16 KB blocks, and each block gets the full bound.
- An Instagram upload that fails on every fresh container now records every try's error, not only the last one.
