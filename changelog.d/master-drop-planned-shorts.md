### Fixed

- The per-day master drop's shorts integrity check compared against a contiguous `short_01..short_N` range, so a day whose shorts were dropped and refilled (gapped file numbers) reported "missing 5 short(s) (expected 21, found 21)" and kept a ~59 GB master. It now checks the day's own planned short file names; a genuinely absent planned short still fails closed.
