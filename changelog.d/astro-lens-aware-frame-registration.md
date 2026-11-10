### Fixed

- Night stacks on sparse star fields now keep frames the old fit threw away. On the 2026-11-09 card, night groups with about 90 stars a frame stacked only 3-4 of their 15-19 good frames: the old 4-number fit (shift, turn, scale) could not follow the lens bending stars near the corners as the sky drifted, so the outer-corner check refused it. Those frames are now fitted with the lens-aware model chunk combining already uses, against the same checks, which are not loosened.
