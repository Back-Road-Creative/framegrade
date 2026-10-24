## Waypost retained-region dependency

GMS pins Waypost to `5da82077b15e55adc4dfee9d82c2ae672fb06a17` (main after #46, which adds the
backlog `day_gpx` and `camera_utc_offset_hours` fields `health backlog` reads). It contains everything in
release tag `v0.1.0`: Waypost #18's bounded Census region review, #19's separate
region-only long-form title policy, and #20's millisecond comparison of proof
gap lengths (GMS rebuilds a gap end as start + length, so an exact-fit gap
carried float noise and every bounded proof was refused).
`tests/test_waypost_region_pin.py` exercises these contracts against the
installed package. The prior pin lacked the
boundary-review version, so GMS refused a polygon fallback before encoding.

The pin alone does not choose a title. GMS must still prove the retained route,
give the exact approved county or state as a Waypost fact, persist the title
decision before encoding, and recheck the same claim at publication. The
dependency does not authorize a render or upload on its own.
