## Waypost retained-region dependency

GMS pins Waypost to `f879603d0275ff001d7e630b9fa3796a54718b6d`, the commit of
release tag `v0.1.0`, which contains Waypost #18's bounded Census region review, #19's separate
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
