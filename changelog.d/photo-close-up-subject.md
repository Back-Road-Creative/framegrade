### Added

- Photo subject recognition gains `close_up`: a frame focused within 3 m (about 10 ft), or shot on a macro lens or at 0.25x or more implied magnification, reads as a close-up (0.65, 0.75 with a macro signal). It ranks after the astro rules and before wildlife, so a long lens focused near is a close-up, not wildlife. `CaptureOptics` carries `focus_distance_m` and `lens_model`, and `recognise_photos` accepts an `optics` mapping to supply them. Operator subject sidecars accept `close_up`; look routing is unchanged.
