### Fixed

- The privacy blur no longer blurs signs, logos and lettering that the plate
  model boxes at plate shape and size (a SPAR logo on a fuel sign, a yield sign).
  A plate now blurs only once its center sits inside a vehicle box, from the
  vehicle model at `blur_plate_vehicle_conf` (0.6) or a tracked vehicle. Set it
  to 0 to blur every plate box as before. redactcam is pinned to the commit
  that adds `plate_vehicle_conf`. Renders and cached masks made before this keep
  the old blur until they are re-rendered.
