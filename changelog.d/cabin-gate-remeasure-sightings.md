### Changed

- The cabin gate's second pass after a repair now re-measures the first pass's recorded sightings against the repaired mask (reading only the mask at those frames) instead of re-decoding and re-detecting the whole source. A full pass records every sighting beside the mask (`*_cabin_sightings.json`) under the cabin signature minus the mask fingerprint; any missing, partial, unreadable or mismatching record (signature, mask frame count) falls back to a full pass and logs why. The verdict and leak report are the same a full pass would give.
