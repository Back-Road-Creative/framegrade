### Fixed

- The paired picture gate no longer reads a day's tone grade as softness. Each reference frame's sharpness is now put at the delivered frame's contrast (luma variance ratio) before the medians are compared, so a grade that lowers contrast passes, a smear still fails, and a grade that raises contrast makes the gate stricter. The verdict message notes the factor.
