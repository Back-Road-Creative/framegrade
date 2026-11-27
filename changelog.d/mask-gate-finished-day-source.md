### Fixed

- The publish gate's blur-mask frame check can count a finished day again. The encode deletes the raw concat and the master-drop deletes the master, so a day refused every publish as `mask_frame_count_unknown` (rec0060, Kerry-to-Cork). The day's recorded episode files now stand in for the raw, summed, but only when their fingerprints match the colour cache's record.
- A master encoded at a different frame rate from the mask no longer reads as a mismatch. Old Head's half-rate master (184,421 frames) was compared with its 368,837-frame mask. Such a master is now not used as the source count, and the gate stays fail-closed.
