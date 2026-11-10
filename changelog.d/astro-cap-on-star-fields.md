### Fixed

- Night stacks now tell real dark-sky frames from lens-cap-on darks. On the 2026-11-09 card the cap-on test had it backwards: every real frame of seven night groups was refused as "lens cap on" ("0 usable of 19"), while the end-of-run darks shot in the same sequences passed as usable. A frame at the sensor's black floor is now judged by how many star-sized bright spots survive an erosion that removes hot pixels: 20-383 on the card's real frames, 0-5 on its darks.
