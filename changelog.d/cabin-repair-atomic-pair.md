### Fixed

- A cabin repair now swaps the privacy mask and its regions sidecar as a journaled pair. A pending marker beside the sidecar is written before the first replace and removed after the second; an interrupted swap leaves it, and the frame-pairing gate, the repair and the mask cache refuse the mismatched pair until a fresh detection (previously a crash between the two replaces left a new mask beside the old sidecar).
- A repair round in which no vehicle detector could be built is no longer counted as a spent round or recorded as an exhausted repair. The gate still fails closed, with `detector unavailable` in the error, and a retry once the model loads gets its full rounds.
