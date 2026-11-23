### Fixed

- A finished replace swap is no longer undone by a sibling row left at the old hash, and a record
  carried from the plan file must be exactly the swap of the reviewed one (a mismatch is refused
  with an incident).
- The Gate C/D hash ignores the fields a swap rewrites, so a confirmation holds whichever order it
  was taken in; a real content edit still refuses.
- A reclaimed stage attestation of a finished day no longer defers the other Shorts; a Short with no
  `recording_date` key is refused per Short; a zero retry backoff no longer escapes the drain.
