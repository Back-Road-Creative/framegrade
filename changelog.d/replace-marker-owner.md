### Fixed

- A replace retry marker is cleared only by that replace's own launch. Another launch in the marker's tick (an auditing drive's publish, a heal resume) is returned as it is and leaves the marker for the next tick; a raise after the replace launched never re-marks it or runs a second drive.
- A crash between the replace's renames and its plan write is no longer a permanent give-up: the retry plans from the set-aside run, and a failing `_supersede` puts the run back and retries.
- A project gone from its region's backlog rows is the permanent refusal even while another row is eligible.
- A marker with no region whose manifest row is gone is left unclaimed and reported as `unclaimed_replace_retry`, never stamped with the current tick's region.
