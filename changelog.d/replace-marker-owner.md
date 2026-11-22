### Fixed

- A replace retry marker is cleared only by that replace's own launch. Another launch in the marker's tick (an auditing drive's publish, a heal resume) is returned as it is and leaves the marker for the next tick; a raise after the replace launched never re-marks it or runs a second drive.
- A crash between the replace's renames and its plan write is no longer a permanent give-up: the retry plans from the set-aside run, and a failing `_supersede` puts the run back and retries.
- A project gone from its region's backlog rows is the permanent refusal even while another row is eligible.
- A marker with no region whose manifest row is gone is given up, never stamped with the current tick's region: its `replace_retry` incident is recorded and resolved `given up: no row in any region` (durable, read by `status`), the marker is cleared and the tick lists it as `given_up_replace_retry`.
- `_supersede` writes each `superseded.json` before renaming the dir, so a crash right after the out rename no longer leaves a set-aside run with no record (which the next tick read as "nothing to supersede"). A record left in a dir not yet renamed is ignored and overwritten by the retry.
- The set-aside fallback matches only `<project>-<stamp>` dirs whose record names that project, never another project's run (`day-a` vs `day-a-long`).
- The marker is stamped `launching_at` before the launcher runs. A crash after the launcher returned ok finds that launch's job on the next tick, clears the marker and never replaces the fresh run twice.
- A raise after another drive's launch or publish in the marker tick returns that launch: no re-mark, no fall-through. The audit `since` filter now applies when retiring a dropped row's failed audit and when republishing a failed replace row.
