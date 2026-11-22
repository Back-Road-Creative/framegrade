### Fixed

- A replace retry marker is cleared only by that replace's own launch. Another launch in the marker's tick (an auditing drive's publish, a heal resume) is returned as it is and leaves the marker for the next tick; a raise after the replace launched never re-marks it or runs a second drive.
- A crash between the replace's renames and its plan write is no longer a permanent give-up: the retry plans from the set-aside run, and a failing `_supersede` puts the run back and retries.
- A project gone from its region's backlog rows is the permanent refusal even while another row is eligible.
- A marker with no region whose manifest row is gone is given up, never stamped with the current tick's region: its `replace_retry` incident is recorded and resolved `given up: no row in any region` (durable, read by `status`), the marker is cleared and the tick lists it as `given_up_replace_retry`.
- `_supersede` writes each `superseded.json` before renaming the dir, so a crash right after the out rename no longer leaves a set-aside run with no record (which the next tick read as "nothing to supersede"). A record left in a dir by a crash before its rename is overwritten by the retry; one left by a raise is removed.
- The set-aside fallback matches only `<project>-<stamp>` dirs whose record names that project, never another project's run (`day-a` vs `day-a-long`).
- The marker is stamped `launching_at` before the launcher runs. A crash after the launcher returned ok finds that launch's job on the next tick, clears the marker and never replaces the fresh run twice.
- A raise after another drive's launch or publish in the marker tick returns that launch: no re-mark, no fall-through. The audit `since` filter now applies when retiring a dropped row's failed audit and when republishing a failed replace row.
- A supersede that raises before a move no longer buries the live run: `_unsupersede` removes only the record it wrote when the move never happened, so the next tick still plans from the old run instead of giving the replace up as "nothing to supersede". No `superseded.json` is left in a live run (`cleanup` reads one there).
- `margin_kept` incidents are recorded before the marker is cleared, and by the launch-stamp branch too, once per LF.
- A launcher that reports a failure (a timeout) after its job was written counts as launched: the fresh run is kept and never replaced twice.

### Added

- `gomoveshift status --json` lists incidents resolved in the last 7 days (`incidents.recently_resolved`: project, kind, code, outcome), so a given-up replace is named; the existing keys are unchanged.
