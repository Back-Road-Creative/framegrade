### Selection visibility in the log (`still_selection.select_kept_photos`)

`select_kept_photos` logs a summary through the stdlib `logging` module the
moment selection finishes — within the first minute of a run, not only in
the finished run's receipt. One INFO line reports the totals (candidates in,
groups formed, kept, dropped); one further INFO line per group that actually
collapsed (more than one member) names the winner, the losers, and the
selector's own `GroupResult.reason` string. A group of one member logs
nothing beyond the totals line, so an ordinary shoot with no look-alikes
doesn't spam a line per photo. When selection is disabled
(`SelectionOptions.enabled=False`) it logs a single "disabled" line and
nothing per-group. The receipt's `photo_selection` block (built by
`photo_selection_block()` in `film.py`) remains the machine-readable record
of the same decision; the log is the operator-facing one, visible while the
film is still rendering.
