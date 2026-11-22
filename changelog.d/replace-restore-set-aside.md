### Fixed

- A replace's old run set aside by a crash is always put back: when the retry tick ends without launching (refused, transient or permanent), the newest set-aside run is restored before the marker can clear, never over a live run (a `restore_failed` incident and the marker are kept instead).
- Only a running or succeeded job of this launch counts as launched: the launcher's `run_id` is pinned into the marker and matched, and a failed or stale job is ignored.
- `_unsupersede` counts a pair as moved only when its set-aside dir carries this attempt's record, so an unrelated dir there never lets the live run be renamed away; a stray `superseded.json` is removed when a replace is given up or permanently refused.
