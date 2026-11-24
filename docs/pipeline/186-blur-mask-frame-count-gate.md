## Blur-mask frame-count gate (`mask_index_mismatch`)

A drive rendered before the redactcam frame-count fix carries a blur mask built on more frames than
its source container holds (Kerry-to-Cork: 306,385 vs 306,377), so the blur lags the picture.
`gomoveshift/publishing/mask_frame_gate.py` is the one check every upload entry point calls:
`health publish` (and the backlog republish, which runs `run_publish`) refuses before any store
write, and Phase D raises `MaskFrameCountError` at the top of an upload run (an audit pass is not
gated). Per `day_<date>` stage dir it compares `source_frame_count` in
`day_<date>_blur_regions.json` with the container's ffprobe `nb_frames` (`day_<date>_raw.mp4`, else
`day_<date>_master.mp4`). Unequal counts refuse, naming the day, both counts and the difference. A
missing sidecar, a missing count, or no probeable container refuses too, as
`mask_frame_count_unknown` (fail closed). Once the raw and the master are both purged, the check
reads `day_<date>_source_frames.json` instead, which `blur_gate.persist_source_frames` writes (atomic
tmp + rename) beside the sidecar only after the final frame-pairing check passes, on the day-master
and the stage path. It holds the three equal counts, the source fingerprint, the sidecar's sha256,
the mask's size and mtime, the renderer identity and the redactcam commit (from `direct_url.json`,
`null` when absent). The gate accepts it only while the sidecar's sha256 and the mask's size/mtime
still equal what is on disk, and never when a raw or master exists (their live probe wins); a
missing, unreadable, malformed, stale or unequal file leaves the day `mask_frame_count_unknown`, and
counts that disagree with the sidecar refuse as a mismatch. The file (schema `v2`) also records the
`date` and `day_dir` it was written for and the gate accepts it only for that day, so a proof copied
into another day dir with preserved mtimes stays unknown; a `v1` file names no day, so it is unknown
too (fail closed). A raw or master that is a dangling symlink counts as present (no fallback). Persist
snapshots the sidecar sha256 and mask identity before and after counting and writes nothing if they
moved; it is best-effort (any exception is logged and the encode goes on). `redactcam_commit` is the
commit installed when the proof was written, not necessarily the one that rendered a cache-reused
mask: no reader may treat it as the mask's renderer commit. Every skipped proof is logged (WARNING naming the day and the reason), a failed write removes its
`.tmp`, and no proof is written for the stage path (its sidecar is named for the project, not a
`day_<date>` dir, so nothing would read it). Cleanup (`drop_day_master`) drops the day master only
once the gate can measure the day without it (`blur_gate.can_measure_without_master`: the raw
is a regular file that probes to a positive frame count, else no raw exists and the persisted proof
validates). With no such proof it first runs the pairing check `persist_source_frames` requires
(`blur_gate.frame_pairing_error`; a refusal, e.g. a pending repair swap, keeps the master with a
WARNING and writes no proof), then writes a proof from the master and asks the same rule again. A
stray `day_<date>_raw.mp4` (the pipeline's own concat intermediate, which the encoder unlinks after
the master encode) that is 0-byte, unprobeable or a dangling symlink is removed, with a WARNING
naming it, once that proof validates, so the master can then drop. A directory or any other raw is
never removed and keeps the master with a WARNING, as does a proof that does not validate or an
unlink that fails. A mask changed after the drop still leaves that day
`mask_frame_count_unknown` (fail closed). A day with no blur sidecar is unaffected. A
`day_<date>` dir holding no `day_<date>_*` file is `mask_frame_count_unknown` when it holds finals
(`regular_lfs`, `shorts`, `superlong`), else skipped with a WARNING. A day rendered before this file existed
has none, so it stays unknown until a re-render. Each refused day gets one `mask_index_mismatch` incident
(`mask_index_mismatch_<drive>_<date>.json`, key `mask_index_mismatch:<drive>:<date>`), never one
per tick; it is resolved once the counts agree. There is no override flag or variable: re-render
the day on the fixed renderer, which rewrites the sidecar. A `--dry-run` refuses but records no
incident. Queue-dispatched Shorts are not re-checked; they are enqueued by Phase D, behind it.
