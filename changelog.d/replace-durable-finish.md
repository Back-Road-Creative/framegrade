### Fixed

- `--replace-scheduled` can no longer leave two scheduled uploads in one slot or delete a public
  video: durable per-LF state, a private insert, a live recheck before the delete (else the new
  video goes), and a resume of unfinished LFs on every Phase D entry. A plan run emits no shorts.
