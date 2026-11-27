## 191 - A running render holds back only what conflicts with it

Renders run back to back for up to 45 hours, so anything that waits on "some drive is running"
waits for ever. A running render now holds back only a run on its own project or stage dirs
(the run locks, `ConcurrentRunError`) and the GPU itself (the shared GPU lock).

**Gone: the global "a drive is running" stops.** `heal.default_drive_running` (any live drive job
record, or any `gomoveshift … drive` process, Phase D uploads included) gated three paths:
`health backlog` (`a drive is running; one heavy drive at a time`, before it re-published failed
publishes, published passed audits or launched anything, in either region), `health publish`
(the same refusal) and `health heal` (`a drive job is running`). None of them asks it now:

- `health publish` and the backlog's publish steps refuse only `heal.drive_conflict`: a live run
  holding the run lock of the drive's own stages dir (`pipeline_state.lock`) or project dir
  (`pipeline_project.lock`), read without taking the flock, or a live drive job with the same
  `--output-dir` (a job still starting up). Publishing needs no GPU.
- `health heal` skips a row only on the same `drive_conflict` for its recipe.
- `health backlog` launches the next row while another drive renders; a running audit still holds
  its own region, so at most one render per region waits for the card.
- `cleanup --abandoned` refuses a stage dir only while a live drive's command line names its
  project (`heal.live_drive_cmdlines`, `pgrep -af`), not while any drive runs anywhere.

**"One heavy drive at a time" is the GPU lock's job.** The refusal served one card shared by
NVENC encodes. A render takes the lock with `queue_behind_renders=True`: while another render
(owner `gomoveshift`) holds it, the wait never times out, and the run logs every 10 min
(`GPU_QUEUE_LOG_S`) so a supervisor reading log growth sees it alive. `GMS_GPU_LOCK_WAIT_S` still
bounds a foreign holder (local inference). The queued run already holds its run locks, so nothing
else starts on its dirs, and its pre-flight and disk checks run once it has the card.

**The lock covers GPU work only.** A full run drops it as Phase D starts (no encode is left once
`phase_bc_day_first` is complete; `state_invariants` skips its NVENC probe for the same reason),
so uploads never hold the next render back. A `--phase D` run (the `gms-publish-<project>` job)
never takes it, and the job runner no longer marks it `--heavy`: an upload is not a heavy job.
Day reels try the lock once, never wait (`held` until the next hour), so `dispatch-queue` in the
same cron tick is never delayed. The host worker puts a request whose lock stays busy back in
`pending` (status `queued`) instead of failing it.

Guards: `tests/test_render_never_gates.py` (behaviour, and an AST scan that fails if any health,
launch, publish, heal or cleanup path asks a no-argument "is any drive running" probe again).
