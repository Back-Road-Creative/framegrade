### A refused frame outlives the run that refused it

`render_route_story_film` used to draw every frame inside a `TemporaryDirectory`
under the staging root, deleted as the refusal unwinds — so for a while the
only thing a dead render left behind was a log line and a pose dict, and
diagnosing one meant re-rendering that pose by hand against the same bundle to
look at a picture the run had already drawn and thrown away.

It now draws into `film._kept_frames_root(output)` instead — a persistent
directory under the staging root, keyed on the output's own resolved path, one
retained directory per output (bounded the same way the frame cache is: a
fresh render for that output clears whatever a prior refusal left there
before it starts). A render that finishes with no QA fault clears that
directory again on its way out; a QA-refused run — mid-capture or the
end-of-render ledger — leaves every composited frame it drew sitting there
instead of deleting them, for a reviewer or a later repair pass to use.

The film also still catches `FrameQaError` on the way out and calls
`frame_qa.save_refusal_evidence`, which
writes `<film-stem>.qa-refused.png` (the raw refused pixels off the exception's
`rgb`) and `<film-stem>.qa-refused.json` beside the operator's own `output`
path, which survives the staging cleanup. The JSON carries the pose, every
problem sentence and the measured values behind them — arguing about a
threshold needs the number that lost, and re-deriving it costs another render.
Naming both files after the film keeps two renders sharing one output directory
from overwriting each other. The helper never raises: an evidence write that
threw would surface as an unrelated `OSError` about a directory and bury the
refusal that is the real failure, so it returns whatever landed instead. It is
not SILENT, though: it logs at WARNING what it could not write and why. A render once refused a frame, this helper wrote nothing, and because the
failure was swallowed bare there was no way to tell whether the evidence had
been lost or never attempted. `TypeError` is caught alongside `OSError` and
`ValueError` because a pose carrying a numpy scalar is not JSON-serialisable,
and that must not be the one exception that escapes.
