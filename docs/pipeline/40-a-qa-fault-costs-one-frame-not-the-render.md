### A QA fault costs one frame, not the render (`frame_qa.FrameQaLedger`)

A route-story film once captured for roughly four hours, reached
film frame 4353, and then died with no mp4, no receipt and no evidence
(**4353 is where it DIED, not how long the film is** — see the warning below):

    FrameQaError: frame 11503 failed frame QA:
    frame 11503: travelled line ends 4.0px from the marker (max 4.0px)

Four separate defects met in that one sentence.

**`11503` was not a frame.** `scene_renderer._render_frame` passed
`int(at_time)` — trip SECONDS — to `check_frame`, which prints it as `frame N`.
Against a run that had captured 4353 frames, that reads as a frame index roughly
2.6× beyond anything rendered, and it cost two wrong diagnoses before anyone read
the call site.

> **4353 is a crash point, not a film length.** No route-story render had
> ever produced an mp4 when that number was written down, so nothing here ever
> measured the film's true length — the render simply stopped there. A later
> render (carrying this ledger, so a QA fault no longer kills the run)
> captured past frame 4600 and kept going. Do not use 4353 to estimate a
> render's progress, remaining time, or completion: three successive ETAs were
> built on it as a total and all three were wrong. The film declares no total in
> its log; the only honest progress signal is the per-leg `map leg N/M` line
> plus the source photo count, since each photo ends a map leg and starts a beat.
`_render_frame` now takes a `frame_index` and both leg paths (serial and the
worker fan-out) pass the real one; a caller with no index — `render_frame`,
judging one pose on its own — still falls back to the clock, and the pose in
the message pins the frame either way.

**A failing value printed as a passing one.** The comparison is strictly
`line_end_px > thresholds.max_line_end_to_marker_px`, but both numbers were
formatted `:.1f`, so a 4.03px overshoot rendered as `4.0px … (max 4.0px)` — a
sentence that reads as a pass. Every pixel measurement now goes through
`frame_qa.format_px` at two decimals, so the value that lost can never render
identically to the ceiling it broke.

**A four-pixel cosmetic overshoot destroyed the artifact.** This is the
late-gate anti-pattern in the project's own code-quality tiers: the cost of the
gate firing was the entire four-hour render. The gate itself is correct and
stays — it is the detector for the marker lagging the travelled line, a defect
reported by eye on an earlier film — so what changed is the price of it firing.
`SceneMapRenderer` takes an optional `qa_ledger`; given one, a fault is
RECORDED and the render carries on. `render_route_story_film` attaches a ledger
(the same optional-field pattern as `pins`), and after `_concat` has written the
film, `_report_qa_faults` writes `<film-stem>.qa-faults.json` — the faulted
frame indices and each problem sentence with a count, so one defect on 900
frames is one row, not 900 lines — saves the refused pixels, and returns the
`FrameQaError` the caller raises. A faulted render is still a FAILED render;
nothing downstream treats the film as shippable. The operator simply gets the
footage AND the list instead of four hours of nothing.

The error is returned rather than raised inside the `try` so it escapes outside
the `except FrameQaError` handler above, which exists to rescue pixels from the
staging `TemporaryDirectory`; letting it re-handle this one would overwrite the
evidence just written.

Left at `qa_ledger=None`, a fault raises immediately — which is what the film
preflight wants, since `preflight.py` judges a handful of frames precisely so a
structurally broken run can refuse before spending hours. Only the long capture
collects. The ledger is written from capture WORKER THREADS
(`_capture_leg_parallel` fans the capture step across workers), so every
mutation takes a lock, and only the FIRST fault keeps its `rgb`: a long leg is
thousands of multi-megabyte frames and holding them all would turn a reporting
feature into an out-of-memory crash.

<a id="route-story-line-width"></a>
