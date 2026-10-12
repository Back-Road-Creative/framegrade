### `route-story --repair`: re-render only a prior run's QA-faulted frames

A QA-refused render (see 39-a-refused-frame-outlives-the-run-that-refused-it.md,
40-a-qa-fault-costs-one-frame-not-the-render.md) already leaves the good frames
sitting in `film._kept_frames_root(output)` and writes
`<output-stem>.qa-faults.json` naming which frames faulted and why. Before
`--repair`, the only way to fix one bad frame was to re-render the whole film —
minutes to hours of GPX/photo/scene work redone to fix pixels a single tile
glitch spoiled.

`--repair QA_FAULTS_JSON` skips all of that. It reads the fault report and the
kept frames, recaptures ONLY the faulted poses through the scene renderer
(bypassing the frame cache so the recapture can't just re-read the same stale
tile the fault blamed — see `scene.capture.SceneCapture.render`'s
`bypass_cache`), overwrites those frame files in place, re-encodes, and
concatenates — never touching a frame that already QA-passed. It requires
`--bundle` (a repair recaptures through the scene renderer; there is no
per-frame pose or QA mechanism on the default Pillow renderer, so `--repair`
only ever applies to a `--renderer scene` run). Every other required flag
(`--gpx`, `--photos-dir`, etc.) still has to parse but is otherwise unused — a
repair run never re-touches the GPX, photos, or rasters.

`_report_qa_faults` now writes two additive fields alongside the original
`frames_faulted`/`frames`/`problems` an old reader still loads unchanged:
`render_identity` (`output`, `fps`, `width`, `height`, `legs` — the shape a
repair must match before it will touch anything) and `poses` (a `{frame_index:
pose}` map, but only for frames that actually faulted with a pose recorded —
a fault at the ledger level with no per-frame QA capture, for example, has
none, which `--repair` treats as a refusal, never a gap it papers over).

`repair_route_story_film` (`film.py`) is deliberately renderer-agnostic — it
takes a `recapture: Callable[[dict], Image.Image]` closure rather than
constructing a `SceneCapture` itself, which is what makes its refusal logic
unit-testable without a browser. `main()`'s `--repair` branch is the only
caller that builds a real one, and every check below is a REFUSAL, never a
fallback to a full re-render:

- **Identity mismatch** — this run's `--output`/`--fps`/`--width`/`--height`
  don't match the report's `render_identity`. A report is written for one
  specific render; splicing its frames into a different one is silent
  corruption, not a repair.
- **Missing kept directory** — no `film._kept_frames_root(output)` on disk.
  `--repair` has nothing to patch without the frames a QA-refused render left
  behind; it never falls back to rendering them from scratch.
- **Multi-map-leg report** (`render_identity["legs"] != 1`) — a frame index
  resets to 0 at every map leg boundary (`scene_renderer.py`'s per-leg
  `frame_index`, entangled with the dashboard cadence counter — deliberately
  untouched by this change), so across more than one leg a bare frame number
  is not unique and `--repair` cannot tell which leg's frame it would
  overwrite. A route with no placed media beat splitting it — the common
  case — renders as a single leg and is unaffected.
- **A faulted frame with no recorded pose** — nothing to recapture it with.
- **A recaptured frame that still fails frame QA** — `film._repair_scene_frame`
  runs `check_frame` on the raw recaptured capture (before labels and the
  canvas, the same point `scene_renderer` checks a normal frame) and raises
  `FrameQaError`. The CLI writes the refusal evidence next to `--output` and
  exits non-zero, so a pose that recaptures into the same bad tile is never
  pasted back into the film.

A recaptured frame carries the map and its labels only, never the dashboard
overlay: the overlay's numbers depend on timeline/cadence state a repair-only
invocation has no way to reconstruct without replaying the whole render. This
is a disclosed limitation, not an oversight — a QA fault is almost always a
map-tile artifact the dashboard drawn on top of it doesn't explain.
