### Photos (`console/photos.py`, `GET/POST /photos`)

Launches `photos ingest` and `photos develop` only -- selector, reel and astro are launched
from the command line, not from this screen. That is a console gap, not a missing engine, and
The page says which: `capability_status()` reports one row per capability whose
`installed` flag is measured with `importlib.util.find_spec` against the real engine module
(`pipeline/photos/{ingest,develop,selector,reel,astro_orchestrator}.py`) and whose `console`
flag is the narrower claim that this screen launches it, with `command` naming what to type
when it does not. Deriving it means the page cannot go stale against the build: it previously
read "Selector, reel and astro are not built yet" while all three engines shipped and each had
its own `gomoveshift photos ...` command, which reads as "the feature does not exist".
`journey_stages(journey)` renders the normal creator path -- Import, Suggested treatment,
Review, Schedule, Results -- as links to the screens that already drive each stage
(`/photos#import`, `/photos#develop`, `/library`, `/schedule`, `/seo`), each threaded with
`?journey=<slug>` so the selection survives navigation; `GET /photos?journey=<slug>` prefills
the ingest form from it. The film-finish and white-balance dials live in a collapsed
`<details>` Advanced block -- available, not removed. The `required` path and output-dir fields
stay outside it deliberately: a browser cannot focus a required control inside a closed
`<details>`, so the form would refuse to submit with nothing shown to explain why. Replacing
one-path-per-line entry with a picker, and the asset-presentation work (thumbnail, readable
name, date, media kind, source location, treatment, privacy/export status on `library.html`
/ `library_asset.html`) are follow-ups.
`GET /photos` renders two forms; `POST /photos/run` dispatches on the `action` field to
`photos.build_photos_argv("ingest", ...)` or `build_photos_argv("develop", ...)`, each mirroring
the real `gomoveshift photos <action> ...` argv from `cli/parser.py::_add_photos_parser` --
`"photos"` added to `console/jobs.py`'s `ALLOWED_SUBCOMMANDS`. Ingest takes card/source dir,
journey slug, day, optional location (blank defers to EXIF GPS per the photos contract), and a
dry-run checkbox default ON. Develop takes one photo path per line, an output dir, an optional
comma-separated `--styles` list, an optional named look, a "Master format" select (JPEG by default;
opt-in 16-bit TIFF master, `--format tiff16` -- JPEG appends nothing so the default argv is
unchanged, and `_develop_argv` refuses any value outside `config.PHOTO_EXPORT_FORMATS`), five
film-finish dials --
`--vignette` (0..1), `--clarity` (-1..1), `--grain` (0..1), `--grain-size` (0.25..4) and
`--halation` (0..1) -- and the white-balance group: `--white-balance-g`
(-0.5..0.5), `--white-balance-r` and `--white-balance-b` (`config.WB_OPERATOR_SPAN`, -0.1..0.1),
and their highlight-only halves `--white-balance-r-high`, `--white-balance-b-high` (-0.1..0.1) and
`--white-balance-g-high` (-0.5..0.5) -- matching `cli/parser.py`'s develop subparser exactly.
Each is `None` unless the operator fills it in, and an omitted dial contributes nothing to the
argv -- the console never sends a default value that would freeze the engine's own.
`_develop_argv` range-checks each supplied dial itself, upstream of the subprocess, and a
violation raises `ValueError` the route turns into a 400 naming the flag and its accepted range,
the same door `--look` uses. All eleven dials share one `_DIAL_RANGES` table and one validate/argv
loop, so a group's ranges can never drift from the flag name that builds the argv.

**A refusal is an inline correction here too.** Same rule and same shape as `/drives/run`
above, with two differences this page forced. First, the eleven dials used to be typed
`float | None = Form(None)`, so a typo never reached the route at all -- FastAPI answered it with a
422 whose body is a list of pydantic error dicts, naming the wire field rather than the label. They
are now plain strings parsed by the same `_number` helper against a `_PHOTO_NUMBERS` label table,
and the inputs are `type="text" inputmode="decimal"`, because a browser blanks a `type="number"`
input it cannot parse -- which is exactly the field the operator needs to see to spot the typo.
Second, **this page carries two forms**, so the template splits the returned `form` by its `action`
and refills only the one that was submitted; putting typed values into the other would be text in a
form nobody touched. Ingest's dry-run checkbox follows the drives rule exactly: ticked on a fresh
page or when `form.dry_run` is set, never re-ticked after the operator cleared it. `_ingest_argv`
also now refuses a blank card dir, journey or day -- the page marks all three `required`, which a
browser enforces and a direct POST does not, and blank they built `--day ''` and launched a job that
could only fail inside the log, by which point the form was gone. The remaining
develop knobs (~37, after the white-balance group closes the gap by 6) group into: look/LUT
selection (`--look-variant`, `--lut`, `--lut-strength`, `--lut-over`, `--lut-auto`,
`--look-strength`, `--style-strength` -- the `--lut*` family takes a file path and needs its own
validation story), optics/film-finish effects (`--graduated-nd(-angle)`, `--halation-tint`,
`--light-leak`, `--chromatic-ab`, `--holo-sheen`, `--scanline`, `--glow`, `--glitch`, `--dehaze`,
`--sky-darken(-saturate)`, `--sharpen`, `--vibrance`), and colour grading (`--bw-mix`, `--duotone*`,
`--layers`, `--reference`, `--hue-*`, `--curve-*`, `--split-tone-*`) -- none are exposed by this
slice. The look dropdown is
populated by `photos.resolved_looks()`, the same `{**BUILTIN_LOOKS, **photos.looks}` merge the
CLI's `--look` handler resolves against (`cli/handlers/photos.py::_cmd_photos_develop`) -- one
source of truth, so an operator-configured look appears without a console change, and a name
outside that merged set is refused with a 400 before it ever reaches the subprocess argv. An
unrecognized `action` is a 400, not a silent no-op.

## ML upscale for stills (`MediaProfile.upscale: ml`)

`ml` applies to stills only: `PhotoResizer(upscale="ml")` enlarges a photo narrower than the
target width. The Lanczos resample is always computed first and is the fallback. Video keeps
resolving `ml` to the spline36 resample (`quality.upscale_plan`). Nothing else changes: an
export that is not an upscale is byte-identical to before. `photos/develop.py` has no
upscale path (gallery masters ship at native size), so it is untouched.

`quality.ml_upscale_still` accepts the ML frame only when sharpness >= the resample's,
ringing <= `RINGING_MAX` and SSIM-to-resample >= `ML_UPSCALE_SSIM_FLOOR` (provisional). Any miss,
missing/mismatched weights or model error ships the resample and logs the numbers. The
provenance entry gains `upscale: {scaler: "ml"|"resample", upscaled_from, reason, sharpness_ml,
sharpness_resampled, ringing_ml, ssim_to_resampled}`.

The weights are vendored by hand (the service never downloads). The pinned file is
`realesrgan-x4plus.onnx` (sha256 in `quality.ML_UPSCALE_WEIGHTS_SHA256`, the one home for the pin);
`tools/export_realesrgan_onnx.py` builds it from the upstream `.pth`. Point `GMS_ML_UPSCALE_WEIGHTS` at it
or copy it to `<service>/models/` (`models/*.onnx` is gitignored, never commit it). A sha256
mismatch refuses the model and the resample ships. I/O: input `input` float32 NCHW RGB 0..1,
output `output` float32 NCHW RGB 0..1 at 4x, dynamic N/H/W.

`quality.load_ml_upscaler` calls `onnxruntime.preload_dlls()` (when present) before building the
session: the CUDA libs are pip wheels, and without it CUDA fails to bind
(`libcublasLt.so.12 not found`) and the resample silently ships. `_run_model` runs the still in
`ML_UPSCALE_TILE` (256 px) tiles with `ML_UPSCALE_TILE_PAD` (16 px) of context per side, cropped
after x4 and stitched; a still no larger than one tile runs whole. Whole-still inference on the
8 GB RTX 4070 saturated memory from 960 px (960x540 35 s, 1280x720 157 s, 1920x1080 409 s).

Regenerate: fetch upstream `RealESRGAN_x4plus.pth` (xinntao/Real-ESRGAN v0.1.0 release, BSD-3-Clause;
sha256 `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`) by hand, then
`.venv/bin/python tools/export_realesrgan_onnx.py RealESRGAN_x4plus.pth out.onnx` (needs the
`onnx` package, absent from the service venv; install it into a scratch `--target` dir on
`PYTHONPATH`). It prints the sha256; two exports on torch 2.10 / onnx 1.23 were byte-identical. A
different torch/onnx may change the bytes: re-pin the constant if so.
`preflight.check_ml_upscale(profile.media.upscale)` reports weights and CUDA state without
raising; it is not yet called from the run preflight (the caller that has the profile does).
