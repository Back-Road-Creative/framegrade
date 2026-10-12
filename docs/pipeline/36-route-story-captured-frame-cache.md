## Route Story captured-frame cache (`gomoveshift/pipeline/route_story/scene/frame_cache.py`)

A full 1080p30 film costs ~2.5h of headless-browser capture, and nothing upstream cached the
*captured pixels* — `_terrain_tile_from_ancestor` in `capture.py` only memoises DEM tile bytes
in-process, and `scene_renderer.py` only memoises a feature count — so an unrelated change to the
dashboard, a still, or photo selection re-captured every map frame. `SceneCapture.render` now
consults a `FrameCache` before driving the browser and stores the result after: a cache hit never
opens or touches the Playwright page for that pose.

- **Key** (`FrameCache._key_for`): sha256 of canonical JSON over `CACHE_VERSION`, the bundle
  identity (`FrameCache._bundle_identity` — always a hash of the raw manifest bytes, which list a
  sha256 of every bundle file, so a re-prepared tile always misses; a manifest `id` is never
  trusted in its place), the
  **scene code identity** (`_scene_code_identity` — sha256 of this package's own `scene.js` +
  `index.html` bytes, computed ONCE per `FrameCache` instance at construction, never per frame — a
  scene.js edit must miss every previously cached frame, not silently replay them), the **vendored
  MapLibre identity** (`_vendor_identity` — the sorted `<bundle>/vendor/maplibre-gl-*` directory
  NAMEs, not their bytes; a version bump changes the directory name by construction in
  `vendor.ensure_maplibre`, so hashing the name is enough and far cheaper than hashing the vendored
  JS), the **capture settings** (`readback`, `jpeg_quality`, `device_scale_factor` — everything
  `page.screenshot`/the browser context depends on that isn't already in the pose; `capture.py`'s
  `DEVICE_SCALE_FACTOR` constant is the single place that literal lives, threaded into both the
  Playwright context and the cache key), the pose dict (including any `light`, since `render` folds
  a `SceneLight` into the pose before the cache lookup), and the output `width`/`height`. Bump
  `CACHE_VERSION` on any format/behavior change so a stale on-disk entry from a prior version is
  never read back.
- **The resolved GL backend is part of the capture settings** (`scene/gl_backend.py`, and the
  "Which GL backend the capture uses" section of `docs/route-story-scene.md`). Two rasterisers draw
  one pose a hair differently — measured GPU-vs-SwiftShader: mean absolute difference 1.2/255, 0.11%
  of pixels off by more than 8, all edge antialiasing — so a film stitched from both would shimmer
  frame to frame. Software is left OUT of the key rather than named in it: every frame cached before
  the backend was a choice was a software frame, and keeps its key. Determinism is therefore a
  within-backend property; `determinism_check` still compares two renders from one capture.
- **Storage**: two files per key under the cache dir — `<key>.png` (lossless pixel round-trip,
  never JPEG, saved at `compress_level=1` — the write is on the capture hot path, up to ~4,300
  frames per film, and the PNG default `compress_level=6` costs ~0.2-0.4s per 1080p frame the
  render can't spare) and `<key>.json` (labels) — each written atomically (`_atomic_write_image`/
  `_atomic_write_json`: a `uuid4`-suffixed temp file in the same dir, then `os.replace`), because
  several capture workers run concurrently in threads/processes. A missing, corrupt, or truncated
  entry (bad PNG bytes, malformed JSON, a non-list labels value) is always a miss, never a raise.
  A `put` failure (disk full, permission denied) is logged once per `FrameCache` instance
  (`_put_failure_logged`) and swallowed — rendering continues uncached rather than losing a
  multi-hour capture to a cache write error.
- **Directory** (`cache_dir`): `$GMS_INTERMEDIATE_DIR/route-story-frame-cache` when that env var is
  set, else the cache home plus `/route-story/frame-cache` (`paths.gms_cache_home()`: `$GMS_CACHE_HOME`, else `$XDG_CACHE_HOME/gomoveshift`, else `~/.cache/gomoveshift`) — never `/tmp` (tmpfs, wiped on reboot;
  same reasoning as `paths.gms_intermediate_dir`'s other route-story caches). Opt out entirely with
  `GMS_ROUTE_STORY_FRAME_CACHE=0` (`is_enabled`).
- **Size cap and pruning** (`FrameCache._prune`): a 1080p30 film is ~4,300 frames, ~3-4 MB each
  lossless — 12-17 GB per render, unbounded without pruning. `GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB`
  (default 40) bounds total cache bytes; over cap, `_prune` deletes oldest-mtime image+labels pairs
  until back under it. Runs at `FrameCache` construction and every `_PRUNE_EVERY_N_PUTS` (200)
  `put` calls, not every frame — a full directory scan per frame would defeat the point of caching.
  `get` touches both files' mtime on a hit (`_touch`) so a reused entry is never the one pruning
  picks first. Pruning tolerates another worker having already deleted a file it's about to prune
  (`Path.unlink(missing_ok=True)`, and a `stat()` race is just skipped) and never raises —
  `_prune_quietly` catches `OSError` and logs instead.
- **Lazy construction** (`SceneCapture._get_frame_cache`): the cache is built on first `render`
  call, not in `__init__`, and any `OSError` reading the manifest (e.g. a bundle root that never
  got a `manifest.json`, as some capture-harness test doubles use) disables caching for that
  instance rather than raising — construction failures are a miss, same as a corrupt entry.
