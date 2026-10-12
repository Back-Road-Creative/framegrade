# Route Story scene page and capture harness

`gomoveshift/pipeline/route_story/scene/` renders Route Story's cinematic
export frames using real 3D terrain (MapLibre GL JS) inside a pinned headless
Chromium, frame by frame, fully offline and deterministic.

## Which GL backend the capture uses

`gl_backend.py` decides, once per process, and `GMS_ROUTE_STORY_GL_BACKEND`
overrides it: `swiftshader` (default), `auto`, or `gpu`.

**The GPU is opt-in and single-worker only.** One GPU browser reaches ready and captures at under
half the software time per frame, but two started together never become ready on the render box
(measured: 344 s, five fresh browsers each, both failed) -- and a film opens one browser per capture
worker. Four software workers beat one GPU worker on wall-clock, so software stays the default.

`auto` launches a throwaway browser, asks it what it draws with
(`UNMASKED_RENDERER_WEBGL`), and takes the GPU only when that string names real
hardware — then falls back to SwiftShader with a warning. It judges the answer,
never the flags: Chromium accepts `--use-angle=gl` and then draws on SwiftShader
anyway, and `--use-angle=vulkan` resolves to Mesa's llvmpipe, a second software
rasteriser that looks like a success. `gpu` raises rather than downgrade
quietly; `swiftshader` pins the old software path.

A host with `/dev/dxg` but no `/dev/dri` can still reach a GPU: ANGLE's GL-EGL backend gets there through Mesa's
d3d12 driver, selected by the `GALLIUM_DRIVER` / `MESA_D3D12_DEFAULT_ADAPTER_NAME`
defaults `launch_env` supplies. On one test host (a bundle at 1920×923, one
worker, frame cache off) a GPU settled a frame in roughly half the time SwiftShader took, with the same pose rendered twice byte-identical.

Determinism holds *within* a backend, not across them — two rasterisers differ
by a hair of edge antialiasing (measured: mean absolute difference 1.2/255,
0.11% of pixels off by more than 8). So the frame cache key carries the resolved
backend, and a film is never stitched from both. Software is left out of that
key rather than named in it, so every frame cached before the backend was a
choice keeps its key.

## Layout

- `vendor.py` — `ensure_maplibre(bundle_root, fetch)` downloads and verifies
  (SRI sha512, checked *before* extraction) the vendored MapLibre GL JS
  build (ES modules: main, shared chunk, worker, plus CSS) into
  `<bundle>/vendor/maplibre-gl-6.10.0/`. `fetch` is injected so
  callers control the actual network access; the module never calls out
  itself. Idempotent via a recorded `SHA256SUMS`.
- `index.html` / `scene.js` — a plain ES2020 page (no build step) that loads
  the vendored MapLibre build and exposes `window.routeStory`:
  - `init(config)` builds a raster+terrain+hillshade+route style from
    `config.style` (imagery/terrain min/max zoom, exaggeration) and starts
    loading tiles.
  - `ready` is a promise that resolves once the style and `labels.geojson`
    are loaded.
  - `renderPose(pose)` jumps the camera to `{longitude, latitude, zoom,
    pitch, bearing, marker, marker_heading, progress, light}`, updates the
    marker, the route comet gradient, and hillshade illumination direction,
    waits for `idle` + `areTilesLoaded()`, and resolves with each label's
    projected `{x, y, visible}`. `progress` (0..1) is optional: omitted, the
    route renders fully bright as before. `SceneMapRenderer` decides whether
    to set it from the BUNDLE's own `route.geojson` feature count (read once
    from `capture.bundle_root` and cached), never from the film's own
    display segments -- `film.clean_display_segments` can split a route
    further than `prepare._route_geojson` did (an implausible-speed-jump
    split on top of a GPS gap), so a route can be 2+ display segments while
    the bundle still drew it as a single LineString feature; gating on the
    display segments would wrongly turn the comet off in exactly that case.
    2+ features in the bundle (or no `bundle_root` at all, e.g. a bare test
    double) falls back to no comet, since one scalar can't address more than
    one feature's `line-progress`. `marker_heading` is optional too:
    omitted, the marker is the plain circle; set, an in-page-drawn chevron
    (`map.addImage`, no network) rotates to face the route's own direction of
    travel, independent of the camera's own rate-limited `bearing`.
- `capture.py`:
  - `BundleServer(root)` — a loopback-only `ThreadingHTTPServer` that serves
    bundle files under `root` and this package's `index.html`/`scene.js`
    under `/scene/`. Path containment is enforced by resolving the request
    path and checking it is still under the resolved root — this also
    rejects a symlink that escapes the root, since `Path.resolve()` follows
    it before the containment check runs. No directory listings.
  - `is_allowed_request(url, server_origin)` — the network policy a
    `SceneCapture` browser context enforces on every request: only the
    exact loopback bundle-server origin, plus `data:`/`blob:` URLs, pass.
  - `SceneCapture(bundle_root, width, height, chromium=...)` — lazily
    imports Playwright (raising a clear error naming the optional
    `route-story-3d` extra if it's missing), launches the pinned Chromium
    with the args for the resolved GL backend (above), and drives
    `render(pose) -> (ndarray, labels)` by screenshotting the canvas and
    decoding it with Pillow. The chosen backend is on `capture.gl_backend`.
  - `determinism_check(capture, pose)` renders the same pose twice and
    diffs the arrays exactly.

## Installing the optional extra

```
pip install -e '.[route-story-3d]'
```

This adds Playwright; it does **not** install a browser. Point
`GMS_ROUTE_STORY_CHROMIUM` (or pass `chromium=`) at a pinned
`chrome-headless-shell` executable — the export pipeline never resolves a
browser implicitly.

## Testing

`tests/test_route_story_scene.py` covers `vendor.py` and `capture.py`
without a browser (tarball SRI/extraction safety, server containment, the
request policy, and the "extra missing" error path). One integration test,
`test_scene_renders_deterministically`, actually drives a browser against a
tiny synthetic bundle; it is skipped unless Playwright is importable, a
Chromium executable path resolves, and `GMS_ROUTE_STORY_MAPLIBRE_DIR` points
at a directory holding the real MapLibre 6 `dist/` files (`maplibre-gl.mjs`,
`maplibre-gl-shared.mjs`, `maplibre-gl-worker.mjs`, `maplibre-gl.css`) — it never
touches the network itself.
