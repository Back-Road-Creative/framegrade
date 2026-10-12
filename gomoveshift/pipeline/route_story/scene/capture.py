"""Headless-Chromium capture harness for the offline MapLibre scene page.

Everything here runs fully offline: ``BundleServer`` serves only files under
one bundle directory (plus this package's ``index.html``/``scene.js`` under
``/scene/``), and ``is_allowed_request`` is the network policy a
``SceneCapture`` browser context enforces on every request the page makes —
only the loopback bundle server, ``data:`` and ``blob:`` URLs are ever let
through.
"""

from __future__ import annotations

import functools
import http.server
import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

from ..sun import SceneLight
from . import frame_cache, gl_backend

logger = logging.getLogger(__name__)

_SCENE_DIR = Path(__file__).parent

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


_ALLOWED_URL_SCHEMES = ("data:", "blob:")


def _content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _resolve_under(base: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``base``, refusing traversal, symlink escapes and dirs."""
    if not rel or rel.endswith("/"):
        return None  # no directory listings
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


_TERRAIN_TILE_RE = re.compile(r"^terrain/(\d+)/(\d+)/(\d+)\.png$")
_TERRAIN_FALLBACK_MAX_LEVELS = 4
_TILE_PX = 256


@functools.lru_cache(maxsize=512)
def _terrain_tile_from_ancestor(bundle_root: Path, z: int, x: int, y: int) -> bytes | None:
    """Cut a terrain tile the bundle does not hold out of its nearest ancestor.

    The bundle carries its finest DEM zoom only over the focal block around
    the route, but a pitched follow camera sees well past that block.
    MapLibre falls back to a parent for a missing *raster* tile; for a missing
    *raster-dem* tile it does not, and the terrain mesh is left with a hole
    the page shows through -- pure white, ringed by stretched skirt walls.
    The coarser DEM under that ground is already in the bundle, so serve the
    matching window of it, upsampled. Interpolation happens on decoded metres,
    never on the RGB-packed bytes (averaging those corrupts the elevation).
    """
    for levels in range(1, _TERRAIN_FALLBACK_MAX_LEVELS + 1):
        if z - levels < 0:
            return None
        scale = 1 << levels
        ancestor = _resolve_under(
            bundle_root, f"terrain/{z - levels}/{x // scale}/{y // scale}.png"
        )
        if ancestor is None:
            continue
        with Image.open(ancestor) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.int64)
        metres = -10000.0 + (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) * 0.1
        window = _TILE_PX // scale
        left, top = (x % scale) * window, (y % scale) * window
        # One ancestor pixel of context on every side (edge-replicated at the
        # ancestor's own border) so neighbouring cut tiles interpolate across
        # the same samples and meet without a step.
        padded = np.pad(metres, 1, mode="edge")[top : top + window + 2, left : left + window + 2]
        size = (window + 2) * scale
        upsampled = np.asarray(
            Image.fromarray(padded.astype(np.float32), "F").resize((size, size), Image.BILINEAR)
        )[scale : scale + _TILE_PX, scale : scale + _TILE_PX]
        code = np.clip(np.round((upsampled + 10000.0) * 10.0), 0, 0xFFFFFF).astype(np.int64)
        packed = np.stack([(code >> 16) & 255, (code >> 8) & 255, code & 255], axis=-1)
        buffer = io.BytesIO()
        Image.fromarray(packed.astype(np.uint8), "RGB").save(buffer, format="PNG")
        return buffer.getvalue()
    return None


def _make_handler(bundle_root: Path) -> type[http.server.BaseHTTPRequestHandler]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # quiet: tests assert on responses, not stderr chatter

        def do_GET(self) -> None:
            self._serve(send_body=True)

        def do_HEAD(self) -> None:
            self._serve(send_body=False)

        def _resolve(self) -> Path | None:
            path = unquote(urlsplit(self.path).path)
            if path.startswith("/scene/"):
                return _resolve_under(_SCENE_DIR, path[len("/scene/") :])
            return _resolve_under(bundle_root, path.lstrip("/"))

        def _terrain_fallback(self) -> bytes | None:
            match = _TERRAIN_TILE_RE.match(unquote(urlsplit(self.path).path).lstrip("/"))
            if match is None:
                return None
            z, x, y = (int(group) for group in match.groups())
            return _terrain_tile_from_ancestor(bundle_root, z, x, y)

        def _serve(self, *, send_body: bool) -> None:
            target = self._resolve()
            if target is None:
                data = self._terrain_fallback()
                if data is None:
                    self.send_error(404)
                    return
                content_type = _CONTENT_TYPES[".png"]
            else:
                data = target.read_bytes()
                content_type = _content_type(target)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if send_body:
                self.wfile.write(data)

    return _Handler


class BundleServer:
    """Loopback-only HTTP server scoped to one bundle directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> BundleServer:
        handler = _make_handler(self.root)
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("BundleServer used outside its context manager")
        return self._httpd.server_address[1]

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc_info: object) -> bool:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False


def is_allowed_request(url: str, server_origin: str) -> bool:
    """Only the exact loopback bundle-server origin, plus ``data:``/``blob:``, pass."""
    if url.startswith(_ALLOWED_URL_SCHEMES):
        return True
    parts = urlsplit(url)
    origin = urlsplit(server_origin)
    return (
        parts.scheme == origin.scheme
        and parts.hostname == origin.hostname
        and (parts.port or None) == (origin.port or None)
    )


DEFAULT_SETTLE_TIMEOUT_S = 20.0
# A pose whose tiles are merely slow (a CPU-starved box) is rendered again with
# double the budget; the last attempt waits 4x the first. A frame is only ever
# captured from a fully settled scene, so a retry never changes a pixel.
DEFAULT_SETTLE_ATTEMPTS = 3
# The scene's own words for "nothing failed, the deadline just passed". A tile
# that answered an error is named instead, and no amount of waiting loads it.
_STILL_LOADING = "tiles still loading"

# How long one browser gets to fire MapLibre's "load". The wait is enforced by
# the Playwright driver, not by a timer in the page, so a wedged page still
# times out; a browser that misses it is thrown away and a fresh one launched.
DEFAULT_READY_TIMEOUT_S = 60.0
# Five, not three: on a loaded box a 1080p render can have a worker start only
# on a later browser. Two spare tries cost at most two more minutes against a
# multi-hour render lost to one more miss.
DEFAULT_READY_ATTEMPTS = 5

# How many scene browsers may be alive at once. ONE, because on a typical
# export host a second concurrent browser stops BOTH from ever painting.
#
# A lone capture becomes ready within a couple of seconds. With two or more
# browsers alive at once, whether in the same process or in separate OS
# processes, and even with Chromium's anti-backgrounding flags, none of them
# became ready. Every stalled browser reported the same thing: style loaded,
# tiles loaded, labels loaded, zero tile failures -- and `loadFired: false`,
# MapLibre's "load" being the first COMPLETED render pass. So the scenes load
# fine and simply never paint. It is not a startup race: a second browser
# opened well after the first had settled still never became ready, while the
# first kept painting normally. And it is a property of the BOX rather than of
# one process. `gl_backend` records the same shape for the GPU path and
# software rendering is not exempt.
#
# This is an empirical default, not a law about headless Chromium, so an
# operator on a box where browsers do coexist raises it rather than editing
# code.
#
# A lone browser could also miss "load" with map.loaded() already true and no
# frame pending -- MapLibre fires "load" only inside a render pass it never
# scheduled. The scene's own ready poll asks for that pass (scene.js isReady),
# so a lone browser does not depend on the retry budget; the concurrency
# ceiling is a separate matter and stands.
MAX_CONCURRENT_CAPTURES_ENV = "GMS_ROUTE_STORY_MAX_CAPTURES"
DEFAULT_MAX_CONCURRENT_CAPTURES = 1


def max_concurrent_captures() -> int:
    """The capture-worker ceiling: the env override, else one.

    Junk is ignored rather than raised on: this sizes a render, and a typo in
    an env var must not abort a multi-hour film that would run fine serially.
    """
    raw = os.environ.get(MAX_CONCURRENT_CAPTURES_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(
                "%s=%r is not a number; holding the scene to %d capture(s)",
                MAX_CONCURRENT_CAPTURES_ENV,
                raw,
                DEFAULT_MAX_CONCURRENT_CAPTURES,
            )
    return DEFAULT_MAX_CONCURRENT_CAPTURES


_VALID_READBACKS = ("jpeg", "png")
JPEG_READBACK_QUALITY = 95

# Pinned so a captured frame's pixel dimensions never silently double against
# a HiDPI host -- and so the frame cache key (which includes this value) only
# ever needs to change when this line does.
DEVICE_SCALE_FACTOR = 1


class SceneRenderError(RuntimeError):
    """A pose never settled — normally a bundle tile the manifest promised but lacks."""


class SceneCapture:
    """Drives the offline scene page in a pinned Chromium, on the GPU when there is one."""

    def __init__(
        self,
        bundle_root: Path,
        width: int,
        height: int,
        *,
        chromium: str | None = None,
        settle_timeout_s: float = DEFAULT_SETTLE_TIMEOUT_S,
        settle_attempts: int = DEFAULT_SETTLE_ATTEMPTS,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        ready_attempts: int = DEFAULT_READY_ATTEMPTS,
        readback: str = "jpeg",
    ) -> None:
        if readback not in _VALID_READBACKS:
            raise ValueError(f"readback must be one of {_VALID_READBACKS!r}, got {readback!r}")
        if settle_attempts < 1:
            raise ValueError(f"settle_attempts must be at least 1, got {settle_attempts!r}")
        if ready_attempts < 1:
            raise ValueError(f"ready_attempts must be at least 1, got {ready_attempts!r}")
        self.bundle_root = Path(bundle_root)
        self.ready_timeout_s = ready_timeout_s
        self.ready_attempts = ready_attempts
        self.width = width
        self.height = height
        self.settle_timeout_s = settle_timeout_s
        self.settle_attempts = settle_attempts
        self.readback = readback
        self.chromium = chromium or os.environ.get("GMS_ROUTE_STORY_CHROMIUM")
        self._frame_cache: frame_cache.FrameCache | None = None
        self._frame_cache_tried = False
        # Decided when the first browser opens (``gl_backend.resolve``); None
        # until then, which the frame cache keys exactly like software.
        self.gl_backend: str | None = None

    # Enough distinct problems to show a pattern (one bad tile URL repeats per
    # tile), few enough that the message stays readable in a log.
    _MAX_PAGE_PROBLEMS = 12

    def _record_page_problem(self, text: str) -> None:
        """Keep the first few DISTINCT browser-side problems for the error message."""
        problems = getattr(self, "_page_errors", None)
        if problems is None or text in problems:
            return
        if len(problems) < self._MAX_PAGE_PROBLEMS:
            problems.append(text)

    def _page_problem_report(self) -> str:
        problems = getattr(self, "_page_errors", None)
        if not problems:
            return " (the browser reported no error of its own)"
        return "; browser reported: " + " | ".join(problems)

    # How often the readiness wait forces a frame and re-checks the flag. Low
    # enough that a scene ready in ~1 s is not made to wait, high enough that
    # the screenshots themselves are not what starves the renderer.
    _READY_POLL_S = 0.25

    def _wait_for_ready(self, budget_ms: int) -> None:
        """Wait for the scene's ready flag, FORCING a frame on every poll.

        Headless Chromium paints only when something asks it to, and MapLibre's
        "load" fires on the first rendered frame -- so a page nobody screenshots
        can finish loading its style, its tiles and its labels and still never
        report ready. With a passive wait the flag can stay false
        indefinitely (style loaded, tiles loaded, labels loaded, zero tile
        failures); with one tiny screenshot it arrives within seconds. Without
        the screenshot, a long render would die after several fresh browsers
        that were each waiting for a frame nobody had requested.

        The screenshot is deliberately tiny: it exists to make the compositor
        produce a frame, and its pixels are never read. It only helps when
        MapLibre has a frame REQUESTED; the page's own ready poll covers the
        other stall, where every source finished between two render passes
        and MapLibre never asked for another (see scene.js isReady).
        """
        deadline = time.monotonic() + budget_ms / 1000.0
        while True:
            self._page.screenshot(clip={"x": 0, "y": 0, "width": 8, "height": 8})
            if self._page.evaluate("() => window.routeStory.isReady()"):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"scene ready flag still false after {budget_ms} ms")
            time.sleep(self._READY_POLL_S)

    def _scene_stall_report(self) -> str:
        """The scene's own account of which half of readiness is outstanding."""
        try:
            diagnosis = self._page.evaluate("() => window.routeStory.diagnose()")
        except Exception:  # the page may already be gone, or predate diagnose()
            return ""
        return f"; scene reports: {json.dumps(diagnosis, sort_keys=True)}"

    def __enter__(self) -> SceneCapture:
        budget_ms = int(self.ready_timeout_s * 1000)
        for attempt in range(1, self.ready_attempts + 1):
            try:
                self._open_page()
                self._wait_for_ready(budget_ms)
                return self
            except Exception as exc:  # playwright raises its own TimeoutError type
                # Read the browser's own account BEFORE `_close()` tears the
                # page down -- afterwards there is nothing left to ask.
                report = self._page_problem_report() + self._scene_stall_report()
                self._close()
                if type(exc).__name__ != "TimeoutError":
                    raise
                if attempt == self.ready_attempts:
                    raise SceneRenderError(
                        f"scene did not become ready within {budget_ms} ms "
                        f"(after {attempt} fresh browsers){report}"
                    ) from exc
                logger.warning(
                    "scene not ready after %d ms (attempt %d of %d)%s; launching a fresh browser",
                    budget_ms,
                    attempt,
                    self.ready_attempts,
                    report,
                )
        raise AssertionError("unreachable: ready_attempts >= 1")

    def _open_page(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "SceneCapture requires the optional 'route-story-3d' extra "
                "(pip install 'gomoveshift-video[route-story-3d]')"
            ) from exc

        self.gl_backend = gl_backend.resolve(self._probe_renderer)
        self._server = BundleServer(self.bundle_root).__enter__()
        origin = self._server.origin

        self._playwright = sync_playwright().start()
        self._browser = self._launch(self._playwright, self.gl_backend)
        self._context = self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            device_scale_factor=DEVICE_SCALE_FACTOR,
            service_workers="block",
        )

        def _route(route: Any, request: Any) -> None:
            if is_allowed_request(request.url, origin):
                route.continue_()
            else:
                route.abort()

        self._context.route("**/*", _route)
        self._page = self._context.new_page()
        # A readiness timeout used to report only THAT the scene never settled.
        # The browser always knew why -- a thrown exception, a console error, or
        # a request `_route` aborted above because `is_allowed_request` rejected
        # it -- and every one of those was discarded, so the only way to learn
        # the cause was to re-run a four-hour render and watch it abort again.
        # These three listeners are what make the failure self-describing.
        self._page_errors = []
        self._page.on(
            "pageerror",
            lambda exc: self._record_page_problem(f"uncaught: {exc}"),
        )
        self._page.on(
            "console",
            lambda msg: (
                self._record_page_problem(f"console.{msg.type}: {msg.text}")
                if msg.type in ("error", "warning")
                else None
            ),
        )
        self._page.on(
            "requestfailed",
            lambda req: self._record_page_problem(
                f"request failed ({(req.failure or 'aborted')}): {req.url}"
            ),
        )
        # A 404 is a SUCCESSFUL response, so "requestfailed" never sees it and
        # Chromium's console text for one ("Failed to load resource: ... 404")
        # carries no URL. Without this listener the scene can only report that
        # something 404'd, never which tile -- which is the whole question when
        # the failure is "a bundle tile the manifest promised but lacks".
        self._page.on(
            "response",
            lambda resp: (
                self._record_page_problem(f"HTTP {resp.status}: {resp.url}")
                if resp.status >= 400
                else None
            ),
        )
        self._page.goto(f"{origin}/scene/index.html")
        self._page.evaluate("(config) => window.routeStory.init(config)", self._scene_config())

    def _launch(self, playwright: Any, backend: str) -> Any:
        launch_kwargs: dict[str, Any] = {}
        if self.chromium:
            launch_kwargs["executable_path"] = self.chromium
        return playwright.chromium.launch(
            headless=True,
            args=list(gl_backend.launch_args(backend)),
            env=gl_backend.launch_env(backend, os.environ),
            **launch_kwargs,
        )

    def _probe_renderer(self, backend: str) -> str:
        """What a browser launched for ``backend`` says it draws with.

        A throwaway browser on a blank page: it costs a second or two, once per
        process, and it is the only honest test -- Chromium accepts GL flags it
        then does not honour.
        """
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        try:
            browser = self._launch(playwright, backend)
            try:
                page = browser.new_page()
                page.goto("about:blank")
                return str(page.evaluate(gl_backend.RENDERER_PROBE_JS))
            finally:
                browser.close()
        finally:
            playwright.stop()

    def _scene_config(self) -> dict[str, Any]:
        manifest = json.loads((self.bundle_root / "manifest.json").read_text())
        return {
            "width": self.width,
            "height": self.height,
            "style": {
                "imagery": manifest["imagery"],
                "terrain": manifest["terrain"],
                "exaggeration": manifest["terrain"].get("exaggeration", 1.0),
            },
        }

    def _get_frame_cache(self) -> frame_cache.FrameCache | None:
        """Lazily build the disk cache on first use so a bundle without a
        manifest yet (e.g. a test double that never renders) never raises."""
        if self._frame_cache_tried:
            return self._frame_cache
        self._frame_cache_tried = True
        if not frame_cache.is_enabled():
            return None
        try:
            self._frame_cache = frame_cache.FrameCache(
                self.bundle_root,
                self.width,
                self.height,
                readback=self.readback,
                jpeg_quality=JPEG_READBACK_QUALITY,
                device_scale_factor=DEVICE_SCALE_FACTOR,
                gl_backend=self.gl_backend,
            )
        except OSError:
            self._frame_cache = None
        return self._frame_cache

    def render(
        self,
        pose: dict[str, Any],
        light: SceneLight | None = None,
        *,
        bypass_cache: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Render one pose, or raise ``SceneRenderError`` if the scene cannot settle.

        The scene's wait is bounded on purpose: a tile the bundle does not
        hold answers 404 and leaves MapLibre permanently unsettled, so an
        unbounded wait would hang the whole film instead of naming the tile.
        A pose that is only slow -- nothing failed, the deadline passed -- is
        rendered again up to ``settle_attempts`` times, doubling the budget
        each time, so one starved frame does not cost an hours-long film. A
        named tile failure raises at once: waiting cannot load a missing tile.

        ``light``, when given, replaces the pose's light with the real sun's
        ``SceneLight.to_scene()`` at this pose's zoom (``pose["zoom"]`` must be
        present); the caller's ``pose`` dict is never mutated. With no
        ``light`` argument, any ``light`` already on ``pose`` is passed
        through untouched.

        ``bypass_cache`` skips a would-be cache HIT and always recaptures --
        route-story film's ``--repair`` needs this to force a fresh render of
        a pose whose cached tile is the very thing a QA fault blamed; a
        normal render never sets it, and this render's result still WRITES
        the cache afterward, overwriting the stale entry rather than leaving
        it to answer the next lookup.
        """
        if light is not None:
            if "zoom" not in pose:
                raise ValueError("pose must have a 'zoom' key when light is given")
            pose = {**pose, "light": light.to_scene(pose["zoom"])}
        cache = self._get_frame_cache()
        if cache is not None and not bypass_cache:
            cached = cache.get(pose)
            if cached is not None:
                return cached
        result = self._render_pose_settled(pose)
        canvas_box = self._page.evaluate(
            "() => {"
            " const r = document.querySelector('canvas').getBoundingClientRect();"
            " return {x: r.x, y: r.y, width: r.width, height: r.height}; }"
        )
        screenshot_kwargs: dict[str, Any] = {"clip": canvas_box, "type": self.readback}
        if self.readback == "jpeg":
            screenshot_kwargs["quality"] = JPEG_READBACK_QUALITY
        image_bytes = self._page.screenshot(**screenshot_kwargs)
        with Image.open(io.BytesIO(image_bytes)) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
        labels = list(result.get("labels", []))
        if cache is not None:
            cache.put(pose, array, labels)
        return array, labels

    def _render_pose_settled(self, pose: dict[str, Any]) -> dict[str, Any]:
        budget_ms = int(self.settle_timeout_s * 1000)
        for attempt in range(1, self.settle_attempts + 1):
            try:
                result: dict[str, Any] = self._page.evaluate(
                    "(arg) => window.routeStory.renderPose(arg.pose, arg.settleTimeoutMs)",
                    {"pose": pose, "settleTimeoutMs": budget_ms},
                )
                return result
            except Exception as exc:  # playwright raises its own Error type
                message = str(exc)
                if "did not settle" not in message:
                    raise
                if _STILL_LOADING not in message:
                    raise SceneRenderError(f"{exc} (pose: {pose})") from exc
                if attempt == self.settle_attempts:
                    raise SceneRenderError(
                        f"{exc} (after {attempt} attempts, pose: {pose})"
                    ) from exc
                logger.warning(
                    "scene pose still loading after %d ms (attempt %d of %d); rendering it again",
                    budget_ms,
                    attempt,
                    self.settle_attempts,
                )
                budget_ms *= 2
        raise AssertionError("unreachable: settle_attempts >= 1")

    def __exit__(self, *exc_info: object) -> bool:
        self._close()
        return False

    def _close(self) -> None:
        """Release whatever ``_open_page`` got as far as opening, newest first."""
        for name, close in (
            ("_context", "close"),
            ("_browser", "close"),
            ("_playwright", "stop"),
        ):
            handle = getattr(self, name, None)
            setattr(self, name, None)
            if handle is not None:
                getattr(handle, close)()
        server = getattr(self, "_server", None)
        self._server = None
        if server is not None:
            server.__exit__(None, None, None)


def determinism_check(capture: SceneCapture, pose: dict[str, Any]) -> bool:
    """Render the same pose twice; the frames must be pixel-identical."""
    first, _ = capture.render(pose)
    second, _ = capture.render(pose)
    return first.shape == second.shape and bool(np.array_equal(first, second))
