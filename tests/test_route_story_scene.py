"""Tests for the offline MapLibre scene page and headless capture harness."""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import math
import os
import tarfile
from pathlib import Path

import numpy as np
import importlib.util

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story.scene import capture as capture_mod
from gomoveshift.pipeline.route_story.scene import vendor
from gomoveshift.pipeline.route_story.sun import SceneLight


def _make_tgz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


_DIST = {
    "package/dist/maplibre-gl.mjs": b"/* main */",
    "package/dist/maplibre-gl-shared.mjs": b"/* shared */",
    "package/dist/maplibre-gl-worker.mjs": b"/* worker */",
    "package/dist/maplibre-gl.css": b"/* css */",
}


def _sri_for(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


class TestEnsureMaplibre:
    def test_rejects_tarball_that_fails_sri_and_writes_nothing(self, tmp_path):
        data = _make_tgz(_DIST)
        wrong_sri = _sri_for(b"not the real tarball")

        with pytest.raises(ValueError, match="SRI"):
            vendor.ensure_maplibre(tmp_path, fetch=lambda url: data, sri=wrong_sri)

        assert not (tmp_path / "vendor").exists()

    def test_extracts_only_dist_js_and_css(self, tmp_path):
        js = b"/* maplibre js */"
        css = b"/* maplibre css */"
        data = _make_tgz(
            {
                "package/dist/maplibre-gl.mjs": js,
                "package/dist/maplibre-gl-shared.mjs": b"/* shared */",
                "package/dist/maplibre-gl-worker.mjs": b"/* worker */",
                "package/dist/maplibre-gl.css": css,
                "package/README.md": b"not vendored",
                "package/dist/maplibre-gl-dev.mjs": b"not vendored either",
            }
        )
        sri = _sri_for(data)

        dest = vendor.ensure_maplibre(tmp_path, fetch=lambda url: data, sri=sri)

        assert (dest / "maplibre-gl.mjs").read_bytes() == js
        assert (dest / "maplibre-gl.css").read_bytes() == css
        assert not (dest / "README.md").exists()
        assert not (dest / "maplibre-gl-dev.mjs").exists()
        assert (dest / "SHA256SUMS").is_file()

    def test_rejects_path_traversal_member_and_extracts_the_rest(self, tmp_path):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, content in _DIST.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            evil = tarfile.TarInfo(name="../../../tmp/evil.js")
            evil.size = 3
            tar.addfile(evil, io.BytesIO(b"bad"))
        data = buffer.getvalue()
        sri = _sri_for(data)

        vendor.ensure_maplibre(tmp_path, fetch=lambda url: data, sri=sri)

        assert not (tmp_path.parent / "tmp" / "evil.js").exists()
        assert not Path("/tmp/evil.js").exists()

    def test_rejects_symlink_member_disguised_as_dist_file(self, tmp_path):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="package/dist/maplibre-gl.mjs")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        data = buffer.getvalue()
        sri = _sri_for(data)

        with pytest.raises(ValueError, match="regular file"):
            vendor.ensure_maplibre(tmp_path, fetch=lambda url: data, sri=sri)

    def test_idempotent_skips_fetch_when_hashes_match(self, tmp_path):
        data = _make_tgz(_DIST)
        sri = _sri_for(data)
        calls = []

        def fetch(url):
            calls.append(url)
            return data

        vendor.ensure_maplibre(tmp_path, fetch=fetch, sri=sri)
        vendor.ensure_maplibre(tmp_path, fetch=fetch, sri=sri)

        assert len(calls) == 1


class TestBundleServer:
    def test_serves_bundle_files_and_scene_assets(self, tmp_path):
        (tmp_path / "manifest.json").write_text('{"schema": "route-story-bundle/1"}')
        with capture_mod.BundleServer(tmp_path) as server:
            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", "/manifest.json")
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.read() == b'{"schema": "route-story-bundle/1"}'
            assert "json" in resp.getheader("Content-Type")
            conn.close()

            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", "/scene/index.html")
            resp = conn.getresponse()
            assert resp.status == 200
            assert b"scene.js" in resp.read()
            conn.close()

            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", "/scene/scene.js")
            resp = conn.getresponse()
            assert resp.status == 200
            assert "javascript" in resp.getheader("Content-Type")
            conn.close()

    def test_rejects_path_traversal(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{}")
        secret = tmp_path.parent / "secret.txt"
        secret.write_text("outside the bundle")
        with capture_mod.BundleServer(tmp_path) as server:
            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", "/../secret.txt")
            resp = conn.getresponse()
            assert resp.status == 404
            conn.close()

    def test_rejects_symlink_escaping_root(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("nope")
        (tmp_path / "escape.txt").symlink_to(outside)
        with capture_mod.BundleServer(tmp_path) as server:
            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", "/escape.txt")
            resp = conn.getresponse()
            assert resp.status == 404
            conn.close()

    def test_no_directory_listing(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.txt").write_text("x")
        with capture_mod.BundleServer(tmp_path) as server:
            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", "/sub/")
            resp = conn.getresponse()
            assert resp.status == 404
            conn.close()

    def test_404_for_missing_file(self, tmp_path):
        with capture_mod.BundleServer(tmp_path) as server:
            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", "/nope.json")
            resp = conn.getresponse()
            assert resp.status == 404
            conn.close()


def _dem_png(elevation_m: np.ndarray) -> bytes:
    """Mapbox terrain-RGB encode: elevation = -10000 + (R*65536 + G*256 + B) * 0.1."""
    code = np.round((elevation_m + 10000.0) * 10.0).astype(np.int64)
    rgb = np.stack([(code >> 16) & 255, (code >> 8) & 255, code & 255], axis=-1).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _dem_elevation(png: bytes) -> np.ndarray:
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.int64)
    return -10000.0 + (rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]) * 0.1


class TestTerrainParentFallback:
    """A pitched camera sees past the focal z13 block; a 404 there is a hole in the ground."""

    def _get(self, server, path):
        conn = http.client.HTTPConnection("127.0.0.1", server.port)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, body

    def test_a_missing_terrain_tile_is_cut_from_its_nearest_ancestor(self, tmp_path):
        # A west-to-east ramp, 1000 m at the west edge rising 1 m per parent pixel.
        ramp = np.tile(1000.0 + np.arange(256, dtype=np.float64), (256, 1))
        parent = tmp_path / "terrain" / "12" / "1097"
        parent.mkdir(parents=True)
        (parent / "1619.png").write_bytes(_dem_png(ramp))
        with capture_mod.BundleServer(tmp_path) as server:
            # z13 child (2195, 3238) is the parent's north-EAST quadrant.
            status, body = self._get(server, "/terrain/13/2195/3238.png")
            assert status == 200
            child = _dem_elevation(body)
            assert child.shape == (256, 256)
            # East half of the ramp: 1128 m .. 1255 m, still rising eastward, no pit.
            assert child.min() >= 1127.0
            assert child.max() <= 1256.0
            assert np.all(np.diff(child[128]) >= 0)
            # Two levels down works too (z14 from the same z12 ancestor).
            status, body = self._get(server, "/terrain/14/4388/6476.png")
            assert status == 200
            assert _dem_elevation(body).min() >= 999.0

    def test_a_real_tile_is_served_untouched(self, tmp_path):
        tile_dir = tmp_path / "terrain" / "13" / "2195"
        tile_dir.mkdir(parents=True)
        payload = _dem_png(np.full((256, 256), 1500.0))
        (tile_dir / "3238.png").write_bytes(payload)
        with capture_mod.BundleServer(tmp_path) as server:
            assert self._get(server, "/terrain/13/2195/3238.png") == (200, payload)

    def test_no_ancestor_is_still_a_404_and_imagery_is_never_synthesised(self, tmp_path):
        parent = tmp_path / "imagery" / "12" / "1097"
        parent.mkdir(parents=True)
        Image.new("RGB", (256, 256), "#224422").save(parent / "1619.jpg")
        with capture_mod.BundleServer(tmp_path) as server:
            assert self._get(server, "/terrain/13/2195/3238.png")[0] == 404
            assert self._get(server, "/imagery/13/2195/3238.jpg")[0] == 404


class TestIsAllowedRequest:
    ORIGIN = "http://127.0.0.1:54321"

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("http://127.0.0.1:54321/scene/index.html", True),
            ("http://127.0.0.1:54321/imagery/0/0/0.jpg", True),
            ("data:image/png;base64,AAAA", True),
            ("blob:http://127.0.0.1:54321/some-uuid", True),
            ("http://127.0.0.1:9999/other-port", False),
            ("https://tiles.example.com/x.png", False),
            ("http://evil.example.com/", False),
            ("file:///etc/passwd", False),
            ("ws://127.0.0.1:54321/socket", False),
        ],
    )
    def test_policy(self, url, expected):
        assert capture_mod.is_allowed_request(url, self.ORIGIN) is expected


def _playwright_importable() -> bool:
    # find_spec, not try/import: one branch, so the helper is fully covered
    # whether or not the route-story-3d extra is installed.
    return importlib.util.find_spec("playwright") is not None


class TestSceneCaptureRequiresExtra:
    @pytest.mark.skipif(
        _playwright_importable(),
        reason="playwright is installed in this environment -- the "
        "missing-extra error path is only reachable without it",
    )
    def test_missing_playwright_raises_clear_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="route-story-3d"):
            with capture_mod.SceneCapture(tmp_path, 640, 360):
                pass


def _encode_terrain_rgb(elevation: float) -> tuple[int, int, int]:
    value = round((elevation + 10000.0) / 0.1)
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def _build_tiny_bundle(root: Path, maplibre_dir: Path) -> Path:
    (root / "imagery" / "0" / "0").mkdir(parents=True)
    (root / "terrain" / "0" / "0").mkdir(parents=True)
    Image.new("RGB", (256, 256), (90, 130, 90)).save(root / "imagery" / "0" / "0" / "0.jpg")
    ridge = Image.new("RGB", (256, 256))
    pixels = ridge.load()
    for x in range(256):
        for y in range(256):
            elevation = 200.0 + 50.0 * math.sin(x / 20.0)
            pixels[x, y] = _encode_terrain_rgb(elevation)
    ridge.save(root / "terrain" / "0" / "0" / "0.png")

    (root / "route.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-122.42, 37.78], [-122.40, 37.80]],
                        },
                    }
                ],
            }
        )
    )
    (root / "labels.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Test Peak", "kind": "summit", "source": "test"},
                        "geometry": {"type": "Point", "coordinates": [-122.41, 37.79]},
                    }
                ],
            }
        )
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "route-story-bundle/1",
                "footprint": {"west": -123, "south": 37, "east": -122, "north": 38},
                "imagery": {"minzoom": 0, "maxzoom": 0},
                "terrain": {"minzoom": 0, "maxzoom": 0, "encoding": "mapbox", "exaggeration": 1.5},
            }
        )
    )

    vendor_dest = root / "vendor" / "maplibre-gl-6.10.0"
    vendor_dest.mkdir(parents=True)
    for name in (
        "maplibre-gl.mjs",
        "maplibre-gl-shared.mjs",
        "maplibre-gl-worker.mjs",
        "maplibre-gl.css",
    ):
        (vendor_dest / name).write_bytes((maplibre_dir / name).read_bytes())
    return root


def _integration_available() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    chromium = os.environ.get("GMS_ROUTE_STORY_CHROMIUM")
    if not chromium or not Path(chromium).is_file():
        return False
    maplibre_dir = os.environ.get("GMS_ROUTE_STORY_MAPLIBRE_DIR")
    if not maplibre_dir or not (Path(maplibre_dir) / "maplibre-gl.mjs").is_file():
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(
    not _integration_available(),
    reason="requires playwright + GMS_ROUTE_STORY_CHROMIUM + GMS_ROUTE_STORY_MAPLIBRE_DIR",
)
def test_scene_renders_deterministically(tmp_path):
    maplibre_dir = Path(os.environ["GMS_ROUTE_STORY_MAPLIBRE_DIR"])
    bundle = _build_tiny_bundle(tmp_path / "bundle", maplibre_dir)
    pose = {
        "longitude": -122.41,
        "latitude": 37.79,
        "zoom": 12,
        "pitch": 45,
        "bearing": 0,
        "marker": [-122.41, 37.79],
        "light": {"azimuth": 210, "altitude": 45},
    }
    with capture_mod.SceneCapture(bundle, 320, 240) as scene:
        assert capture_mod.determinism_check(scene, pose)
        image, labels = scene.render(pose)
        assert isinstance(image, np.ndarray)
        assert image.shape == (240, 320, 3)
        assert image.std() > 1.0
        assert any(label["name"] == "Test Peak" for label in labels)


@pytest.mark.skipif(
    not _integration_available(),
    reason="requires playwright + GMS_ROUTE_STORY_CHROMIUM + GMS_ROUTE_STORY_MAPLIBRE_DIR",
)
def test_scene_draws_a_white_pin_where_the_pose_says_one_played(tmp_path):
    """scene.js itself, in a real browser: a pose's pin lands on its own spot."""
    maplibre_dir = Path(os.environ["GMS_ROUTE_STORY_MAPLIBRE_DIR"])
    bundle = _build_tiny_bundle(tmp_path / "bundle", maplibre_dir)
    pose = {
        "longitude": -122.41,
        "latitude": 37.79,
        "zoom": 12,
        "pitch": 0,
        "bearing": 0,
        "marker": [-122.41, 37.79],
        "light": {"azimuth": 210, "altitude": 45},
    }
    # 0.01 deg of longitude at zoom 12 is 58 px east of the centred marker.
    pinned = {**pose, "pins": [[-122.40, 37.79]]}
    with capture_mod.SceneCapture(bundle, 320, 240) as scene:
        plain, _ = scene.render(pose)
        with_pin, _ = scene.render(pinned)
        again, _ = scene.render(pose)

    assert with_pin[120, 160 + 58].min() >= 235
    assert plain[120, 160 + 58].min() < 235
    changed = np.argwhere((with_pin != plain).any(axis=2))
    assert len(changed) > 0
    # The pin and nothing else -- one lossy-screenshot block of slack around it.
    assert np.abs(changed - np.array([120, 160 + 58])).max() <= 24
    assert np.array_equal(again, plain)  # a pin-less pose clears the last pose's pins


class _FakePage:
    """Stands in for the Playwright page: records evaluate calls, fakes a screenshot."""

    def __init__(
        self,
        *,
        render_error: Exception | None = None,
        render_errors: list[Exception] | None = None,
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.render_error = render_error
        # Raised one per renderPose call, in order; then renders succeed.
        self.render_errors = list(render_errors or [])

    def evaluate(self, script: str, arg: object = None) -> object:
        self.calls.append((script, arg))
        if "getBoundingClientRect" in script:
            return {"x": 0, "y": 0, "width": 2, "height": 2}
        if self.render_error is not None:
            raise self.render_error
        if self.render_errors:
            raise self.render_errors.pop(0)
        return {"labels": [{"name": "Test Peak"}]}

    def screenshot(self, **kwargs: object) -> bytes:
        self.screenshot_kwargs = kwargs
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), (5, 6, 7)).save(buffer, format="PNG")
        return buffer.getvalue()


def _capture_with_fake_page(tmp_path, **kwargs) -> tuple[object, _FakePage]:
    capture = capture_mod.SceneCapture(tmp_path, 2, 2, **kwargs)
    page = _FakePage()
    capture._page = page
    return capture, page


class TestRenderBoundsItsWait:
    def test_render_hands_the_settle_budget_to_the_scene(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path, settle_timeout_s=7.5)

        capture.render({"longitude": 1.0, "latitude": 2.0, "zoom": 12.0})

        pose_arg = page.calls[0][1]
        assert pose_arg["settleTimeoutMs"] == 7500
        assert pose_arg["pose"]["zoom"] == 12.0

    def test_default_settle_budget_is_finite(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path)

        capture.render({"longitude": 1.0, "latitude": 2.0, "zoom": 12.0})

        assert 0 < page.calls[0][1]["settleTimeoutMs"] < 10 * 60 * 1000

    def test_a_scene_that_never_settles_raises_with_the_pose_and_the_cause(self, tmp_path):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2)
        capture._page = _FakePage(
            render_error=RuntimeError(
                "Error: scene did not settle within 20000ms: "
                "tile http://127.0.0.1:1/imagery/11/2/3.jpg failed"
            )
        )

        with pytest.raises(capture_mod.SceneRenderError) as raised:
            capture.render({"longitude": 1.0, "latitude": 2.0, "zoom": 11.5})

        message = str(raised.value)
        assert "imagery/11/2/3.jpg" in message
        assert "11.5" in message

    def test_an_unrelated_page_error_is_not_reshaped(self, tmp_path):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2)
        capture._page = _FakePage(render_error=RuntimeError("Target page closed"))

        with pytest.raises(RuntimeError, match="Target page closed"):
            capture.render({"longitude": 1.0, "latitude": 2.0, "zoom": 11.5})


def _still_loading(budget_ms: int = 20000) -> RuntimeError:
    return RuntimeError(f"Error: scene did not settle within {budget_ms}ms: tiles still loading")


def _render_budgets(page: _FakePage) -> list[int]:
    return [arg["settleTimeoutMs"] for script, arg in page.calls if "renderPose" in script]


class TestRenderRetriesAStarvedPose:
    """A busy box can starve one pose past its budget; that must not kill a film."""

    POSE = {"longitude": 1.0, "latitude": 2.0, "zoom": 11.5}

    def test_a_pose_still_loading_once_is_rendered_again_and_returns_its_frame(self, tmp_path):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2)
        capture._page = page = _FakePage(render_errors=[_still_loading()])

        image, labels = capture.render(self.POSE)

        assert image.shape == (2, 2, 3)
        assert labels == [{"name": "Test Peak"}]
        assert len(_render_budgets(page)) == 2

    def test_each_retry_gets_a_longer_settle_budget(self, tmp_path):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2, settle_timeout_s=10.0)
        capture._page = page = _FakePage(render_errors=[_still_loading(), _still_loading()])

        capture.render(self.POSE)

        assert _render_budgets(page) == [10000, 20000, 40000]

    def test_a_pose_that_never_loads_raises_with_the_pose_after_bounded_attempts(self, tmp_path):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2, settle_attempts=3)
        capture._page = page = _FakePage(render_error=_still_loading())

        with pytest.raises(capture_mod.SceneRenderError) as raised:
            capture.render(self.POSE)

        message = str(raised.value)
        assert "11.5" in message
        assert "3 attempts" in message
        assert len(_render_budgets(page)) == 3

    def test_a_missing_tile_fails_at_once_because_waiting_cannot_fix_it(self, tmp_path):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2)
        capture._page = page = _FakePage(
            render_error=RuntimeError(
                "Error: scene did not settle within 20000ms: "
                "tile http://127.0.0.1:1/imagery/11/2/3.jpg failed"
            )
        )

        with pytest.raises(capture_mod.SceneRenderError, match="imagery/11/2/3.jpg"):
            capture.render(self.POSE)

        assert len(_render_budgets(page)) == 1

    def test_an_unrelated_page_error_is_never_retried(self, tmp_path):
        capture = capture_mod.SceneCapture(tmp_path, 2, 2)
        capture._page = page = _FakePage(render_error=RuntimeError("Target page closed"))

        with pytest.raises(RuntimeError, match="Target page closed"):
            capture.render(self.POSE)

        assert len(_render_budgets(page)) == 1

    def test_settle_attempts_below_one_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="settle_attempts"):
            capture_mod.SceneCapture(tmp_path, 2, 2, settle_attempts=0)


class TestRenderBypassCache:
    """``--repair`` overwrites a bad cached tile; a normal render never does."""

    POSE = {"longitude": 1.0, "latitude": 2.0, "zoom": 11.5}

    def _inject_cache(self, capture, tmp_path):
        from gomoveshift.pipeline.route_story.scene import frame_cache as frame_cache_mod

        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir(exist_ok=True)
        (bundle_root / "manifest.json").write_bytes(b"{}")
        cache = frame_cache_mod.FrameCache(
            bundle_root, capture.width, capture.height, cache_root=tmp_path / "cache"
        )
        capture._frame_cache_tried = True
        capture._frame_cache = cache
        return cache

    def test_a_cache_hit_is_returned_unchanged_by_default(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path)
        cache = self._inject_cache(capture, tmp_path)
        stale = np.full((2, 2, 3), 9, dtype=np.uint8)
        cache.put(self.POSE, stale, [{"name": "stale"}])

        array, labels = capture.render(self.POSE)

        assert np.array_equal(array, stale)
        assert labels == [{"name": "stale"}]
        assert page.calls == []  # never captured -- the cache hit answered it

    def test_bypass_cache_recaptures_and_overwrites_the_stale_entry(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path)
        cache = self._inject_cache(capture, tmp_path)
        stale = np.full((2, 2, 3), 9, dtype=np.uint8)
        cache.put(self.POSE, stale, [{"name": "stale"}])

        array, labels = capture.render(self.POSE, bypass_cache=True)

        assert not np.array_equal(array, stale)  # the fake page's own fill, not the stale cache
        assert page.calls  # the page WAS asked to capture
        refreshed, refreshed_labels = cache.get(self.POSE)
        assert np.array_equal(refreshed, array)
        assert refreshed_labels == labels


class TestRenderWithLight:
    def test_render_adds_light_dict_from_scene_light(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path)
        light = SceneLight(azimuth_deg=270.0, elevation_deg=30.0, intensity=0.8, night=False)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 11.0}

        capture.render(pose, light=light)

        pose_arg = page.calls[0][1]["pose"]
        assert pose_arg["light"] == light.to_scene(11.0)

    def test_render_without_light_passes_pose_light_through_untouched(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 11.0, "light": {"azimuth": 210}}

        capture.render(pose)

        pose_arg = page.calls[0][1]["pose"]
        assert pose_arg["light"] == {"azimuth": 210}

    def test_render_with_light_requires_zoom(self, tmp_path):
        capture, _page = _capture_with_fake_page(tmp_path)
        light = SceneLight(azimuth_deg=270.0, elevation_deg=30.0, intensity=0.8, night=False)
        pose = {"longitude": 1.0, "latitude": 2.0}

        with pytest.raises(ValueError, match="zoom"):
            capture.render(pose, light=light)

    def test_render_with_light_does_not_mutate_caller_pose(self, tmp_path):
        capture, _page = _capture_with_fake_page(tmp_path)
        light = SceneLight(azimuth_deg=270.0, elevation_deg=30.0, intensity=0.8, night=False)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 11.0}

        capture.render(pose, light=light)

        assert "light" not in pose


class TestReadback:
    def test_defaults_to_jpeg_quality_95(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path)

        capture.render({"longitude": 1.0, "latitude": 2.0, "zoom": 12.0})

        assert page.screenshot_kwargs["type"] == "jpeg"
        assert page.screenshot_kwargs["quality"] == 95

    def test_png_readback_carries_no_quality_kwarg(self, tmp_path):
        capture, page = _capture_with_fake_page(tmp_path, readback="png")

        capture.render({"longitude": 1.0, "latitude": 2.0, "zoom": 12.0})

        assert page.screenshot_kwargs["type"] == "png"
        assert "quality" not in page.screenshot_kwargs

    def test_rejects_an_unknown_readback(self, tmp_path):
        with pytest.raises(ValueError, match="readback"):
            capture_mod.SceneCapture(tmp_path, 2, 2, readback="tiff")

    def test_determinism_check_uses_the_capture_own_readback_consistently(self, tmp_path):
        capture, _page = _capture_with_fake_page(tmp_path, readback="png")
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 12.0}

        assert capture_mod.determinism_check(capture, pose) is True


class TestSceneJsSettlePoll:
    _SCENE_JS = (Path(capture_mod.__file__).resolve().parent / "scene.js").read_text()

    def test_polls_every_60ms_via_a_single_named_constant(self):
        assert "SETTLE_POLL_MS = 60" in self._SCENE_JS
        assert self._SCENE_JS.count("setTimeout(watchdog, SETTLE_POLL_MS)") == 2
        assert "setTimeout(poll, 250)" not in self._SCENE_JS

    def test_only_idle_declares_a_pose_settled(self):
        # A timer that resolves on areTilesLoaded() alone can capture a canvas before
        # it paints, which breaks the determinism check; success must come from the
        # "idle" handler after a forced repaint.
        watchdog = self._SCENE_JS.split("function watchdog()")[1].split("\n      }\n")[0]
        assert "resolve(" not in watchdog
        on_idle = self._SCENE_JS.split("function onIdle()")[1].split("\n      }\n")[0]
        assert "resolve(" in on_idle
        assert self._SCENE_JS.index('map.once("idle", onIdle);\n      map.triggerRepaint();') > 0


class TestSceneJsLightContract:
    """scene.js runs only in a browser; assert on its text since it isn't executed here."""

    _SCENE_JS = (Path(capture_mod.__file__).resolve().parent / "scene.js").read_text()

    def test_sets_maplibre_scene_light(self):
        assert "map.setLight(" in self._SCENE_JS

    def test_sets_hillshade_exaggeration_from_light(self):
        assert "hillshadeExaggeration" in self._SCENE_JS
        assert '"hillshade-exaggeration"' in self._SCENE_JS

    def test_azimuth_only_branch_is_guarded(self):
        assert "light.elevation" in self._SCENE_JS


class TestSceneJsTravelledLine:
    """scene.js runs only in a browser; assert on its text since it isn't executed here."""

    _SCENE_JS = (Path(capture_mod.__file__).resolve().parent / "scene.js").read_text()

    def test_route_is_no_longer_loaded_as_a_file_source(self):
        assert "route.geojson" not in self._SCENE_JS
        assert "lineMetrics" not in self._SCENE_JS

    def test_the_old_progress_gradient_is_gone(self):
        assert "gradientForProgress" not in self._SCENE_JS
        assert "setRouteProgress" not in self._SCENE_JS
        assert "line-gradient" not in self._SCENE_JS
        assert "pose.progress" not in self._SCENE_JS

    def test_route_and_travelled_are_separate_geojson_sources(self):
        assert 'route: { type: "geojson"' in self._SCENE_JS
        assert 'travelled: { type: "geojson"' in self._SCENE_JS

    def test_travelled_line_layer_draws_from_the_travelled_source(self):
        assert 'id: "travelled-line"' in self._SCENE_JS
        assert '"line-color": ROUTE_BRIGHT_COLOR' in self._SCENE_JS

    def test_render_pose_no_longer_feeds_the_hidden_maplibre_sources(self):
        # setRoute/setTravelled feed the "route"/"travelled" GeoJSON sources,
        # whose only consumers are route-casing/route-line/travelled-line --
        # all three visibility:"none" (see
        # TestSceneJsRouteOverlay.test_draped_route_layers_are_hidden) since
        # drawRouteOverlay took over painting the real, visible line straight
        # from pose.route/pose.travelled in screen space (see
        # TestSceneJsRouteOverlay). Calling them every pose was pure retiling
        # cost for zero pixels: setTravelled has no dedup and retiles a
        # strictly-growing LineString every frame of a multi-hour film --
        # the cost per frame grows steadily as the travelled line grows.
        # Removing the feed cannot change an output pixel: drawRouteOverlay's own inputs
        # (pose.route/pose.travelled) are untouched.
        render_pose = self._SCENE_JS.split("function renderPose(")[1]
        assert "setTravelled(pose.travelled)" not in render_pose
        assert "setRoute(pose.route)" not in render_pose

    def test_set_route_skips_setdata_when_the_route_is_unchanged(self):
        # The renderer now sends pose.route on EVERY pose (a pose must fully
        # describe its own frame -- it is the frame cache's key), so scene.js
        # is what must avoid re-tiling an unchanged source every frame.
        set_route = self._SCENE_JS.split("function setRoute(")[1].split("\n  }\n")[0]
        assert "JSON.stringify" in set_route
        assert "return" in set_route
        assert "setData" in set_route


class TestSceneJsRouteOverlay:
    """The whole-route/travelled-line screen-space overlay that replaced the
    draped line layers (see docs/pipeline/41-travelled-line-minimum-width.md).
    Assertions that changed from an earlier
    OVERLAY_RADIUS_PX/drawTravelledOverlay version, and why:

    - the draped route-casing/route-line/travelled-line layers are now
      hidden (`visibility: "none"`) rather than left visible at a bumped
      paint width, since a screen-space overlay now draws them instead;
    - the entry point is renamed drawRouteOverlay (draws BOTH the route and
      the travelled line, not just a marker-radius slice of the travelled
      line), called from the same single settle-resolution path;
    - new assertions for the occlusion test (public map.project/unproject
      only) and the marker knockout (destination-out) that didn't exist
      before, since the earlier version had neither.
    """

    _SCENE_JS = (Path(capture_mod.__file__).resolve().parent / "scene.js").read_text()

    def test_draped_route_layers_are_hidden(self):
        for layer_id in ("route-casing", "route-line", "travelled-line"):
            layer_def = self._SCENE_JS.split(f'id: "{layer_id}"')[1][:200]
            assert 'visibility: "none"' in layer_def

    def test_render_pose_draws_the_overlay_on_the_settle_path_only(self):
        render_pose = self._SCENE_JS.split("function renderPose(")[1]
        assert "drawRouteOverlay(pose)" in render_pose
        assert "drawTravelledOverlay" not in self._SCENE_JS

    def test_occlusion_test_uses_only_public_project_unproject(self):
        assert "function isVertexOccluded(" in self._SCENE_JS
        # The raycast itself lives in occlusionMissMeters, which both
        # isVertexOccluded and vertexSample call.
        occlusion_fn = self._SCENE_JS.split("function occlusionMissMeters(")[1].split("\n  }\n")[0]
        assert "map.project(" in occlusion_fn
        assert "map.unproject(" in occlusion_fn

    def test_marker_is_knocked_out_of_the_overlay_after_stroking(self):
        overlay_fn = self._SCENE_JS.split("function drawRouteOverlay(")[1].split("\n  }\n")[0]
        assert 'globalCompositeOperation = "destination-out"' in overlay_fn
        assert 'getPaintProperty("marker", "circle-radius")' in overlay_fn

    def test_set_travelled_still_calls_setdata_every_pose(self):
        set_travelled = self._SCENE_JS.split("function setTravelled(")[1].split("\n  }\n")[0]
        assert "JSON.stringify" not in set_travelled
        assert "setData" in set_travelled

    def test_no_external_network_urls(self):
        assert "http://" not in self._SCENE_JS
        assert "https://" not in self._SCENE_JS


class TestSceneJsHeadingMarker:
    """scene.js runs only in a browser; assert on its text since it isn't executed here."""

    _SCENE_JS = (Path(capture_mod.__file__).resolve().parent / "scene.js").read_text()

    def test_symbol_layer_rotates_from_the_marker_heading_property(self):
        assert 'id: "marker-heading"' in self._SCENE_JS
        assert 'type: "symbol"' in self._SCENE_JS
        assert '"icon-rotate": ["get", "heading"]' in self._SCENE_JS
        assert '"icon-rotation-alignment": "map"' in self._SCENE_JS

    def test_symbol_layer_only_draws_when_marker_carries_a_heading(self):
        assert 'filter: ["has", "heading"]' in self._SCENE_JS

    def test_icon_is_drawn_in_page_and_added_before_ready_resolves(self):
        # No network fetch for the glyph, and map.addImage happens
        # synchronously in init(), ahead of the load/labels Promise.all that
        # resolves `ready` -- so frame 0 already has the image.
        assert "buildHeadingChevron" in self._SCENE_JS
        assert "getImageData(" in self._SCENE_JS
        init_body = self._SCENE_JS.split("function init(config)")[1].split("\n  }\n")[0]
        assert init_body.index("map.addImage(") < init_body.index("Promise.all(")

    def test_set_marker_takes_an_optional_heading(self):
        assert "function setMarker(marker, heading)" in self._SCENE_JS
        assert "properties.heading = heading" in self._SCENE_JS
        assert "setMarker(pose.marker || null, pose.marker_heading)" in self._SCENE_JS

    def test_no_external_network_urls(self):
        assert "http://" not in self._SCENE_JS
        assert "https://" not in self._SCENE_JS
        assert "light.intensity" in self._SCENE_JS


class _FakeLegCapture:
    """Records every pose it is asked to render and returns a solid frame."""

    def __init__(self, width: int, height: int, color: tuple[int, int, int] = (80, 90, 100)):
        self.width = width
        self.height = height
        self.color = color
        self.poses: list[dict] = []

    def __enter__(self) -> "_FakeLegCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def render(self, pose, light=None):
        self.poses.append(dict(pose))
        array = np.full((self.height, self.width, 3), self.color, dtype=np.uint8)
        return array, []


def _leg_segments(count: int = 90, step_s: float = 1.0):
    from gomoveshift.pipeline.route_story.metrics import RoutePoint

    return [
        [
            RoutePoint(i * step_s, 35.0 + i * 0.0004, -80.0 + i * 0.0006, 100.0 + i)
            for i in range(count)
        ]
    ]


class TestSceneMapRendererParallelCapture:
    """Fanning the CAPTURE step across workers must not change what is shown."""

    def _render(self, workers: int, capture_factory=None):
        from datetime import datetime, timezone

        from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer

        segments = _leg_segments()
        width, height = 320, 180
        map_height = int(height * 0.855)
        times = [float(i + 1) for i in range(60)]
        capture = _FakeLegCapture(width, map_height)
        factory = capture_factory or (lambda: _FakeLegCapture(width, map_height))
        renderer = SceneMapRenderer(
            capture=capture,
            trip_epoch=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            max_zoom=16.0,
            follow_zoom=14.0,
            fps=30.0,
            workers=workers,
            capture_factory=factory if workers > 1 else None,
        )
        frames = list(renderer.render_leg(segments, times, width, height))
        return frames, times

    def test_worker_fanout_renders_the_same_poses_in_the_same_order_as_serial(self):
        serial_frames, times = self._render(workers=1)
        parallel_frames, _ = self._render(workers=3)

        assert len(serial_frames) == len(parallel_frames) == len(times)
        for serial_image, parallel_image in zip(serial_frames, parallel_frames):
            assert np.array(serial_image).tolist() == np.array(parallel_image).tolist()

    def test_a_worker_failure_raises_instead_of_yielding_a_partial_film(self):
        calls = {"n": 0}

        class _FailingCapture(_FakeLegCapture):
            def render(self, pose, light=None):
                calls["n"] += 1
                if calls["n"] == 5:
                    raise RuntimeError("capture crashed")
                return super().render(pose, light=light)

        with pytest.raises(RuntimeError, match="capture crashed"):
            self._render(workers=3, capture_factory=lambda: _FailingCapture(320, int(180 * 0.855)))

    def test_a_long_leg_is_streamed_not_held_whole_in_memory(self):
        """A leg can be thousands of 6 MB frames: workers may only run a bounded
        distance ahead of the consumer, never render the whole leg up front."""
        from datetime import datetime, timezone

        from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer

        width, height = 320, 180
        map_height = int(height * 0.855)
        rendered = {"n": 0}

        class _CountingCapture(_FakeLegCapture):
            def render(self, pose, light=None):
                rendered["n"] += 1
                return super().render(pose, light=light)

        renderer = SceneMapRenderer(
            capture=_FakeLegCapture(width, map_height),
            trip_epoch=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            max_zoom=16.0,
            follow_zoom=14.0,
            fps=30.0,
            workers=2,
            capture_factory=lambda: _CountingCapture(width, map_height),
        )
        times = [float(i + 1) for i in range(200)]
        frames = renderer.render_leg(_leg_segments(), times, width, height)

        next(frames)
        import time as _time

        _time.sleep(0.5)  # give eager workers every chance to run ahead
        assert rendered["n"] < 40, (
            f"{rendered['n']} of 200 frames rendered before frame 1 was taken"
        )
        assert sum(1 for _ in frames) == 199
