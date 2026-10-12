"""Tests for the on-disk captured-frame cache keyed on pose/bundle identity."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story.scene import capture as capture_mod
from gomoveshift.pipeline.route_story.scene import frame_cache as frame_cache_mod


def _write_bundle(root: Path, imagery_maxzoom: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "route-story-bundle/1",
                "imagery": {"minzoom": 0, "maxzoom": imagery_maxzoom},
                "terrain": {"minzoom": 0, "maxzoom": 0, "encoding": "mapbox"},
            }
        )
    )
    return root


class TestFrameCacheDir:
    def test_default_dir_is_under_home_cache_never_tmp(self, monkeypatch):
        monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
        monkeypatch.delenv("GMS_CACHE_HOME", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        path = str(frame_cache_mod.cache_dir())
        assert "/tmp" not in path
        assert ".cache/gomoveshift/route-story/frame-cache" in path

    def test_default_dir_follows_the_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
        monkeypatch.setenv("GMS_CACHE_HOME", str(tmp_path / "cache-home"))
        assert (
            frame_cache_mod.cache_dir() == tmp_path / "cache-home" / "route-story" / "frame-cache"
        )
        # An explicit intermediate dir still wins over the cache home.
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "fast-disk"))
        assert frame_cache_mod.cache_dir() == tmp_path / "fast-disk" / "route-story-frame-cache"

    def test_gms_intermediate_dir_env_relocates_the_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "fast-disk"))
        path = frame_cache_mod.cache_dir()
        assert path == tmp_path / "fast-disk" / "route-story-frame-cache"


class TestFrameCacheOptOut:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("GMS_ROUTE_STORY_FRAME_CACHE", raising=False)
        assert frame_cache_mod.is_enabled() is True

    def test_env_zero_disables(self, monkeypatch):
        monkeypatch.setenv("GMS_ROUTE_STORY_FRAME_CACHE", "0")
        assert frame_cache_mod.is_enabled() is False


class TestFrameCacheRoundTrip:
    def _cache(self, tmp_path, bundle_root):
        return frame_cache_mod.FrameCache(
            bundle_root=bundle_root, width=4, height=3, cache_root=tmp_path / "cache"
        )

    def _array(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, size=(3, 4, 3), dtype=np.uint8)

    def test_miss_on_empty_cache(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}

        assert cache.get(pose) is None

    def test_hit_returns_the_stored_array_and_labels(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()
        labels = [{"name": "Test Peak", "kind": "summit"}]

        cache.put(pose, array, labels)
        hit = cache.get(pose)

        assert hit is not None
        got_array, got_labels = hit
        assert np.array_equal(got_array, array)
        assert got_labels == labels

    def test_changed_pose_misses(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        array = self._array()
        cache.put({"longitude": 1.0, "latitude": 2.0, "zoom": 10}, array, [])

        assert cache.get({"longitude": 1.5, "latitude": 2.0, "zoom": 10}) is None

    def test_changed_light_misses(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10, "light": {"azimuth": 90}}
        array = self._array()
        cache.put(pose, array, [])

        other_pose = {**pose, "light": {"azimuth": 91}}
        assert cache.get(other_pose) is None

    def test_changed_size_misses(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()
        cache_a = frame_cache_mod.FrameCache(
            bundle_root=bundle, width=4, height=3, cache_root=tmp_path / "cache"
        )
        cache_a.put(pose, array, [])
        cache_b = frame_cache_mod.FrameCache(
            bundle_root=bundle, width=8, height=6, cache_root=tmp_path / "cache"
        )

        assert cache_b.get(pose) is None

    def test_changed_bundle_manifest_misses(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()
        cache = self._cache(tmp_path, bundle)
        cache.put(pose, array, [])

        _write_bundle(bundle, imagery_maxzoom=5)  # rewrite manifest, different bytes
        cache_after_change = self._cache(tmp_path, bundle)

        assert cache_after_change.get(pose) is None

    def test_corrupt_entry_is_a_miss_not_a_raise(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()
        cache.put(pose, array, [{"name": "x"}])

        key = cache._key_for(pose)
        image_path = cache._image_path(key)
        image_path.write_bytes(b"not a valid png")

        assert cache.get(pose) is None

    def test_corrupt_entry_is_overwritten_by_the_next_put(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()
        key = cache._key_for(pose)
        cache._image_path(key).parent.mkdir(parents=True, exist_ok=True)
        cache._image_path(key).write_bytes(b"garbage")

        cache.put(pose, array, [])

        hit = cache.get(pose)
        assert hit is not None
        assert np.array_equal(hit[0], array)

    def test_truncated_labels_json_is_a_miss(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()
        cache.put(pose, array, [{"name": "x"}])

        key = cache._key_for(pose)
        cache._labels_path(key).write_text('{"labels": [')

        assert cache.get(pose) is None

    def test_writes_are_atomic_no_leftover_temp_files(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        cache.put(pose, self._array(), [])

        leftovers = list((tmp_path / "cache").rglob("*.tmp*"))
        assert leftovers == []


def _write_scene_files(scene_dir: Path, scene_js: bytes = b"// scene v1") -> Path:
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "scene.js").write_bytes(scene_js)
    (scene_dir / "index.html").write_bytes(b"<html></html>")
    return scene_dir


class TestFrameCacheKeysOnSceneCodeAndCaptureSettings:
    def _array(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(3, 4, 3), dtype=np.uint8)

    def test_changed_scene_js_bytes_misses(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        scene_dir = _write_scene_files(tmp_path / "scene-v1")
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()

        cache_v1 = frame_cache_mod.FrameCache(
            bundle_root=bundle,
            width=4,
            height=3,
            cache_root=tmp_path / "cache",
            scene_dir=scene_dir,
        )
        cache_v1.put(pose, array, [])

        scene_dir_v2 = _write_scene_files(tmp_path / "scene-v2", scene_js=b"// scene v2, edited")
        cache_v2 = frame_cache_mod.FrameCache(
            bundle_root=bundle,
            width=4,
            height=3,
            cache_root=tmp_path / "cache",
            scene_dir=scene_dir_v2,
        )

        assert cache_v2.get(pose) is None

    def test_changed_readback_misses(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()

        cache_jpeg = frame_cache_mod.FrameCache(
            bundle_root=bundle, width=4, height=3, cache_root=tmp_path / "cache", readback="jpeg"
        )
        cache_jpeg.put(pose, array, [])

        cache_png = frame_cache_mod.FrameCache(
            bundle_root=bundle, width=4, height=3, cache_root=tmp_path / "cache", readback="png"
        )

        assert cache_png.get(pose) is None

    def test_changed_vendor_directory_name_misses(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 10}
        array = self._array()
        (bundle / "vendor" / "maplibre-gl-6.10.0").mkdir(parents=True)

        cache = frame_cache_mod.FrameCache(
            bundle_root=bundle, width=4, height=3, cache_root=tmp_path / "cache"
        )
        cache.put(pose, array, [])

        (bundle / "vendor" / "maplibre-gl-6.10.0").rmdir()
        (bundle / "vendor" / "maplibre-gl-6.11.0").mkdir(parents=True)
        cache_bumped = frame_cache_mod.FrameCache(
            bundle_root=bundle, width=4, height=3, cache_root=tmp_path / "cache"
        )

        assert cache_bumped.get(pose) is None


class TestFrameCachePruning:
    def _array(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(20, 20, 3), dtype=np.uint8)

    def _cache(self, tmp_path, bundle_root, **kwargs):
        return frame_cache_mod.FrameCache(
            bundle_root=bundle_root, width=20, height=20, cache_root=tmp_path / "cache", **kwargs
        )

    def test_over_cap_dir_prunes_oldest_pairs_under_the_cap(self, tmp_path, monkeypatch):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        array = self._array()
        for i in range(10):
            cache.put({"i": i}, array, [])
            time.sleep(0.01)  # distinct mtimes to order oldest-first

        total_before = sum(f.stat().st_size for f in (tmp_path / "cache").glob("*"))
        monkeypatch.setenv(
            "GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB", str(total_before * 0.5 / (1024**3))
        )

        pruning_cache = self._cache(tmp_path, bundle)  # prune runs at construction
        total_after = sum(f.stat().st_size for f in (tmp_path / "cache").glob("*"))

        assert total_after < total_before
        assert pruning_cache.get({"i": 0}) is None  # oldest evicted
        assert pruning_cache.get({"i": 9}) is not None  # newest kept

    def test_a_hit_refreshes_mtime_so_reuse_keeps_an_entry_alive(self, tmp_path):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        array = self._array()
        cache.put({"i": "kept-fresh"}, array, [])
        key = cache._key_for({"i": "kept-fresh"})
        old_time = 1_000_000_000.0  # a fixed moment in 2001: no clock read needed
        import os

        os.utime(cache._image_path(key), (old_time, old_time))
        os.utime(cache._labels_path(key), (old_time, old_time))

        cache.get({"i": "kept-fresh"})

        assert cache._image_path(key).stat().st_mtime > old_time

    def test_concurrent_delete_during_prune_is_tolerated(self, tmp_path, monkeypatch):
        bundle = _write_bundle(tmp_path / "bundle")
        cache = self._cache(tmp_path, bundle)
        array = self._array()
        cache.put({"i": 0}, array, [])
        key = cache._key_for({"i": 0})
        cache._image_path(key).unlink()  # simulate another worker already deleted it

        monkeypatch.setenv("GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB", "0")
        cache._prune_quietly()  # must not raise despite the missing image file


class TestFrameCachePutNeverFails:
    def test_a_put_that_raises_oserror_does_not_propagate_from_render(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "cache-root"))
        bundle = _write_bundle(tmp_path / "bundle")
        capture = capture_mod.SceneCapture(bundle, 2, 2)
        capture._page = _FakePage()
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 12.0}

        cache = capture._get_frame_cache()
        assert cache is not None
        monkeypatch.setattr(
            cache,
            "_atomic_write_image",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )

        array, labels = capture.render(pose)  # must not raise

        assert array.shape == (2, 2, 3)


class TestSceneCaptureUsesFrameCache:
    """A cache hit must never touch the browser page."""

    def _bundle(self, tmp_path):
        return _write_bundle(tmp_path / "bundle")

    def test_second_render_of_same_pose_makes_no_browser_calls(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "cache-root"))
        bundle = self._bundle(tmp_path)
        capture = capture_mod.SceneCapture(bundle, 2, 2)
        page = _FakePage()
        capture._page = page
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 12.0}

        first_array, first_labels = capture.render(pose)
        assert len(page.calls) > 0

        capture._page = None  # a cache hit must not need a page at all
        second_array, second_labels = capture.render(pose)

        assert np.array_equal(first_array, second_array)
        assert first_labels == second_labels

    def test_opt_out_env_disables_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "cache-root"))
        monkeypatch.setenv("GMS_ROUTE_STORY_FRAME_CACHE", "0")
        bundle = self._bundle(tmp_path)
        capture = capture_mod.SceneCapture(bundle, 2, 2)
        capture._page = _FakePage()
        pose = {"longitude": 1.0, "latitude": 2.0, "zoom": 12.0}

        capture.render(pose)

        capture._page = None
        with pytest.raises(AttributeError):
            capture.render(pose)


class _FakePage:
    """Local stand-in mirroring test_route_story_scene._FakePage, kept import-free."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def evaluate(self, script: str, arg: object = None) -> object:
        self.calls.append((script, arg))
        if "getBoundingClientRect" in script:
            return {"x": 0, "y": 0, "width": 2, "height": 2}
        return {"labels": [{"name": "Test Peak"}]}

    def screenshot(self, **kwargs: object) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), (5, 6, 7)).save(buffer, format="PNG")
        return buffer.getvalue()


class TestFrameCacheReviewHardening:
    def _array(self) -> np.ndarray:
        return np.zeros((3, 4, 3), dtype=np.uint8)

    def test_an_unparseable_size_cap_never_fails_construction(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMS_ROUTE_STORY_FRAME_CACHE_MAX_GB", "forty")
        bundle = _write_bundle(tmp_path / "bundle")

        cache = frame_cache_mod.FrameCache(bundle, 4, 3, cache_root=tmp_path / "cache")
        cache.put({"i": 0}, self._array(), [])
        cache._prune_quietly()  # the cap is read here; a bad value must not raise

        assert cache.get({"i": 0}) is not None

    def test_a_manifest_id_never_shadows_a_change_in_bundle_content(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        manifest = {"id": "same-id", "files": {"tiles/0/0/0.jpg": "aaa"}}
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        before = frame_cache_mod.FrameCache(bundle, 4, 3, cache_root=tmp_path / "cache")
        before.put({"i": 0}, self._array(), [])

        manifest["files"]["tiles/0/0/0.jpg"] = "bbb"  # a re-prepared tile, same id
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        after = frame_cache_mod.FrameCache(bundle, 4, 3, cache_root=tmp_path / "cache")

        assert after.get({"i": 0}) is None
