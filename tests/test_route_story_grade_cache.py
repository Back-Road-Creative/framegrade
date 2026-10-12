"""A raster's graded pixels are computed once per machine, not once per render.

``layers._graded_raster_image`` grades each opening raster with the still-photo
grade. For a typical scene bundle that is 14 rasters / 298 MP at 2.7 s per MP --
13.5 minutes spent at the start of EVERY render (and again on every eviction
from the eight-entry memory cache) recomputing pixels that depend only on the
raster's bytes and on ``grade.py``. The disk cache keys on exactly those two.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import layers


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.delenv("GMS_ROUTE_STORY_GRADE_CACHE", raising=False)
    monkeypatch.delenv("GMS_ROUTE_STORY_GRADE_CACHE_MAX_GB", raising=False)
    layers._graded_raster_image.cache_clear()
    layers._raster_image.cache_clear()
    yield
    layers._graded_raster_image.cache_clear()
    layers._raster_image.cache_clear()


def _raster(path, colour=(140, 150, 160), size=(96, 48)):
    image = Image.new("RGB", size, colour)
    # A little structure, so the grade has edges to work on.
    for x in range(0, size[0], 8):
        image.putpixel((x, size[1] // 2), (30, 40, 50))
    image.save(path, quality=95)
    return path


def _counting_grade(monkeypatch):
    calls = []
    real = layers.grade.grade_still

    def counting(image, **kwargs):
        calls.append(1)
        return real(image, **kwargs)

    monkeypatch.setattr(layers.grade, "grade_still", counting)
    return calls


def _fresh_process():
    """What a new render sees: nothing in memory, the disk untouched."""
    layers._graded_raster_image.cache_clear()
    layers._raster_image.cache_clear()


def test_a_second_render_reuses_the_grade_from_disk(tmp_path, monkeypatch):
    source = _raster(tmp_path / "a.jpg")
    calls = _counting_grade(monkeypatch)

    first = layers._graded_raster_image(source)
    _fresh_process()
    second = layers._graded_raster_image(source)

    assert len(calls) == 1
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert second.mode == "RGB"


def test_the_key_is_the_bytes_not_the_path(tmp_path, monkeypatch):
    """Every render re-writes the bundle's opening rasters into a new temp
    dir, so a path key would never hit."""
    (tmp_path / "run1").mkdir()
    first_copy = _raster(tmp_path / "run1" / "band.jpg")
    second_copy = tmp_path / "run2" / "band.jpg"
    second_copy.parent.mkdir()
    second_copy.write_bytes(first_copy.read_bytes())
    calls = _counting_grade(monkeypatch)

    layers._graded_raster_image(first_copy)
    _fresh_process()
    layers._graded_raster_image(second_copy)

    assert len(calls) == 1


def test_different_pixels_are_graded_again(tmp_path, monkeypatch):
    calls = _counting_grade(monkeypatch)

    layers._graded_raster_image(_raster(tmp_path / "a.jpg", colour=(140, 150, 160)))
    _fresh_process()
    layers._graded_raster_image(_raster(tmp_path / "b.jpg", colour=(90, 120, 60)))

    assert len(calls) == 2


def test_an_edit_to_the_grade_invalidates_every_entry(tmp_path, monkeypatch):
    source = _raster(tmp_path / "a.jpg")
    calls = _counting_grade(monkeypatch)

    layers._graded_raster_image(source)
    _fresh_process()
    monkeypatch.setattr(layers, "_grade_code_identity", lambda: "a different grade.py")
    layers._graded_raster_image(source)

    assert len(calls) == 2


def test_an_unreadable_entry_is_regraded_not_raised(tmp_path, monkeypatch):
    source = _raster(tmp_path / "a.jpg")
    layers._graded_raster_image(source)
    for entry in (tmp_path / "intermediate" / "route-story-grade-cache").glob("*.npy"):
        entry.write_bytes(b"not an array")
    calls = _counting_grade(monkeypatch)
    _fresh_process()

    graded = layers._graded_raster_image(source)

    assert len(calls) == 1
    assert graded.size == (96, 48)


def test_the_cache_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_ROUTE_STORY_GRADE_CACHE", "0")
    source = _raster(tmp_path / "a.jpg")
    calls = _counting_grade(monkeypatch)

    layers._graded_raster_image(source)
    _fresh_process()
    layers._graded_raster_image(source)

    assert len(calls) == 2
    assert not (tmp_path / "intermediate" / "route-story-grade-cache").exists()


def test_the_oldest_entries_go_first_past_the_size_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_ROUTE_STORY_GRADE_CACHE_MAX_GB", str(20_000 / 1024**3))
    cache = tmp_path / "intermediate" / "route-story-grade-cache"
    sources = [
        _raster(tmp_path / f"{index}.jpg", colour=(40 * index, 100, 100)) for index in range(3)
    ]

    for source in sources:  # each entry is 96 x 48 x 3 = 13,824 bytes of pixels
        layers._graded_raster_image(source)

    remaining = list(cache.glob("*.npy"))
    assert len(remaining) == 1
    assert remaining[0] == layers._graded_cache_path(sources[-1])


def test_the_default_cache_dir_follows_the_cache_home(tmp_path, monkeypatch):
    from gomoveshift.pipeline.route_story import graded_cache

    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
    monkeypatch.setenv("GMS_CACHE_HOME", str(tmp_path / "cache-home"))

    assert graded_cache.cache_dir() == tmp_path / "cache-home" / "route-story" / "graded-rasters"

    # The intermediate-dir override still wins over the cache home.
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    assert graded_cache.cache_dir() == tmp_path / "intermediate" / "route-story-grade-cache"
