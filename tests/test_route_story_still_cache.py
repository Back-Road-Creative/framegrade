"""A photo is graded once per machine, and a render's photos are graded in parallel.

``hud._load_still`` grades the full-resolution still (haze, exposure, vibrance):
roughly a minute for one 20 MP camera still. Still selection loaded
every candidate serially at startup, then every kept photo's beat
loaded and graded it AGAIN -- ~27 minutes of each render spent on pixels that
depend only on the photo's bytes and the code that grades it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import hud, still_selection


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "intermediate"))
    monkeypatch.delenv("GMS_ROUTE_STORY_GRADE_CACHE", raising=False)


def _photo(path: Path, colour=(120, 140, 160), size=(160, 120)) -> Path:
    image = Image.new("RGB", size, colour)
    for x in range(0, size[0], 7):
        image.putpixel((x, size[1] // 3), (20, 30, 40))
    image.save(path, quality=95)
    return path


def _counting_grade(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = hud.grade.grade_still

    def counting(image, **kwargs):
        calls.append(1)
        return real(image, **kwargs)

    monkeypatch.setattr(hud.grade, "grade_still", counting)
    return calls


def test_a_photo_is_graded_once_then_read_back_from_disk(tmp_path, monkeypatch) -> None:
    photo = _photo(tmp_path / "a.jpg")
    calls = _counting_grade(monkeypatch)
    first = hud._load_still(photo)
    second = hud._load_still(photo)
    assert len(calls) == 1
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_the_cache_keys_on_the_photo_bytes_not_its_path(tmp_path, monkeypatch) -> None:
    photo = _photo(tmp_path / "a.jpg")
    copy = tmp_path / "elsewhere" / "renamed.jpg"
    copy.parent.mkdir()
    copy.write_bytes(photo.read_bytes())
    calls = _counting_grade(monkeypatch)
    hud._load_still(photo)
    hud._load_still(copy)
    assert len(calls) == 1


def test_a_different_photo_is_graded_on_its_own(tmp_path, monkeypatch) -> None:
    calls = _counting_grade(monkeypatch)
    hud._load_still(_photo(tmp_path / "a.jpg"))
    hud._load_still(_photo(tmp_path / "b.jpg", colour=(200, 90, 40)))
    assert len(calls) == 2


def test_the_still_source_is_part_of_the_key(tmp_path, monkeypatch) -> None:
    photo = _photo(tmp_path / "a.jpg")
    calls = _counting_grade(monkeypatch)
    hud._load_still(photo, source="developed")
    hud._load_still(photo, source="camera")
    assert len(calls) == 2


def test_opting_out_grades_every_time(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_GRADE_CACHE", "0")
    photo = _photo(tmp_path / "a.jpg")
    calls = _counting_grade(monkeypatch)
    hud._load_still(photo)
    hud._load_still(photo)
    assert len(calls) == 2


class _SerialPool:
    """Stands in for the process pool: records that it was used, runs in-process."""

    used_with: list[int] = []

    def __init__(self, max_workers: int, **_: object) -> None:
        _SerialPool.used_with.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def map(self, fn, *iterables):
        return map(fn, *iterables)


def test_candidates_are_loaded_through_a_worker_pool_in_input_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_STILL_WORKERS", "3")
    _SerialPool.used_with = []
    monkeypatch.setattr(still_selection, "ProcessPoolExecutor", _SerialPool)
    colours = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (180, 180, 60)]
    paths = [_photo(tmp_path / f"p{index}.jpg", colour) for index, colour in enumerate(colours)]

    loaded = still_selection.load_stills(paths, still_source="developed")

    assert _SerialPool.used_with == [3]
    monkeypatch.setenv("GMS_ROUTE_STORY_GRADE_CACHE", "0")
    for path, image in zip(paths, loaded):
        assert np.array_equal(np.asarray(image), np.asarray(hud._load_still(path)))


def test_one_worker_or_one_photo_skips_the_pool(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_STILL_WORKERS", "1")
    _SerialPool.used_with = []
    monkeypatch.setattr(still_selection, "ProcessPoolExecutor", _SerialPool)
    paths = [_photo(tmp_path / "a.jpg"), _photo(tmp_path / "b.jpg", colour=(9, 9, 9))]
    assert len(still_selection.load_stills(paths, still_source="developed")) == 2
    monkeypatch.setenv("GMS_ROUTE_STORY_STILL_WORKERS", "4")
    assert len(still_selection.load_stills(paths[:1], still_source="developed")) == 1
    assert _SerialPool.used_with == []
