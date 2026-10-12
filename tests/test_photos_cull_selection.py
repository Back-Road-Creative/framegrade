"""Cull scores gate photo selection (soft, clipped, near-duplicate)."""

import cv2
import numpy as np
import pytest
import yaml

from gomoveshift.pipeline.photos import cull
from gomoveshift.pipeline.photos.selector import PhotoSelector


def _blocks(seed: int) -> np.ndarray:
    """Sharp-edged random block image (high Laplacian variance, seed-distinct pHash)."""
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return cv2.resize(grid, (256, 256), interpolation=cv2.INTER_NEAREST)


def _write(path, rgb):
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    images = root / "static" / "images" / "journeys" / "trip"
    images.mkdir(parents=True)
    sharp = _blocks(1)
    clipped = _blocks(2)
    clipped[:, :180] = 255  # ~70% blown highlights, edge detail kept elsewhere
    frames = {
        "a-dup-soft.jpg": cv2.GaussianBlur(sharp, (0, 0), 0.6),
        "b-dup-sharp.jpg": sharp,
        "c-soft.jpg": cv2.GaussianBlur(_blocks(3), (0, 0), 12),
        "d-clipped.jpg": clipped,
        "e-other.jpg": _blocks(4),
    }
    for name, rgb in frames.items():
        _write(images / name, rgb)
    fm = {
        "title": "Trip",
        "days": [{"day_number": 1, "date": "2025-01-01", "photos": [{"photos": list(frames)}]}],
    }
    content = root / "content" / "journeys"
    content.mkdir(parents=True)
    (content / "trip.md").write_text(f"---\n{yaml.safe_dump(fm)}---\n")
    return root, tmp_path / "state.json"


def test_selector_skips_soft_and_duplicate_photos(site):
    root, state = site
    selector = PhotoSelector(site_root=root, state_path=state)

    picked = selector.select_batch(2)

    assert [p.filename for p in picked] == ["b-dup-sharp.jpg", "e-other.jpg"]
    skips = {s.photo_key: s for s in selector.cull_skips}
    assert "soft" in skips["trip/c-soft.jpg"].reason
    assert "highlight" in skips["trip/d-clipped.jpg"].reason
    assert "duplicate" in skips["trip/a-dup-soft.jpg"].reason
    assert skips["trip/c-soft.jpg"].score.sharpness < cull.CullFloors().min_sharpness
    # the displaced duplicate is not consumed: it stays selectable later
    held = set(selector.release_pending())
    assert "trip/a-dup-soft.jpg" not in held


def test_cull_decision_reasons():
    floors = cull.CullFloors()
    ok = cull.CullScore(500.0, 0.0, 0.0, 0)
    assert cull.cull_decision(ok, floors=floors).keep
    soft = cull.cull_decision(cull.CullScore(1.0, 0.0, 0.0, 0), floors=floors)
    assert not soft.keep and "soft" in soft.reason
    blown = cull.cull_decision(cull.CullScore(500.0, 0.0, 0.9, 0), floors=floors)
    assert not blown.keep and "highlight" in blown.reason
    crushed = cull.cull_decision(cull.CullScore(500.0, 0.99, 0.0, 0), floors=floors)
    assert not crushed.keep and "shadow" in crushed.reason


def test_unreadable_photo_is_kept(site):
    root, state = site
    for p in (root / "static" / "images" / "journeys" / "trip").iterdir():
        p.write_bytes(b"not an image")
    picked = PhotoSelector(site_root=root, state_path=state).select_batch(2)
    assert len(picked) == 2
