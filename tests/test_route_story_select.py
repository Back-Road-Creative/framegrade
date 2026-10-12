"""Tests for the route-story best-of-look-alike-group photo selector."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.route_story.select import (
    PhotoCandidate,
    SelectionOptions,
    select_best_photos,
)


def _textured_image(seed: int, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    """A synthetic scene with plenty of ORB-friendly texture."""
    rng = np.random.default_rng(seed)
    height, width = size
    canvas = np.full((height, width, 3), 190, dtype=np.uint8)
    for _ in range(60):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        radius = int(rng.integers(6, 45))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(canvas, (x, y), radius, color, -1)
    return canvas


def _reframe(image: np.ndarray, dx: int = 14, dy: int = 9, scale: float = 1.03) -> np.ndarray:
    """A slight re-shoot: small pan plus a touch of zoom, same content."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def _blurred(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (9, 9), 2.5)


def _flat_color(
    color: tuple[int, int, int], seed: int = 0, size: tuple[int, int] = (200, 300)
) -> np.ndarray:
    """A near-textureless frame (hazy sky) that starves ORB of keypoints."""
    canvas = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    noise = rng.integers(-1, 2, size=canvas.shape, dtype=np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_shifted_and_zoomed_copy_groups_together():
    base = _textured_image(seed=1)
    candidates = [
        PhotoCandidate("a", base, captured_at=0.0),
        PhotoCandidate("b", _reframe(base), captured_at=5.0),
    ]
    result = select_best_photos(candidates)
    assert len(result.groups) == 1
    assert result.groups[0].members == ("a", "b")


def test_blurred_copy_loses_to_sharp_original():
    base = _textured_image(seed=2)
    candidates = [
        PhotoCandidate("sharp", base, captured_at=0.0),
        PhotoCandidate("blurred", _blurred(_reframe(base)), captured_at=4.0),
    ]
    result = select_best_photos(candidates)
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.winner == "sharp"
    assert group.losers == ("blurred",)
    assert group.scores["sharp"].sharpness > group.scores["blurred"].sharpness


def test_different_texture_ten_seconds_apart_both_kept():
    candidates = [
        PhotoCandidate("first", _textured_image(seed=10), captured_at=0.0),
        PhotoCandidate("second", _textured_image(seed=99), captured_at=10.0),
    ]
    result = select_best_photos(candidates)
    assert len(result.groups) == 2
    assert set(result.kept) == {"first", "second"}


def test_far_apart_placements_not_grouped_despite_visual_match():
    base = _textured_image(seed=3)
    candidates = [
        PhotoCandidate("near", base, captured_at=0.0, placement=(45.0, -70.0)),
        PhotoCandidate("far", _reframe(base), captured_at=3.0, placement=(45.01, -70.0)),  # ~1.1km
    ]
    result = select_best_photos(candidates)
    assert len(result.groups) == 2
    assert set(result.kept) == {"near", "far"}


def test_keep_override_survives_even_as_group_loser():
    base = _textured_image(seed=4)
    candidates = [
        PhotoCandidate("sharp", base, captured_at=0.0),
        PhotoCandidate("blurred", _blurred(_reframe(base)), captured_at=2.0),
    ]
    options = SelectionOptions(keep=frozenset({"blurred"}))
    result = select_best_photos(candidates, options)
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.winner == "sharp"
    assert set(group.kept) == {"sharp", "blurred"}
    assert "kept override applied" in group.reason
    assert set(result.kept) == {"sharp", "blurred"}


def test_disabled_returns_everything():
    base = _textured_image(seed=5)
    candidates = [
        PhotoCandidate("a", base, captured_at=0.0),
        PhotoCandidate("b", _reframe(base), captured_at=1.0),
    ]
    result = select_best_photos(candidates, SelectionOptions(enabled=False))
    assert set(result.kept) == {"a", "b"}
    assert len(result.groups) == 2


def test_too_few_keypoints_falls_back_to_histogram():
    candidates = [
        PhotoCandidate("hazy1", _flat_color((150, 200, 230), seed=1), captured_at=0.0),
        PhotoCandidate("hazy2", _flat_color((152, 198, 228), seed=2), captured_at=8.0),
    ]
    result = select_best_photos(candidates)
    assert len(result.groups) == 1
    assert result.groups[0].members == ("hazy1", "hazy2")


def test_chain_links_three_into_one_group():
    base = _textured_image(seed=6)
    candidates = [
        PhotoCandidate("a", base, captured_at=0.0),
        PhotoCandidate("b", _reframe(base, dx=10, dy=6), captured_at=30.0),
        PhotoCandidate("c", _reframe(base, dx=-8, dy=-5, scale=1.05), captured_at=60.0),
    ]
    result = select_best_photos(candidates)
    assert len(result.groups) == 1
    assert result.groups[0].members == ("a", "b", "c")


def test_order_independent_deterministic():
    base = _textured_image(seed=7)
    candidates = [
        PhotoCandidate("first", base, captured_at=0.0),
        PhotoCandidate("second", _reframe(base), captured_at=3.0),
        PhotoCandidate("third", _textured_image(seed=77), captured_at=200.0),
    ]
    forward = select_best_photos(candidates)
    backward = select_best_photos(list(reversed(candidates)))
    assert forward.kept == backward.kept
    assert [g.members for g in forward.groups] == [g.members for g in backward.groups]


def test_result_is_json_serialisable():
    import json

    base = _textured_image(seed=8)
    candidates = [
        PhotoCandidate("a", base, captured_at=0.0),
        PhotoCandidate("b", _reframe(base), captured_at=2.0),
    ]
    result = select_best_photos(candidates)
    payload = json.dumps(result.to_dict())
    assert "kept" in payload


def test_a_handful_of_clipped_pixels_never_decides_a_sharpness_tie():
    """A few blown pixels are noise, not an exposure fault: colour decides."""
    vivid = _textured_image(seed=21)
    vivid[10:22, 10:22] = 255  # ~0.05% of the frame clipped
    # Same luma, less colour, and not one clipped pixel more than the scene has.
    gray = cv2.cvtColor(_textured_image(seed=21), cv2.COLOR_RGB2GRAY)[..., None]
    dull = (0.5 * _textured_image(seed=21) + 0.5 * gray).astype(np.uint8)

    result = select_best_photos(
        [
            PhotoCandidate("vivid", vivid, captured_at=0.0),
            PhotoCandidate("dull", dull, captured_at=5.0),
        ]
    )

    assert result.groups[0].members == ("vivid", "dull")
    assert result.groups[0].winner == "vivid"


# A directory of real camera RAW files plus a JSON manifest describing one burst.
# Neither is configured by default: set GMS_TEST_RAW_DIR and GMS_TEST_RAW_MANIFEST
# to run the real-burst test, which is skipped otherwise. The manifest holds
# {"burst": {"all": [<file name>, ...] in shooting order,
#            "view_a": [<names of one look-alike view>],
#            "view_b": [<names of a second view>],
#            "apart_1": <name>, "apart_2": <name>,
#            "alone": <name>}}
# where apart_1 and apart_2 are shot seconds apart but show different views, and
# alone follows the last frame of view_b by minutes and stands by itself.
_RAW_DIR = Path(os.environ.get("GMS_TEST_RAW_DIR", "/nonexistent-raw-fixtures"))


def _load_burst_manifest() -> dict:
    path = os.environ.get("GMS_TEST_RAW_MANIFEST")
    if not path:
        return {}
    try:
        return dict(json.loads(Path(path).read_text()).get("burst", {}))
    except (OSError, ValueError, AttributeError):
        return {}


_BURST = _load_burst_manifest()


def _burst_present() -> bool:
    keys = ("all", "view_a", "view_b", "apart_1", "apart_2", "alone")
    return all(k in _BURST for k in keys) and all(
        (_RAW_DIR / name).exists() for name in _BURST["all"]
    )


@pytest.mark.skipif(
    not _burst_present(),
    reason="set GMS_TEST_RAW_DIR and GMS_TEST_RAW_MANIFEST (burst) to run",
)
def test_real_burst_matches_human_judgement():
    import rawpy
    from PIL import Image

    candidates = []
    for name in _BURST["all"]:
        path = _RAW_DIR / name
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        image = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        captured_at = float(os.stat(path).st_mtime)
        candidates.append(PhotoCandidate(Path(name).stem, image, captured_at=captured_at))

    result = select_best_photos(candidates)
    groups_by_member = {name: g for g in result.groups for name in g.members}
    stem = lambda n: Path(n).stem  # noqa: E731

    # One view's frames form a single group.
    view_a = [stem(n) for n in _BURST["view_a"]]
    assert set(groups_by_member[view_a[0]].members) == set(view_a)

    # A second view forms another group.
    view_b = [stem(n) for n in _BURST["view_b"]]
    assert set(groups_by_member[view_b[0]].members) == set(view_b)

    # Two frames seconds apart that show different views: both kept, separate groups.
    first, second = stem(_BURST["apart_1"]), stem(_BURST["apart_2"])
    assert first in result.kept
    assert second in result.kept
    assert groups_by_member[first] is not groups_by_member[second]

    # A frame minutes after the last view stands alone.
    alone = stem(_BURST["alone"])
    assert groups_by_member[alone].members == (alone,)
    assert alone in result.kept
