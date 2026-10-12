"""The opening zoom may not go soft partway down the imagery ladder.

A render once passed every absolute opening floor and was still
rejected for being "very dark and fuzzy when zooming in". Measured on that
film's own opening frames, sharpness by ladder fraction ran 2211, 4571,
**1305**, 4778, **1237**, 8165, 11899, 12910, 8585 -- two rungs roughly a
tenth as sharp as their neighbours, where the ladder hands over to a coarser
raster. No single-frame floor can see that: 1237 clears the 50.0 blur floor
by 25x. What gives it away is the ladder's own shape, so the check is
relative to the ladder's median and needs no per-bundle magic number.
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoveshift.pipeline.route_story.frame_qa import (
    FrameQaThresholds,
    check_opening_ladder,
)

RNG = np.random.default_rng(20260920)
SIZE = (120, 160)


def _detailed(scale: float = 1.0) -> np.ndarray:
    """A frame whose sharpness scales with ``scale`` -- noise, then blurred."""
    base = RNG.integers(0, 256, size=(*SIZE, 3), dtype=np.uint8).astype(np.float64)
    flat = np.full((*SIZE, 3), 128.0)
    return np.clip(flat + (base - flat) * scale, 0, 255).astype(np.uint8)


def _sharpness(rgb: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


class TestARungFarSofterThanItsNeighboursIsCaught:
    def test_a_ladder_that_stays_sharp_has_no_problems(self):
        ladder = [_detailed(1.0) for _ in range(7)]

        assert check_opening_ladder(ladder) == ()

    def test_one_soft_rung_among_sharp_ones_is_reported_with_its_index(self):
        ladder = [_detailed(1.0) for _ in range(7)]
        ladder[3] = _detailed(0.15)

        problems = check_opening_ladder(ladder)

        assert len(problems) == 1, problems
        assert "rung 3" in problems[0]
        assert "sharpness" in problems[0]

    def test_the_render_12_ladder_shape_is_refused(self):
        """The real measured shape, reproduced by relative sharpness."""
        measured = [2211.4, 4570.9, 1305.1, 4778.3, 1237.2, 8165.4, 11899.4, 12910.7, 8585.5]
        median = float(np.median(measured))
        ladder = [_detailed(min(1.0, (value / median) ** 0.5)) for value in measured]

        problems = check_opening_ladder(ladder)

        assert problems, "the two soft rungs must be refused"
        assert any("rung 2" in p for p in problems)
        assert any("rung 4" in p for p in problems)

    def test_a_uniformly_soft_ladder_is_left_to_the_absolute_floor(self):
        """Relative shape says nothing when every rung is equally soft --
        that is check_opening_frame's sharpness floor's job, not this one's."""
        ladder = [_detailed(0.12) for _ in range(6)]

        assert check_opening_ladder(ladder) == ()

    def test_a_ladder_too_short_to_have_a_shape_is_not_judged(self):
        assert check_opening_ladder([_detailed(1.0), _detailed(0.1)]) == ()
        assert check_opening_ladder([]) == ()

    def test_the_ratio_is_configurable_and_sane_by_default(self):
        assert 0.0 < FrameQaThresholds().min_opening_ladder_sharpness_ratio < 1.0

        ladder = [_detailed(1.0) for _ in range(7)]
        ladder[2] = _detailed(0.45)
        lenient = FrameQaThresholds(min_opening_ladder_sharpness_ratio=0.01)

        assert check_opening_ladder(ladder, lenient) == ()
        assert check_opening_ladder(ladder) != ()

    def test_a_non_image_rung_raises_rather_than_scoring_it(self):
        with pytest.raises(ValueError):
            check_opening_ladder([_detailed(1.0), np.zeros((4, 4), dtype=np.uint8)])


def test_the_film_walks_the_whole_ladder_not_five_fractions():
    """Five samples can straddle a soft band entirely; the ladder is checked
    at enough rungs that a handover cannot hide between two of them."""
    from pathlib import Path

    from gomoveshift.pipeline.route_story import film

    source = Path(film.__file__).read_text(encoding="utf-8")
    assert "OPENING_LADDER_FRACTIONS" in source
    assert len(film.OPENING_LADDER_FRACTIONS) >= 9
    assert film.OPENING_LADDER_FRACTIONS[0] == 0.0
    assert film.OPENING_LADDER_FRACTIONS[-1] == 1.0
    assert list(film.OPENING_LADDER_FRACTIONS) == sorted(film.OPENING_LADDER_FRACTIONS)
