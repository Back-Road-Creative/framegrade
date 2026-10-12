"""GPS-optional scoring path for UnifiedSegmentScorer.


When ``gps_data is None``, the two GPS-derived signals (``speed_variance``,
weight 0.15; ``scenic_score``, weight 0.20) collapse to zero. Without the
redistribution path, 35% of the composite mass vanishes and Insta360 / any
GPS-less source produces composites in [0, 0.65] — too compressed for the
existing percentile cutoffs. ``redistribute_gps_weights`` is the
construction-correct fix: GPS keys → 0.0, remaining keys scaled by
``1 / visual_weight`` so the new dict still sums to exactly 1.0.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.analysis.unified_scorer import (
    COL_COMPOSITE,
    GPS_SIGNAL_NAMES,
    NUM_COLUMNS,
    UnifiedSegmentScorer,
    redistribute_gps_weights,
)


def _default_weights() -> dict[str, float]:
    """Default 9-signal weights (post brightness PR) summing to 1.0."""
    return {
        "motion_energy": 0.16,
        "speed_variance": 0.15,
        "color_vibrancy": 0.10,
        "scene_change": 0.10,
        "sky_ratio": 0.05,
        "golden_hour": 0.10,
        "scenic_score": 0.20,
        "weather": 0.07,
        "brightness": 0.07,
    }


class TestRedistributeFreeFunction:
    """Pure-function redistribute_gps_weights contract."""

    def test_redistribute_preserves_sum(self):
        # No-GPS path: redistributed weights must still sum to exactly 1.0.
        new = redistribute_gps_weights(_default_weights(), has_gps=False)
        assert sum(new.values()) == pytest.approx(1.0, abs=1e-9)

    def test_redistribute_gps_signals_zeroed(self):
        # speed_variance and scenic_score are removed from the composite.
        new = redistribute_gps_weights(_default_weights(), has_gps=False)
        assert new["speed_variance"] == 0.0
        assert new["scenic_score"] == 0.0

    def test_redistribute_visual_weights_grow(self):
        # The 0.35 mass freed by GPS zeroing is fully absorbed by the
        # remaining (non-GPS) keys — their total grows by exactly 0.35.
        original = _default_weights()
        new = redistribute_gps_weights(original, has_gps=False)
        original_visual = sum(w for name, w in original.items() if name not in GPS_SIGNAL_NAMES)
        new_visual = sum(w for name, w in new.items() if name not in GPS_SIGNAL_NAMES)
        assert (new_visual - original_visual) == pytest.approx(0.35, abs=1e-9)
        # And the visual total is now 1.0 (since GPS keys are 0.0).
        assert new_visual == pytest.approx(1.0, abs=1e-9)

    def test_redistribute_noop_with_gps(self):
        # has_gps=True returns the same mapping (no zeroing, no scaling).
        original = _default_weights()
        new = redistribute_gps_weights(original, has_gps=True)
        assert new == original
        # And it's a copy — mutating the result doesn't affect the input.
        new["motion_energy"] = 999.0
        assert original["motion_energy"] == 0.16

    def test_redistribute_proportional_scaling(self):
        # Each visual key grows by factor 1/visual_weight = 1/0.65.
        original = _default_weights()
        new = redistribute_gps_weights(original, has_gps=False)
        visual_total = sum(w for name, w in original.items() if name not in GPS_SIGNAL_NAMES)
        factor = 1.0 / visual_total
        for name, w in original.items():
            if name in GPS_SIGNAL_NAMES:
                continue
            assert new[name] == pytest.approx(w * factor, abs=1e-9)


class TestMethodWrapperDelegates:
    """The instance-method wrapper preserves the historical no-arg call."""

    def test_method_no_args_redistributes(self):
        # Legacy call site: scorer._redistribute_gps_weights() with no
        # args. Default has_gps=False keeps the existing redistribution
        # behavior intact.
        scorer = UnifiedSegmentScorer(_default_weights())
        new = scorer._redistribute_gps_weights()
        assert new["speed_variance"] == 0.0
        assert new["scenic_score"] == 0.0
        assert sum(new.values()) == pytest.approx(1.0, abs=1e-9)

    def test_method_has_gps_true_is_noop(self):
        scorer = UnifiedSegmentScorer(_default_weights())
        new = scorer._redistribute_gps_weights(has_gps=True)
        assert new == _default_weights()


class TestScoreFileNoGpsEndToEnd:
    """End-to-end: score_file with gps_data=None produces non-zero composite."""

    def test_score_file_without_gps_returns_nonzero_composite(self):
        scorer = UnifiedSegmentScorer(_default_weights())
        # Synthetic non-trivial frames so visual signals have something
        # to bite on. Uniform mid-grey to keep brightness ≈ 0.5.
        h, w = 8, 8
        frames = [
            (0.0, np.full((h, w, 3), 100, dtype=np.uint8)),
            (1.0, np.full((h, w, 3), 150, dtype=np.uint8)),
            (2.0, np.full((h, w, 3), 200, dtype=np.uint8)),
        ]
        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            result = scorer.score_file(Path("dummy.mp4"), gps_data=None)

        assert result.shape == (3, NUM_COLUMNS)
        # Composite must be > 0 — the brightness + weather signals alone
        # guarantee that under the redistributed weights.
        assert (result[:, COL_COMPOSITE] > 0).all()
        # Composite must be in [0, 1] — clip contract from combine_signals.
        assert (result[:, COL_COMPOSITE] <= 1.0).all()


class TestGpsLessChaptersAndRouteContext:
    """Regression: GPS-less callers in chapters / route_context don't raise."""

    def test_extract_route_context_with_empty_chapters(self):
        # route_context.extract_route_context takes a pre-resolved location
        # string + chapters list. GPS-less drives produce an empty chapters
        # list — the helper must degrade to a default title, not raise.
        from gomoveshift.pipeline.route_context import extract_route_context

        ctx = extract_route_context("Killarney, Kerry, Ireland", chapters=[])
        # We don't pin a specific title shape — just assert non-raise and
        # a usable RouteContext-like object.
        assert ctx is not None
        title = ctx.make_default_title()
        assert isinstance(title, str)
        assert title  # non-empty
