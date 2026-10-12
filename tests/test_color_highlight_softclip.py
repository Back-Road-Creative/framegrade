"""Tests for the highlight knee.

The knee fires on ``highlight_pull >= 5.0`` (the same threshold the adaptive
pipeline uses to flag sun-clip / blown highlights). It replaced the old
``colorlevels=romax=ceiling`` soft-clip, which was a whole-range linear gain
(``out = in * ceiling``) that darkened shadows and midtones 2-4% on every
bright-sky clip. The ceiling now scales ONLY the curve's top (1.0) control
point; the 0.75 knee anchor keeps the un-squeezed ``wh_out``, so the curve is
identity through the midtones and only the [0.75, 1.0] highlight segment bends.
"""

import numpy as np

from gomoveshift.pipeline.color_correction import CorrectionParams, _apply_curves
from gomoveshift.pipeline.color_filters import _build_single_filter


def _curves_points(chain: str) -> list[str]:
    """Extract the ``x/y`` control points from the curves filter in a chain."""
    for segment in chain.split(","):
        if segment.startswith("curves=m="):
            return segment.split("'")[1].split(" ")
    raise AssertionError(f"no curves filter in chain: {chain}")


class TestHighlightKnee:
    def test_no_colorlevels_filter_anywhere(self):
        """The whole-range colorlevels soft-clip is gone — the knee lives in
        the curves points now."""
        for pull in (4.5, 5.0, 10.0, 15.0):
            f = _build_single_filter(CorrectionParams(shadow_lift=10, highlight_pull=pull))
            assert "colorlevels" not in f, f"colorlevels leaked at pull={pull}: {f}"

    def test_top_point_squeezed_at_threshold(self):
        """highlight_pull == 5.0 -> top point = wh_out * 0.96 = 0.9412."""
        f = _build_single_filter(CorrectionParams(shadow_lift=10, highlight_pull=5.0))
        pts = _curves_points(f)
        assert pts[-1] == "1.0000/0.9412", f"top point wrong: {pts}"

    def test_top_point_squeeze_scales_with_pull(self):
        """highlight_pull=15 -> ceiling 0.98 -> top point 0.9412 * ... = 0.9224."""
        f = _build_single_filter(CorrectionParams(shadow_lift=10, highlight_pull=15.0))
        pts = _curves_points(f)
        assert pts[-1] == "1.0000/0.9224", f"top point wrong: {pts}"

    def test_knee_anchor_not_squeezed(self):
        """The 0.75 knee anchor keeps the un-squeezed wh_out — proving midtones
        are preserved. If the ceiling wrongly multiplied the whole curve the
        anchor would be 0.7206 * 0.98 = 0.7062, not 0.7206."""
        f = _build_single_filter(CorrectionParams(shadow_lift=10, highlight_pull=15.0))
        pts = _curves_points(f)
        # 0.75 - (1 - (1 - 15/255)) * 0.5 = 0.75 - 0.0294 = 0.7206
        assert pts[2] == "0.7500/0.7206", f"knee anchor was squeezed: {pts}"

    def test_no_squeeze_below_threshold(self):
        """highlight_pull < 5 -> top point is the bare highlight pull, no
        ceiling: 1 - 4.5/255 = 0.9824."""
        f = _build_single_filter(CorrectionParams(shadow_lift=10, highlight_pull=4.5))
        pts = _curves_points(f)
        assert pts[-1] == "1.0000/0.9824", f"unexpected squeeze below threshold: {pts}"

    def test_midtones_not_darkened_numpy_mirror(self):
        """Regression: a mid-gray input must NOT be darkened by
        the highlight knee. The old whole-range colorlevels multiply pulled 0.5
        down to ~0.48 (× 0.96); the knee leaves it at the curve's midtone value
        (>= 0.49). Uses the numpy mirror that bakes the GPU LUT."""
        # Only highlight_pull active (no shadow/contrast) so the sample isolates
        # the knee's effect on the midtone.
        params = CorrectionParams(shadow_lift=0.0, highlight_pull=15.0, contrast_curve=0.0)
        mid = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        out = _apply_curves(mid, params)[0, 0]
        assert out >= 0.49, f"mid-gray darkened by the knee to {out:.4f} (< 0.49)"

    def test_curves_before_colorbalance_before_eq(self):
        """Filter chain order unchanged: curves -> colorbalance -> eq."""
        f = _build_single_filter(
            CorrectionParams(
                shadow_lift=10,
                highlight_pull=10.0,
                white_balance_r=0.02,
                saturation_mult=1.1,
            )
        )
        curves_pos = f.find("curves=")
        colorbalance_pos = f.find("colorbalance=")
        eq_pos = f.find("eq=")
        assert curves_pos >= 0
        assert colorbalance_pos > curves_pos, "colorbalance must come after curves"
        assert eq_pos > colorbalance_pos, "eq must come last"
