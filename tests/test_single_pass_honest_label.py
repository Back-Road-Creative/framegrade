"""Honest method-label contract for the single-pass color step.

single_pass must not log/cache method="auto_adaptive" unless the per-span LUT
filtergraph actually reaches the encode. When the graph collapses to an
averaged grade (identical spans, missing span metadata, or empty corrections),
the label must say so instead of claiming per-segment correction.
"""

import shutil
from unittest.mock import patch

from gomoveshift.config import Config
from gomoveshift.pipeline import single_pass as sp
from gomoveshift.pipeline.color_correction import CorrectionParams


def _params(shadow_lift: float = 5.0, **overrides) -> CorrectionParams:
    base = dict(
        shadow_lift=shadow_lift,
        highlight_pull=5.0,
        saturation_mult=1.05,
        contrast_curve=0.0,
        white_balance_r=0.0,
        white_balance_b=0.0,
        gamma=1.02,
        denoise_strength=0.0,
        sharpen_amount=0.0,
    )
    base.update(overrides)
    return CorrectionParams(**base)


def _processor() -> sp.SinglePassProcessor:
    with patch.object(sp, "detect_nvenc", return_value=True):
        return sp.SinglePassProcessor(Config())


def _cleanup(lut_dir) -> None:
    """The caller owns the returned temp dir; `process` does this for real."""
    if lut_dir is not None:
        shutil.rmtree(lut_dir, ignore_errors=True)


class TestHonestMethodLabel:
    def test_multi_span_bakes_luts_and_labels_adaptive(self) -> None:
        proc = _processor()
        p1, p2 = _params(shadow_lift=2.0), _params(shadow_lift=12.0)
        filter_str, method, lut_dir = proc._corrections_to_filter([p1, p2], [90.0, 60.0])
        try:
            assert method == "auto_adaptive"
            assert "xfade=" in filter_str
            assert lut_dir is not None
            assert len(list(lut_dir.glob("*.cube"))) == 2
        finally:
            _cleanup(lut_dir)

    def test_identity_collapse_labels_global_and_bakes_one_lut(self) -> None:
        proc = _processor()
        p = _params(shadow_lift=8.0)
        filter_str, method, lut_dir = proc._corrections_to_filter([p, p], [90.0, 60.0])
        try:
            assert method == "auto_global"
            assert "xfade=" not in filter_str
            # The collapsed grade still bakes — exactly ONE span LUT, and
            # the per-span dir the adaptive attempt made is not left behind.
            assert lut_dir is not None
            assert len(list(lut_dir.glob("*.cube"))) == 1
        finally:
            _cleanup(lut_dir)

    def test_missing_span_durations_labels_global(self) -> None:
        """Corrections without span metadata collapse to the averaged grade —
        the label must say so instead of claiming per-segment correction."""
        proc = _processor()
        p1, p2 = _params(shadow_lift=2.0), _params(shadow_lift=12.0)
        filter_str, method, lut_dir = proc._corrections_to_filter([p1, p2], None)
        try:
            assert method == "auto_global"
            assert "xfade=" not in filter_str
            assert lut_dir is not None
            assert len(list(lut_dir.glob("*.cube"))) == 1
        finally:
            _cleanup(lut_dir)

    def test_empty_corrections_label_none(self) -> None:
        proc = _processor()
        assert proc._corrections_to_filter([], None) == ("null", "none", None)
