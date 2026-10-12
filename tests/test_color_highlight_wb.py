"""Highlight white-balance term — colour casts concentrated in the bright end.

``colorbalance`` previously set only the shadow (rs/bs) and
midtone (rm/bm) shifts, so a cast living only in the highlights (a blue sky, a
warm blown window) got zero correction by construction. This proves the new
highlight-region statistic, its highlight (rh/bh) white-balance term, and that
the term is:
  * measured, not invented — it comes from a per-channel mean over just the
    highlight pixels, not from the whole-frame means;
  * self-limiting — weighted by the highlight area share, so weak evidence
    earns little correction;
  * no-false-positive — a neutral frame and a shadows-only cast leave it at 0;
  * shared-authority — clamped to the same ±0.05 the shadow/midtone shift uses;
  * applied on both the CPU filter string and the baked LUT (numpy mirror).
"""

import numpy as np

from gomoveshift.pipeline.color import (
    CorrectionParams,
    SegmentStats,
    _apply_colorbalance,
    analyze_frame,
    build_filter_chain,
    compute_correction,
)


def _split_frame(top_rgb, bottom_rgb, top_frac=0.5, size=(90, 160)):
    """A frame whose top ``top_frac`` rows are one colour, the rest another."""
    h, w = size
    frame = np.empty((h, w, 3), dtype=np.uint8)
    split = int(h * top_frac)
    frame[:split] = top_rgb
    frame[split:] = bottom_rgb
    return frame


def _seg(**kw):
    base = dict(
        black_point=20,
        white_point=230,
        mid_tone=128,
        saturation=0.3,
        r_mean=0.5,
        g_mean=0.5,
        b_mean=0.5,
        r_mean_high=0.9,
        g_mean_high=0.9,
        b_mean_high=0.9,
        highlight_frac=0.4,
    )
    base.update(kw)
    return SegmentStats(**base)


class TestHighlightStatistic:
    def test_cast_measured_over_bright_region_not_whole_frame(self):
        # Bright bluish highlights on top, neutral midtone below.
        frame = _split_frame(top_rgb=(200, 210, 240), bottom_rgb=(110, 110, 110))
        stats = analyze_frame(frame)
        assert stats.highlight_frac > 0.0
        # The blue cast is visible in the highlight region ...
        high_cast = stats.b_mean_high - stats.g_mean_high
        assert high_cast > 0.0
        # ... but the neutral lower half dilutes it in the whole-frame means.
        whole_cast = stats.b_mean - stats.g_mean
        assert high_cast > whole_cast

    def test_neutral_frame_has_no_highlight_cast(self):
        frame = _split_frame(top_rgb=(230, 230, 230), bottom_rgb=(90, 90, 90))
        stats = analyze_frame(frame)
        assert abs(stats.r_mean_high - stats.g_mean_high) < 1e-3
        assert abs(stats.b_mean_high - stats.g_mean_high) < 1e-3


class TestHighlightWhiteBalance:
    def test_blue_highlight_cast_pulls_blue_down(self):
        params = compute_correction(_seg(r_mean_high=0.90, g_mean_high=0.90, b_mean_high=0.98))
        assert params.white_balance_b_high < 0.0
        assert params.white_balance_b_high >= -0.05  # shares the ±0.05 authority

    def test_neutral_highlights_get_no_term(self):
        params = compute_correction(_seg(r_mean_high=0.9, g_mean_high=0.9, b_mean_high=0.9))
        assert params.white_balance_r_high == 0.0
        assert params.white_balance_b_high == 0.0

    def test_shadow_only_cast_leaves_highlight_term_zero(self):
        # Warm cast in the whole-frame means, neutral highlights.
        params = compute_correction(
            _seg(
                r_mean=0.6,
                g_mean=0.5,
                b_mean=0.5,
                r_mean_high=0.9,
                g_mean_high=0.9,
                b_mean_high=0.9,
            )
        )
        assert params.white_balance_r_high == 0.0
        assert params.white_balance_r < 0.0  # whole-frame term still fires

    def test_self_limiting_in_highlight_fraction(self):
        cast = dict(r_mean_high=0.90, g_mean_high=0.90, b_mean_high=0.99)
        big = compute_correction(_seg(highlight_frac=0.5, **cast))
        small = compute_correction(_seg(highlight_frac=0.05, **cast))
        assert abs(small.white_balance_b_high) < abs(big.white_balance_b_high)


class TestHighlightFilterAndLutParity:
    def test_colorbalance_emits_highlight_terms(self):
        chain = build_filter_chain([CorrectionParams(white_balance_b_high=-0.03)])
        assert "colorbalance=" in chain
        assert "bh=" in chain

    def test_highlight_only_wb_still_builds_colorbalance(self):
        chain = build_filter_chain([CorrectionParams(white_balance_r_high=0.04)])
        assert "colorbalance=" in chain

    def test_highlight_term_acts_on_bright_end_only(self):
        # Blue highlight pull-down: a bright pixel drops blue, a dark pixel does not.
        params = CorrectionParams(white_balance_b_high=-0.04)
        pts = np.array([[0.9, 0.9, 0.98], [0.1, 0.1, 0.18]], dtype=np.float64)
        out = _apply_colorbalance(pts, params)
        assert out[0, 2] < pts[0, 2]  # highlight blue pulled down
        assert abs(out[1, 2] - pts[1, 2]) < 1e-6  # shadow blue untouched

    def test_cpu_lut_parity_highlight_wb(self, tmp_path):
        from gomoveshift.pipeline.color import params_to_lut3d_cube

        params = CorrectionParams(white_balance_r_high=0.04, white_balance_b_high=-0.05)
        cube = 65
        lut_path = tmp_path / "hl.cube"
        params_to_lut3d_cube(params, cube_size=cube, output_path=lut_path)

        data = [
            ln.split()
            for ln in lut_path.read_text().splitlines()
            if ln.strip()
            and not ln.startswith(("TITLE", "LUT_3D_SIZE", "DOMAIN_MIN", "DOMAIN_MAX"))
        ]
        lut = np.array(data, dtype=np.float64)

        pts = np.array(
            [[1.0, 1.0, 1.0], [0.9375, 0.9375, 0.9375], [0.5, 0.5, 0.5]], dtype=np.float64
        )
        cpu = np.clip(_apply_colorbalance(pts, params), 0.0, 1.0)
        for i, (r, g, b) in enumerate(pts):
            ri = int(round(r * (cube - 1)))
            gi = int(round(g * (cube - 1)))
            bi = int(round(b * (cube - 1)))
            flat = bi * cube * cube + gi * cube + ri
            assert np.max(np.abs(cpu[i] - lut[flat])) < 2.0 / 255.0
