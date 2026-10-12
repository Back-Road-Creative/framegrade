"""The deep product ships the largest rectangle its own corners prove clean (auto scope).

An earlier measurement found the full-frame outer field failing A3 at every useful depth (+55.2% unpaired /
+60.7% paired against a 10% boundary), while centre+middle pass with margin (+2.9% / +0.6% at
n=22).  An earlier fix always cropped to the region proved clean that one time.  Auto scope instead
measures every combine's own corners (`resolve_auto_scope`, scope="auto", the default) and ships
the full sensor whenever they pass -- falling back to that same central-two-thirds rectangle only
when they don't, or can't be proven to.  These tests pin the fallback rectangle's own geometry,
that "auto" makes the right call in both directions, and that opting into a fixed scope is
explicit.
"""

from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_combine


def test_scoped_slice_is_central_two_thirds():
    rows, cols = astro_combine.scoped_frame_slice((3728, 5600))
    assert rows.stop - rows.start == 2484
    assert cols.stop - cols.start == 3732
    # centred: equal margin either side (within a pixel of rounding)
    assert abs(rows.start - (3728 - (rows.stop - rows.start) - rows.start)) <= 1
    assert abs(cols.start - (5600 - (cols.stop - cols.start) - cols.start)) <= 1


def test_every_scoped_pixel_is_inside_the_passing_radius():
    """The crop must not include any pixel outside r <= (2/3)*rmax -- the zone that passes A3."""
    h, w = 3728, 5600
    rows, cols = astro_combine.scoped_frame_slice((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rmax = np.hypot(cx, cy)
    corners = [
        (rows.start, cols.start),
        (rows.start, cols.stop - 1),
        (rows.stop - 1, cols.start),
        (rows.stop - 1, cols.stop - 1),
    ]
    for y, x in corners:
        assert np.hypot(x - cx, y - cy) / rmax <= 2.0 / 3.0 + 1e-9


def test_scoped_is_a_valid_scope_and_garbage_is_rejected():
    with pytest.raises(ValueError):
        astro_combine.scoped_frame_slice((10, 10), scope="not-a-scope")


def test_full_scope_returns_the_whole_frame():
    rows, cols = astro_combine.scoped_frame_slice((3728, 5600), scope="full")
    assert (rows.start, rows.stop, cols.start, cols.stop) == (0, 3728, 0, 5600)


def test_scoped_frame_slice_does_not_understand_auto():
    """ "auto" is resolved to a concrete scope by resolve_auto_scope() before this function
    ever sees it -- it is not a third crop geometry."""
    with pytest.raises(ValueError):
        astro_combine.scoped_frame_slice((10, 10), scope="auto")


# ---------------------------------------------------------------------------
# --scope auto measures A3-by-radial-third on the blended FULL frame (before any crop)
# and ships the largest passing rectangle -- full sensor if it passes, else the scoped
# fallback that is already proved to pass.
# ---------------------------------------------------------------------------


def _ring(center, radius, count, angle0=0.0):
    angles = np.linspace(angle0, angle0 + 2 * np.pi, count, endpoint=False)
    return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]


class TestResolveAutoScope:
    """Same synthetic-star discipline as test_astro_quality.py's TestA3ByRadialThird -- a
    300x300 frame so a max-grown adaptive window never clips at a placed star."""

    SHAPE = (300, 300)
    CENTER_PX = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)
    MAX_RADIUS_PX = float(np.hypot(*CENTER_PX))
    NARROW_SIGMA_PX = 1.8
    BROAD_SIGMA_PX = 1.8 * 1.55
    _GAUSSIAN_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

    CENTER_STARS = _ring(CENTER_PX, 0.15 * MAX_RADIUS_PX, 4)
    MIDDLE_STARS = _ring(CENTER_PX, 0.50 * MAX_RADIUS_PX, 4)
    OUTER_STARS = _ring(CENTER_PX, 0.70 * MAX_RADIUS_PX, 4, angle0=np.pi / 4)

    def _rgb(self, positions_sigmas, noise=0.002, seed=0, flux=0.6, background=0.05):
        h, w = self.SHAPE
        luma = np.clip(
            np.random.default_rng(seed).normal(background, noise, size=self.SHAPE), 0.0, None
        )
        yy, xx = (a.astype(np.float32) for a in np.mgrid[0:h, 0:w])
        for cx, cy, sigma_px in positions_sigmas:
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            luma += flux * np.exp(-r2 / (2.0 * sigma_px**2))
        luma = np.clip(luma, 0.0, 1.0).astype(np.float32)
        return np.repeat(luma[..., None], 3, axis=2)

    def test_ships_full_frame_when_the_outer_third_measurably_passes(self):
        positions = [
            (x, y, self.NARROW_SIGMA_PX)
            for x, y in self.CENTER_STARS + self.MIDDLE_STARS + self.OUTER_STARS
        ]
        rgb = self._rgb(positions, seed=21)
        frame_median_fwhm = self.NARROW_SIGMA_PX * self._GAUSSIAN_SIGMA_TO_FWHM

        scope, decision = astro_combine.resolve_auto_scope(rgb, frame_median_fwhm)

        assert scope == "full"
        assert decision["resolved_scope"] == "full"
        assert decision["outer_passed"] is True

    def test_falls_back_to_scoped_when_the_outer_third_fails(self):
        positions = [
            (x, y, self.NARROW_SIGMA_PX) for x, y in self.CENTER_STARS + self.MIDDLE_STARS
        ] + [(x, y, self.BROAD_SIGMA_PX) for x, y in self.OUTER_STARS]
        rgb = self._rgb(positions, seed=22)
        frame_median_fwhm = self.NARROW_SIGMA_PX * self._GAUSSIAN_SIGMA_TO_FWHM

        scope, decision = astro_combine.resolve_auto_scope(rgb, frame_median_fwhm)

        assert scope == "scoped"
        assert decision["resolved_scope"] == "scoped"
        assert decision["outer_passed"] is False

    def test_falls_back_to_scoped_when_no_reference_fwhm_is_available(self):
        """A pass can never be proven without a frame-median FWHM to grow against --
        absence must not read as success."""
        positions = [
            (x, y, self.NARROW_SIGMA_PX)
            for x, y in self.CENTER_STARS + self.MIDDLE_STARS + self.OUTER_STARS
        ]
        rgb = self._rgb(positions, seed=23)

        scope, decision = astro_combine.resolve_auto_scope(rgb, None)

        assert scope == "scoped"
        assert decision["outer_passed"] is None


class TestFrameMedianFwhmPx:
    def _artifact(self, fwhm_values):
        manifest = {
            "registration_provenance": [
                {"frame": f"F{i}.NEF", "triage": {"fwhm_px": value}}
                for i, value in enumerate(fwhm_values)
            ]
        }
        return astro_combine.ValidatedArtifact(Path("chunk.linear"), manifest)

    def test_pools_triage_fwhm_across_every_artifact(self):
        artifacts = [self._artifact([2.0, 4.0]), self._artifact([6.0])]

        assert astro_combine._frame_median_fwhm_px(artifacts) == pytest.approx(4.0)

    def test_returns_none_when_no_artifact_carries_triage_provenance(self):
        manifest = {"registration_provenance": [{"frame": "F0.NEF"}]}
        artifacts = [astro_combine.ValidatedArtifact(Path("chunk.linear"), manifest)]

        assert astro_combine._frame_median_fwhm_px(artifacts) is None

    def test_returns_none_for_an_artifact_with_no_registration_provenance_key(self):
        artifacts = [astro_combine.ValidatedArtifact(Path("chunk.linear"), {})]

        assert astro_combine._frame_median_fwhm_px(artifacts) is None
