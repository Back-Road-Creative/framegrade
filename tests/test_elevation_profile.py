"""Per-LF elevation PROFILE (range + net climb/descent).

Covers the compute (``_compute_per_lf_elevation_profile``) and the int-keyed
``per_lf_elevation_profile`` state field. The ``📈 Elevation:`` description
line was dropped — low SEO value, GPS-noise-inflated on flat
drives — so its formatter and render splice no longer exist; the compute +
state machinery survive (route-stats sidecar + invariants still consume them).
"""

from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_state_model import _INT_KEY_DICT_FIELDS, PipelineState
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.post_encoding_invariants import _compute_per_lf_elevation_profile


def _journey(elevations):
    j = MagicMock()
    wps = []
    for elev in elevations:
        wp = MagicMock()
        wp.latitude, wp.longitude, wp.elevation = 40.0, -75.0, elev
        wps.append(wp)
    j.waypoints = wps
    return j


def _compute_one(elevations, key=0):
    return _compute_per_lf_elevation_profile(
        {key: "/a.gpx"}, parse_fn=MagicMock(return_value=_journey(elevations))
    )


def _prof(min_m, max_m, net_m=None, descent_m=0.0):
    return {
        "min_m": min_m,
        "max_m": max_m,
        "net_m": (max_m - min_m) if net_m is None else net_m,
        "descent_m": descent_m,
    }


class TestComputePerLfElevationProfile:
    def test_empty_returns_empty(self):
        assert _compute_per_lf_elevation_profile({}, parse_fn=MagicMock()) == {}

    def test_climb(self):
        p = _compute_one([100.0, 250.0, 500.0])[0]
        assert p == pytest.approx({"min_m": 100, "max_m": 500, "net_m": 400, "descent_m": 0})

    def test_descent_net_negative(self):
        p = _compute_one([500.0, 300.0, 100.0])[0]
        assert (p["net_m"], p["descent_m"]) == (pytest.approx(-400.0), pytest.approx(400.0))

    def test_mixed_range_and_descent(self):
        # 100→400(+300)→150(−250)→300(+150); net = 300−100 = 200.
        p = _compute_one([100.0, 400.0, 150.0, 300.0])[0]
        assert p == pytest.approx({"min_m": 100, "max_m": 400, "net_m": 200, "descent_m": 250})

    def test_none_elevations_skipped(self):
        p = _compute_one([None, 200.0, None, 350.0])[0]
        assert (p["min_m"], p["max_m"], p["net_m"]) == (200.0, 350.0, 150.0)

    def test_all_none_or_no_waypoints_omits_lf(self):
        assert _compute_one([None, None]) == {}
        empty = MagicMock()
        empty.waypoints = []
        assert (
            _compute_per_lf_elevation_profile({0: "/a.gpx"}, parse_fn=MagicMock(return_value=empty))
            == {}
        )

    def test_parse_failure_omits_lf(self):
        def boom(_):
            raise OSError("parse fail")

        assert _compute_per_lf_elevation_profile({0: "/a.gpx"}, parse_fn=boom) == {}
        # str keys (resume) coerce to int
        assert next(iter(_compute_one([100.0, 300.0], key="0"))) == 0


class TestStateRoundTrip:
    def test_field_registered_and_defaults_none(self):
        assert "per_lf_elevation_profile" in _INT_KEY_DICT_FIELDS
        assert PipelineState().per_lf_elevation_profile is None

    def test_int_keys_and_values_survive_roundtrip(self):
        state = PipelineState()
        state.per_lf_elevation_profile = {0: _prof(100.0, 500.0, 400.0, 12.0), 1: _prof(0.0, 50.0)}
        restored = PipelineState.from_dict(state.to_dict())
        assert set(restored.per_lf_elevation_profile) == {0, 1}
        assert all(isinstance(k, int) for k in restored.per_lf_elevation_profile)
        assert restored.per_lf_elevation_profile[0]["net_m"] == pytest.approx(400.0)


class TestGenerateIntegration:
    def test_elevation_lines_never_rendered(self):
        """Neither the scalar ⛰️ gain line nor the 📈 profile
        line is rendered, even when elevation data is supplied to generate()."""
        gen = LongFormMetadataGenerator()
        result = gen.generate(
            location="Asheville, NC, USA",
            video_duration=1800.0,
            elevation_gain_m=400.0,
            elevation_profile=_prof(100.0, 500.0, 400.0),
        )
        assert "📈 Elevation:" not in result["description"]
        assert "⛰️ Elevation gain:" not in result["description"]
