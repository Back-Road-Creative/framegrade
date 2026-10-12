"""Per-LF net travel bearing (8-point compass → ``🧭 Heading:`` line).

Wires the existing ``_drive_helpers._compass_from_waypoints`` (no new
haversine-bearing code). Covers the compute (``_compute_per_lf_bearing`` —
first→last waypoint, loop-drive suppression), the ``🧭 Heading:`` formatter,
the int-keyed ``per_lf_bearing`` state field, and the ``generate()`` splice.
"""

from unittest.mock import MagicMock

from gomoveshift.pipeline.drive_state_model import _INT_KEY_DICT_FIELDS, PipelineState
from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _format_bearing_line,
)
from gomoveshift.pipeline.post_encoding_invariants import _compute_per_lf_bearing


def _journey(points):
    """points: list of (lat, lon) → mock Journey."""
    j = MagicMock()
    wps = []
    for lat, lon in points:
        wp = MagicMock()
        wp.latitude, wp.longitude = lat, lon
        wps.append(wp)
    j.waypoints = wps
    return j


def _compute_one(points, key=0):
    return _compute_per_lf_bearing(
        {key: "/a.gpx"}, parse_fn=MagicMock(return_value=_journey(points))
    )


class TestComputePerLfBearing:
    def test_empty_returns_empty(self):
        assert _compute_per_lf_bearing({}, parse_fn=MagicMock()) == {}

    def test_northbound(self):
        # ~0.5 deg north (~55 km), no east/west → "N"
        assert _compute_one([(40.0, -75.0), (40.5, -75.0)])[0] == "N"

    def test_eastbound(self):
        # ~0.5 deg east at this latitude (~42 km), no north/south → "E"
        assert _compute_one([(40.0, -75.0), (40.0, -74.5)])[0] == "E"

    def test_loop_drive_suppressed(self):
        # start ≈ end (net << 2 km) → omitted, no meaningful heading
        assert _compute_one([(40.0, -75.0), (40.2, -75.2), (40.0, -75.0)]) == {}

    def test_single_waypoint_omitted(self):
        assert _compute_one([(40.0, -75.0)]) == {}

    def test_parse_failure_and_str_key(self):
        def boom(_):
            raise OSError("x")

        assert _compute_per_lf_bearing({0: "/a.gpx"}, parse_fn=boom) == {}
        assert next(iter(_compute_one([(40.0, -75.0), (40.5, -75.0)], key="0"))) == 0


class TestFormatBearingLine:
    def test_cardinal_to_word(self):
        assert _format_bearing_line("N", "X") == "🧭 Heading: Northbound\n"
        assert _format_bearing_line("SE", "X") == "🧭 Heading: Southeastbound\n"
        assert _format_bearing_line("NW", "X") == "🧭 Heading: Northwestbound\n"

    def test_none_or_unmapped_returns_empty(self):
        assert _format_bearing_line(None, "X") == ""
        assert _format_bearing_line("NNE", "X") == ""


class TestStateRoundTrip:
    def test_registered_and_default_none(self):
        assert "per_lf_bearing" in _INT_KEY_DICT_FIELDS
        assert PipelineState().per_lf_bearing is None

    def test_int_keys_survive_roundtrip(self):
        state = PipelineState()
        state.per_lf_bearing = {0: "NE", 1: "S"}
        restored = PipelineState.from_dict(state.to_dict())
        assert restored.per_lf_bearing == {0: "NE", 1: "S"}
        assert all(isinstance(k, int) for k in restored.per_lf_bearing)


class TestGenerateIntegration:
    def test_heading_line_present_then_absent(self):
        gen = LongFormMetadataGenerator()
        with_b = gen.generate(location="Kerry, Ireland", video_duration=1800.0, bearing="NE")
        assert "🧭 Heading: Northeastbound" in with_b["description"]
        without = gen.generate(location="Kerry, Ireland", video_duration=1800.0)
        assert "🧭 Heading:" not in without["description"]
