"""TDD tests for the generator side of the full-drive stats gate.

The description redundancy gate ``check_description_redundancy`` hard-fails a
Distance/Pace stat line that reflects the full (pre-cut) drive unless it
carries an explicit ``Full drive:`` scope prefix, keyed on
``metadata_json["full_drive_stats"]``. Nothing populated that key yet. This
locks in the generator's half: when a caller tells ``generate()`` /
``_build_gms_description()`` which stat(s) are full-drive-derived, the
description renders the ``Full drive:`` prefix and the returned metadata dict
carries ``full_drive_stats`` — so the gate passes by construction. Absent
that flag (the kept-segment default), output is byte-identical to
today's.
"""

from __future__ import annotations

from gomoveshift.pipeline.longform_metadata import (
    LongFormMetadataGenerator,
    _format_distance_line,
    _format_speed_line,
)
from gomoveshift.pipeline.validation.content_gates import check_description_redundancy

LOCATION = "Dingle Peninsula, County Kerry, Ireland"
LOCATION_US = "Blue Ridge Parkway, Virginia, USA"


class _StubGenerator(LongFormMetadataGenerator):
    """Same SEO-disabled stub the title and tag tests use —
    exercises ``_generate_basic``, the deterministic path."""

    def _init_seo_manager(self):  # pragma: no cover - simple override
        self._manager = None
        self._optimizer = None
        self._seo_degraded = True

    def _init_ledger(self):  # pragma: no cover - skip ledger IO in tests
        self._ledger = None
        self._ledger_path = None


class TestFormatDistanceLineFullDrive:
    def test_full_drive_prefix_non_us(self):
        line = _format_distance_line(47.0, LOCATION, full_drive=True)
        assert line == "🛣️ Full drive: 47 km (29 mi)\n"

    def test_full_drive_prefix_us(self):
        line = _format_distance_line(47.0, LOCATION_US, full_drive=True)
        assert line == "🛣️ Full drive: 29 mi\n"

    def test_default_unflagged_kept_segment_unchanged(self):
        assert _format_distance_line(47.0, LOCATION) == "🛣️ Distance: 47 km (29 mi)\n"
        assert _format_distance_line(47.0, LOCATION, full_drive=False) == (
            "🛣️ Distance: 47 km (29 mi)\n"
        )


class TestFormatSpeedLineFullDrive:
    def test_full_drive_prefix_non_us(self):
        line = _format_speed_line({"avg_kmh": 78.0, "std_kmh": 30.0}, LOCATION, full_drive=True)
        assert line == "🚗 Full drive: 78 km/h avg · winding roads\n"

    def test_default_unflagged_kept_segment_unchanged(self):
        speed = {"avg_kmh": 78.0, "std_kmh": 30.0}
        assert _format_speed_line(speed, LOCATION) == "🚗 Pace: 78 km/h avg · winding roads\n"


class TestBuildGmsDescriptionFullDriveStats:
    def _gen(self):
        return LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)

    def test_distance_line_gets_prefix_when_flagged(self):
        description = self._gen()._build_gms_description(
            location=LOCATION,
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            distance_km=47.0,
            full_drive_stats=["distance"],
        )
        assert "🛣️ Full drive: 47 km (29 mi)" in description

    def test_speed_line_never_rendered_even_when_flagged(self):
        """The 🚗 Pace/avg-speed line was dropped from the description
        entirely — flagging ``speed`` in ``full_drive_stats`` no
        longer has a line to prefix."""
        description = self._gen()._build_gms_description(
            location=LOCATION,
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            speed={"avg_kmh": 78.0, "std_kmh": 30.0},
            full_drive_stats=["speed"],
        )
        assert "🚗" not in description
        assert "avg" not in description

    def test_no_full_drive_stats_is_byte_identical_to_default(self):
        kwargs = dict(
            location=LOCATION,
            seo_description="POV drive through Dingle.",
            duration_str="32 min",
            publish_day="Wednesday",
            distance_km=47.0,
            speed={"avg_kmh": 78.0, "std_kmh": 30.0},
        )
        assert self._gen()._build_gms_description(**kwargs) == self._gen()._build_gms_description(
            full_drive_stats=None, **kwargs
        )
        description = self._gen()._build_gms_description(**kwargs)
        assert "Full drive:" not in description
        assert "🛣️ Distance: 47 km (29 mi)" in description
        assert "🚗 Pace:" not in description


class TestGenerateStampsFullDriveStatsAndPassesGate:
    def test_full_drive_distance_is_flagged_prefixed_and_passes_gate(self, tmp_path):
        gen = _StubGenerator(seo_storage_path=tmp_path)
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date="2025-09-14",
            distance_km=22.0,
            full_drive_stats=["distance"],
        )
        assert result["full_drive_stats"] == ["distance"]
        assert "🛣️ Full drive: 22 km (14 mi)" in result["description"]
        gate_result = check_description_redundancy(result)
        assert gate_result.passed, gate_result.message

    def test_kept_segment_distance_has_no_flag_and_passes_gate(self, tmp_path):
        gen = _StubGenerator(seo_storage_path=tmp_path)
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date="2025-09-14",
            distance_km=6.0,
        )
        assert "full_drive_stats" not in result
        assert "🛣️ Distance: 6 km (4 mi)" in result["description"]
        gate_result = check_description_redundancy(result)
        assert gate_result.passed, gate_result.message

    def test_unflagged_full_drive_distance_would_fail_the_gate(self, tmp_path):
        """Sanity check the gate side is actually exercised here: an
        un-prefixed line falsely flagged as full-drive still hard-fails."""
        gen = _StubGenerator(seo_storage_path=tmp_path)
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date="2025-09-14",
            distance_km=22.0,
        )
        result["full_drive_stats"] = ["distance"]
        gate_result = check_description_redundancy(result)
        assert not gate_result.passed
