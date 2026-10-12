"""Tests for PipelineState dataclass — round-trip, legacy dict, optional keys."""

from pathlib import Path

from gomoveshift.pipeline.drive_state_model import PipelineState


def test_round_trip():
    """Create PipelineState, to_dict, from_dict — fields must survive."""
    st = PipelineState(
        source_dir=Path("/data/src"),
        output_base=Path("/data/out"),
        region="usa",
        publish_day="sat",
        location="Springfield, IL",
        episode_files=["/data/src/ep_000.mp4", "/data/src/ep_001.mp4"],
        file_durations=[120.0, 300.0],
        file_boundaries=[(0.0, 120.0), (120.0, 420.0)],
        processed_files={0: Path("/data/out/ep_000_FINAL.mp4")},
        phases_completed=["phase_a", "phase_b_preview"],
    )
    d = st.to_dict()
    restored = PipelineState.from_dict(d)

    assert restored.source_dir == Path("/data/src")
    assert restored.output_base == Path("/data/out")
    assert restored.region == "usa"
    assert restored.publish_day == "sat"
    assert restored.location == "Springfield, IL"
    assert restored.episode_files == ["/data/src/ep_000.mp4", "/data/src/ep_001.mp4"]
    assert restored.file_durations == [120.0, 300.0]
    assert restored.file_boundaries == [(0.0, 120.0), (120.0, 420.0)]
    assert restored.processed_files == {0: Path("/data/out/ep_000_FINAL.mp4")}
    assert restored.phases_completed == ["phase_a", "phase_b_preview"]


def test_from_legacy_dict():
    """Dict with string tuples (JSON-round-tripped) -> real tuples."""
    legacy = {
        "source_dir": "/data/src",
        "output_base": "/data/out",
        "region": "ireland",
        "file_boundaries": [[0.0, 120.0], [120.0, 420.0]],
        "gps_per_episode": [["gps_data_placeholder", 0.0], [None, 1.5]],
    }
    st = PipelineState.from_dict(legacy)
    assert st.file_boundaries == [(0.0, 120.0), (120.0, 420.0)]
    assert st.gps_per_episode == [("gps_data_placeholder", 0.0), (None, 1.5)]
    assert st.source_dir == Path("/data/src")


def test_missing_optional_keys():
    """from_dict with minimal keys -> None for optional fields."""
    minimal = {
        "source_dir": "/data/src",
        "output_base": "/data/out",
        "region": "usa",
    }
    st = PipelineState.from_dict(minimal)
    assert st.source_dir == Path("/data/src")
    assert st.region == "usa"
    assert st.episode_files is None
    assert st.drive_scores is None
    assert st.processed_files is None
    assert st.drive_plan is None
    assert st.regular_lf_paths is None
    assert st.shorts_per_lf is None
    assert st.phases_completed is None


def test_int_key_coercion():
    """Dict with string keys for processed_files -> int keys."""
    d = {
        "source_dir": "/data/src",
        "output_base": "/data/out",
        "region": "usa",
        "processed_files": {"0": "/data/out/ep_000.mp4", "1": "/data/out/ep_001.mp4"},
        "shorts_per_lf": {"1": ["short_a.mp4"], "2": ["short_b.mp4"]},
        "delivered_lfs": [1, 2, 3],
    }
    st = PipelineState.from_dict(d)
    assert st.processed_files is not None
    assert all(isinstance(k, int) for k in st.processed_files.keys())
    assert st.processed_files[0] == Path("/data/out/ep_000.mp4")
    assert st.shorts_per_lf is not None
    assert all(isinstance(k, int) for k in st.shorts_per_lf.keys())
    assert st.delivered_lfs == {1, 2, 3}


# ── per_lf_ost_text field ─────────────────────────────────────


class TestPerLFOSTTextField:
    """``per_lf_ost_text`` is the single source of truth for OST burn-in,
    YouTube title, and thumbnail text.

    Mirrors ``per_lf_locations`` / ``per_lf_titles`` as an int-keyed dict
    of canonical route strings. Must round-trip through JSON with int
    keys preserved on resume.
    """

    def test_per_lf_ost_text_field_roundtrip(self):
        """PipelineState accepts and round-trips per_lf_ost_text."""
        st = PipelineState(
            source_dir=Path("/src"),
            output_base=Path("/out"),
            per_lf_ost_text={
                1: "Port Royal to Lady's Island via Sea Island Parkway, SC",
                2: "Asheville to Brevard via Blue Ridge Parkway, NC",
            },
        )
        d = st.to_dict()
        restored = PipelineState.from_dict(d)
        assert restored.per_lf_ost_text == {
            1: "Port Royal to Lady's Island via Sea Island Parkway, SC",
            2: "Asheville to Brevard via Blue Ridge Parkway, NC",
        }
        # Keys must be int (not strings) after JSON round-trip
        assert all(isinstance(k, int) for k in restored.per_lf_ost_text.keys())

    def test_per_lf_ost_text_defaults_none(self):
        """When omitted, the field defaults to None (optional field)."""
        st = PipelineState.from_dict({"source_dir": "/src", "output_base": "/out", "region": "usa"})
        assert st.per_lf_ost_text is None

    def test_per_lf_ost_text_str_key_coerced_to_int(self):
        """Legacy dict with string keys (JSON reload) -> int keys."""
        d = {
            "source_dir": "/src",
            "output_base": "/out",
            "per_lf_ost_text": {
                "1": "Port Royal to Lady's Island via Sea Island Parkway, SC",
                "3": "Asheville to Brevard via Blue Ridge Parkway, NC",
            },
        }
        st = PipelineState.from_dict(d)
        assert st.per_lf_ost_text is not None
        assert set(st.per_lf_ost_text.keys()) == {1, 3}
        assert all(isinstance(k, int) for k in st.per_lf_ost_text.keys())
