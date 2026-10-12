"""Tests for _drive_state serialization — especially string-key dict handling.

Covers dataclass serialization (RenderedShort) and int/str key
normalization on load.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from gomoveshift.pipeline._drive_state import (
    _serialize_state,
    _serialize_value,
    load_drive_state,
    save_drive_state,
)
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.shorts_extractor import RenderedShort
from _state_helpers import make_state


# ── None preservation for fields with non-None class defaults ──────────


class TestNonNoneDefaultNonePreservation:
    """``PipelineState.to_dict`` must emit ``None`` for fields whose
    class default is non-None when the runtime value is ``None``.

    Background: ``to_dict`` skips
    every field with ``val is None``. For fields with non-None defaults
    (e.g. ``region: str = ""``, ``publish_day: str = "wed"``,
    ``fresh: bool = False``), this means a state where the field has
    been *explicitly* set to ``None`` round-trips back to the class
    default — losing the "explicitly unset" signal.

    Fields covered (non-None class defaults on ``PipelineState``):
        region, publish_day, location, fresh, skip_upload,
        skip_youtube_upload, skip_drive_upload, skip_csv, youtube_upload
    """

    def _non_none_default_field_names(self) -> list[str]:
        names = []
        for f in dataclasses.fields(PipelineState):
            default = (
                f.default
                if f.default is not dataclasses.MISSING
                else (f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
            )
            if default is not None:
                names.append(f.name)
        return names

    def test_non_none_default_fields_survive_explicit_none(self):
        """Each non-None-default field set to None must round-trip as None."""
        names = self._non_none_default_field_names()
        # Sanity: the dataclass should have at least the documented non-None
        # defaults; if this changes, update this expectation.
        assert "region" in names
        assert "publish_day" in names
        assert "fresh" in names

        state = PipelineState()
        for n in names:
            object.__setattr__(state, n, None)

        restored = PipelineState.from_dict(state.to_dict())
        for n in names:
            assert getattr(restored, n) is None, (
                f"Field {n!r} did not survive None round-trip: got {getattr(restored, n)!r}"
            )

    def test_default_constructed_state_does_not_emit_default_values_as_none(self):
        """Sanity: a default-constructed PipelineState (no explicit None
        overrides) should NOT serialize the non-None defaults as None.
        Only explicitly-None values flip on the new branch."""
        state = PipelineState()  # all non-None defaults at their class default
        d = state.to_dict()
        # region default is "", publish_day default is "wed" — neither is None,
        # so the serialized dict must carry the actual default values, not None.
        assert d.get("region") == ""
        assert d.get("publish_day") == "wed"
        assert d.get("fresh") is False


def test_serialize_string_key_dict():
    """Dict with string keys round-trips correctly."""
    state = make_state(route_profiles={"all": "/tmp/video.mp4", "day1": "/tmp/day1.mp4"})
    result = _serialize_state(state)
    assert isinstance(result["route_profiles"], dict)
    assert result["route_profiles"]["all"] == "/tmp/video.mp4"
    assert result["route_profiles"]["day1"] == "/tmp/day1.mp4"
    json.dumps(result)


def test_serialize_mixed_key_dict():
    """Dict with string keys and list values serializes without error."""
    state = make_state(
        shorts_per_lf={
            0: ["/tmp/short1.mp4", "/tmp/short2.mp4"],
            1: ["/tmp/short3.mp4"],
        },
    )
    result = _serialize_state(state)
    assert isinstance(result["shorts_per_lf"], dict)
    json.dumps(result)


def test_serialize_nested_dict():
    """Nested dicts are handled recursively."""
    state = {
        "metadata": {
            "episode": {
                "title": "Road Trip",
                "duration": 120,
                "published": True,
                "tags": None,
            }
        }
    }
    result = _serialize_state(state)
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["metadata"]["episode"], dict)
    assert result["metadata"]["episode"]["title"] == "Road Trip"
    assert result["metadata"]["episode"]["duration"] == 120
    assert result["metadata"]["episode"]["published"] is True
    assert result["metadata"]["episode"]["tags"] is None
    json.dumps(result)


def test_serialize_date_keyed_dict():
    """Date-keyed dict with list-of-dict values round-trips."""
    state = {
        "metadata": {
            "2026-02-22": [{"path": "short.mp4", "start": 10.5}],
        }
    }
    result = _serialize_state(state)
    inner = result["metadata"]
    assert isinstance(inner, dict)
    assert "2026-02-22" in inner
    assert isinstance(inner["2026-02-22"], list)
    assert inner["2026-02-22"][0]["path"] == "short.mp4"
    json.dumps(result)


def test_serialize_path_values_in_dict():
    """Path values inside dicts are converted to strings."""
    state = make_state(
        route_profiles={
            "all": Path("/tmp/video.mp4"),
            "day1": Path("/tmp/day1.mp4"),
        },
    )
    result = _serialize_state(state)
    assert isinstance(result["route_profiles"], dict)
    assert result["route_profiles"]["all"] == "/tmp/video.mp4"
    assert isinstance(result["route_profiles"]["all"], str)
    json.dumps(result)


def test_roundtrip_resume_no_error(tmp_path):
    """Save state with string-key dicts, load back, use dict() — no ValueError."""
    state_file = tmp_path / "pipeline_state.json"
    state = make_state(
        shorts_per_lf={
            1: [{"path": "short.mp4"}],
            2: [{"path": "short_all.mp4"}],
        },
        route_profiles={"all": "/tmp/video.mp4"},
        phases_completed=["phase_a", "phase_b_preview"],
    )
    save_drive_state(state_file, state)
    loaded = load_drive_state(state_file)

    shorts = dict(loaded.get("shorts_per_lf", {}))
    assert isinstance(shorts, dict)

    profiles = dict(loaded.get("route_profiles", {}))
    assert profiles["all"] == "/tmp/video.mp4"


def test_int_key_dict_still_works():
    """Existing behavior preserved for int-key dicts."""
    state = {"chapter_times": {0: "intro", 120: "highway", 300: "arrival"}}
    result = _serialize_state(state)
    assert isinstance(result["chapter_times"], dict)
    # Keys become strings for JSON compat
    assert "0" in result["chapter_times"]
    assert result["chapter_times"]["0"] == "intro"
    json.dumps(result)


def test_serialize_set_in_nested_dict():
    """Sets inside nested dicts are converted to sorted lists."""
    state = {
        "day_info": {
            "2026-02-22": {"tags": {"travel", "driving"}},
        }
    }
    result = _serialize_state(state)
    tags = result["day_info"]["2026-02-22"]["tags"]
    assert isinstance(tags, list)
    assert sorted(tags) == tags
    json.dumps(result)


def test_serialize_rendered_short_dataclass():
    """RenderedShort dataclass must serialize to a dict, not str().

    Bug: _serialize_value fell through to str(v) for dataclasses,
    writing 'RenderedShort(...)' strings into state. On reload,
    dict('[]') raised ValueError: dictionary update sequence element
    #0 has length 1; 2 is required.
    """
    short = RenderedShort(
        index=0,
        rank=1,
        start_time=10.0,
        end_time=40.0,
        duration=30.0,
        avg_motion_energy=0.5,
        peak_motion_energy=0.8,
        source_video="/tmp/lf_01.mp4",
        output_path="/tmp/short_01.mp4",
        location_text="Town A to Town B",
        subscribe_text="Subscribe",
    )
    result = _serialize_value(short)
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert result["index"] == 0
    assert result["start_time"] == 10.0
    assert result["output_path"] == "/tmp/short_01.mp4"
    json.dumps(result)


def test_serialize_shorts_per_lf_with_rendered_shorts():
    """shorts_per_lf with RenderedShort lists round-trips correctly."""
    short = RenderedShort(
        index=0,
        rank=1,
        start_time=10.0,
        end_time=40.0,
        duration=30.0,
        avg_motion_energy=0.5,
        peak_motion_energy=0.8,
        source_video="/tmp/lf.mp4",
        output_path="/tmp/s.mp4",
        location_text="Test",
        subscribe_text="Sub",
    )
    state = make_state(shorts_per_lf={1: [short], 2: [short, short]})
    result = _serialize_state(state)
    assert isinstance(result["shorts_per_lf"]["1"], list)
    assert isinstance(result["shorts_per_lf"]["1"][0], dict)
    assert result["shorts_per_lf"]["1"][0]["index"] == 0
    assert len(result["shorts_per_lf"]["2"]) == 2
    json.dumps(result)


def test_roundtrip_shorts_per_lf_resume(tmp_path):
    """Save shorts_per_lf with RenderedShort, reload, dict() must not crash."""
    short = RenderedShort(
        index=0,
        rank=1,
        start_time=10.0,
        end_time=40.0,
        duration=30.0,
        avg_motion_energy=0.5,
        peak_motion_energy=0.8,
        source_video="/tmp/lf.mp4",
        output_path="/tmp/s.mp4",
        location_text="Test",
        subscribe_text="Sub",
    )
    state_file = tmp_path / "pipeline_state.json"
    state = make_state(
        shorts_per_lf={1: [short], 2: []},
        phases_completed=["phase_a", "phase_b_preview"],
    )
    save_drive_state(state_file, state)
    loaded = load_drive_state(state_file)

    # This is the exact call that crashed: dict(loaded["shorts_per_lf"])
    # After PipelineState migration, int keys are coerced from JSON strings.
    shorts = dict(loaded.get("shorts_per_lf", {}))
    assert isinstance(shorts, dict)
    assert isinstance(shorts[1], list)
    assert isinstance(shorts[1][0], dict)


def test_serialize_dataclass_with_to_dict():
    """Dataclass with to_dict() uses that method for serialization."""

    @dataclass
    class FakeResult:
        name: str
        score: float

        def to_dict(self):
            return {"name": self.name, "score": self.score}

    result = _serialize_value(FakeResult(name="test", score=0.9))
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["score"] == 0.9


def test_serialize_dataclass_without_to_dict():
    """Dataclass without to_dict() uses dataclasses.asdict()."""

    @dataclass
    class SimpleData:
        x: int
        y: str

    result = _serialize_value(SimpleData(x=42, y="hello"))
    assert isinstance(result, dict)
    assert result["x"] == 42
    assert result["y"] == "hello"


# ── Int-keyed dict normalization on load (JSON round-trip bug) ──────────


class TestIntKeyedDictNormalization:
    """LF-indexed dicts (``per_lf_locations``, ``shorts_per_lf``, etc.)
    are written with int keys but JSON converts to strings on
    serialize. On load, callers that do ``d.get(lf.index, fallback)``
    with int ``lf.index`` miss every entry and silently fall back.

    Regression: every LF metadata JSON
    and every shorts CSV row showed the drive-wide location because
    phase_d's ``per_lf_locations_override.get(lf.index_int, location)``
    always hit the fallback branch. The fix at ``_drive_state.py``
    normalizes known int-keyed dicts back to int on load so callers
    don't have to care whether they're in a fresh run or a resume.
    """

    def test_per_lf_locations_keys_int_after_roundtrip(self, tmp_path):
        state_file = tmp_path / "pipeline_state.json"
        state = make_state(
            per_lf_locations={
                1: "Beaufort County, SC",
                2: "Colleton County, SC",
                7: "Macon County, NC",
            },
            phases_completed=["phase_a"],
        )
        save_drive_state(state_file, state)
        loaded = load_drive_state(state_file)
        locs = loaded.get("per_lf_locations")
        assert locs == {
            1: "Beaufort County, SC",
            2: "Colleton County, SC",
            7: "Macon County, NC",
        }, f"Keys must be int after round-trip; got {list(locs.keys())}"
        # int access must hit (this is what phase_d.py does)
        assert locs.get(1) == "Beaufort County, SC"
        assert locs.get(2) == "Colleton County, SC"
        assert locs.get(7) == "Macon County, NC"
        # string access should also still work (backward compat for any
        # caller that may have been using str keys)
        # (but primary contract is int)
        assert all(isinstance(k, int) for k in locs.keys()), (
            f"All per_lf_locations keys must be int, got types "
            f"{[type(k).__name__ for k in locs.keys()]}"
        )

    def test_shorts_per_lf_keys_int_after_roundtrip(self, tmp_path):
        state_file = tmp_path / "pipeline_state.json"
        state = make_state(
            shorts_per_lf={
                1: [{"path": "s1.mp4"}],
                2: [{"path": "s2.mp4"}, {"path": "s3.mp4"}],
            },
            phases_completed=["phase_a"],
        )
        save_drive_state(state_file, state)
        loaded = load_drive_state(state_file)
        spl = loaded.get("shorts_per_lf")
        assert list(spl.keys()) == [1, 2], (
            f"shorts_per_lf keys must be int after round-trip; got {list(spl.keys())}"
        )
        assert len(spl.get(1)) == 1
        assert len(spl.get(2)) == 2

    def test_youtube_uploads_keys_int_after_roundtrip(self, tmp_path):
        state_file = tmp_path / "pipeline_state.json"
        state = make_state(
            youtube_uploads={
                1: {"video_id": "abc", "url": "https://youtu.be/abc"},
                3: {"video_id": "def", "url": "https://youtu.be/def"},
            },
            phases_completed=["phase_a"],
        )
        save_drive_state(state_file, state)
        loaded = load_drive_state(state_file)
        yt = loaded.get("youtube_uploads")
        assert 1 in yt and 3 in yt, (
            f"youtube_uploads must have int keys 1, 3; got {list(yt.keys())}"
        )
        assert yt.get(1)["video_id"] == "abc"

    def test_per_day_completed_keys_stay_string(self, tmp_path):
        """``per_day_completed`` is keyed by date-string, NOT int LF
        index. Normalization must NOT convert these."""
        state_file = tmp_path / "pipeline_state.json"
        state = make_state(
            per_day_completed={
                "2026-02-20": {"master_path": "/tmp/m20.mp4"},
                "2026-02-21": {"master_path": "/tmp/m21.mp4"},
            },
            phases_completed=["phase_a"],
        )
        save_drive_state(state_file, state)
        loaded = load_drive_state(state_file)
        pdc = loaded.get("per_day_completed")
        assert "2026-02-20" in pdc, (
            f"per_day_completed keys must remain strings (dates); got {list(pdc.keys())}"
        )
        # Date strings shouldn't get coerced to int
        assert all(isinstance(k, str) for k in pdc.keys())

    def test_per_lf_ost_text_keys_int_after_roundtrip(self, tmp_path):
        """``state["per_lf_ost_text"]`` is int-keyed (one entry
        per LF). JSON serializes int dict keys as strings; _drive_state's
        load-time normalization must coerce them back to int so callers
        that do ``ost_text.get(lf.index, fallback)`` with an int key
        actually hit. Without this entry in ``int_keyed_dicts``, resume
        silently falls back to non-canonical text."""
        state_file = tmp_path / "pipeline_state.json"
        state = make_state(
            per_lf_ost_text={
                1: "Port Royal to Lady's Island via Sea Island Parkway, SC",
                2: "Asheville to Brevard via Blue Ridge Parkway, NC",
                7: "Macon Loop via US-64, NC",
            },
            phases_completed=["phase_a", "phase_bc_day_first"],
            regular_lf_paths=[],
            shorts_per_lf={},
        )
        save_drive_state(state_file, state)
        loaded = load_drive_state(state_file)
        ost = loaded.get("per_lf_ost_text")
        assert ost == {
            1: "Port Royal to Lady's Island via Sea Island Parkway, SC",
            2: "Asheville to Brevard via Blue Ridge Parkway, NC",
            7: "Macon Loop via US-64, NC",
        }, f"Keys must be int after round-trip; got {list(ost.keys())}"
        # int access must hit (this is what day_pipeline does for OST burn)
        assert ost.get(1).startswith("Port Royal to Lady's Island")
        assert all(isinstance(k, int) for k in ost.keys()), (
            f"All per_lf_ost_text keys must be int, got types "
            f"{[type(k).__name__ for k in ost.keys()]}"
        )
