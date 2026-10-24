"""Regression tests for PipelineState key threading and round-trip.

Derived from an audit of how state keys are threaded.

Two test classes:

1. TestStateKeyThreading — one parameterized test per threaded key asserting
   that a value written at the producer site survives __getitem__, .get(),
   attribute access, and to_dict()/from_dict() round-trip. Protects against
   a future refactor accidentally removing a declared field or breaking the
   dict protocol, which would reintroduce a class of bugs.

2. TestStateSerializationFindings — tests documenting known
   serialization gaps.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.models import DrivePlan
from gomoveshift.pipeline.uploader import UploadManifest as _UploadManifest

from gomoveshift.pipeline.drive_state_model import PipelineState


# Keys classified as `threaded` by the audit — writer and consumer
# both exist in the drive pipeline, not counting legacy phase_b/phase_c.
# Each entry: (key, representative_value). The value mirrors the type hint
# on the dataclass field so coercion round-trips are exercised.
THREADED_KEYS: list[tuple[str, object]] = [
    # ── Phase A outputs ────────────────────────────────────────────────
    ("ingest_plan", {"episodes": [{"id": 1}]}),
    ("episode_locations", ["Fabletown", "Mockford"]),
    ("episode_profiles", [{"duration": 120.0}]),
    ("episode_files", [Path("/tmp/ep_001.mp4")]),
    ("gpx_files", [Path("/tmp/ep_001.gpx")]),
    ("gps_per_episode", [(10.5, 20.25), (11.5, 21.75)]),
    ("file_boundaries", [(0.0, 3600.0), (3600.0, 7200.0)]),
    ("file_durations", [3600.0, 3600.0]),
    ("episode_dates", ["2026-02-20", "2026-02-21"]),
    ("location", "Fabletown"),
    # drive_scores is a numpy array — its roundtrip is covered below.
    # Here we smoke-test the in-memory write/read without round-trip.
    # ── Phase BC day-first outputs ─────────────────────────────────────
    ("per_day_locations", {"2026-02-20": "Fabletown"}),
    ("day_artifacts", [{"date": "2026-02-20", "master": "day.mp4"}]),
    ("regular_lf_paths", [Path("/tmp/lf_01.mp4"), Path("/tmp/lf_02.mp4")]),
    ("per_lf_locations", {1: "Fabletown", 2: "Mockford"}),
    # ── Phase D outputs ────────────────────────────────────────────────
    ("metadata_per_lf", {1: {"title": "Fabletown"}}),
    ("drive_schedule", {"lf_slots": [{"lf": 1}]}),
    ("social_posts_per_lf", {1: [{"platform": "ig", "caption": "x"}]}),
    ("upload_manifests", {1: {"youtube_id": "abc"}}),
    ("youtube_uploads", {1: {"video_id": "xyz"}}),
    # ── Orchestrator / CLI overrides ───────────────────────────────────
    ("start_date", "2026-02-20"),
    ("shorts_per_day", 3),
    ("youtube_upload", True),
    ("run_start", "2026-02-20T10:00:00Z"),
    ("skip_upload", True),
    ("processed_files", {1: Path("/tmp/ep_001.mp4")}),
    ("phases_completed", ["phase_a", "phase_bc"]),
    ("delivered_lfs", {0, 1, 2}),
]


class TestStateKeyThreading:
    """Every threaded state key must support the dict protocol + attribute access.

    This catches regressions that would re-introduce a class of bugs
    (a key gets declared on the dataclass but the dict-protocol shim stops
    recognizing it, so writes go somewhere the reader can't reach).
    """

    @pytest.mark.parametrize("key,value", THREADED_KEYS, ids=[k for k, _ in THREADED_KEYS])
    def test_subscript_write_then_subscript_read(self, key: str, value: object):
        state = PipelineState()
        state[key] = value
        assert state[key] == value, f"state[{key!r}] subscript round-trip failed"

    @pytest.mark.parametrize("key,value", THREADED_KEYS, ids=[k for k, _ in THREADED_KEYS])
    def test_subscript_write_then_get_read(self, key: str, value: object):
        state = PipelineState()
        state[key] = value
        assert state.get(key) == value, f"state.get({key!r}) returned the wrong value"

    @pytest.mark.parametrize("key,value", THREADED_KEYS, ids=[k for k, _ in THREADED_KEYS])
    def test_subscript_write_then_attr_read(self, key: str, value: object):
        state = PipelineState()
        state[key] = value
        assert getattr(state, key) == value, f"state.{key} attr read returned the wrong value"

    @pytest.mark.parametrize("key,value", THREADED_KEYS, ids=[k for k, _ in THREADED_KEYS])
    def test_contains_after_write(self, key: str, value: object):
        state = PipelineState()
        state[key] = value
        assert key in state, f"{key!r} not reported by __contains__ after write"

    def test_every_threaded_key_is_a_declared_field(self):
        """Every key we're asserting is threaded must be a declared field on
        the dataclass. A dynamic-attribute write (field not declared) would
        survive in-memory but vanish on save/load.
        """
        declared = {f.name for f in dataclasses.fields(PipelineState)}
        threaded = {k for k, _ in THREADED_KEYS}
        missing = threaded - declared
        assert not missing, (
            f"These threaded keys are NOT declared PipelineState fields and "
            f"will be dropped on save/load (dynamic-attribute pattern): {sorted(missing)}"
        )


class TestStateSerializationRoundTrip:
    """Serialization round-trip tests for threaded keys.

    Covers the save → JSON → load cycle used by the drive pipeline's
    checkpoint/resume path.
    """

    # Subset of THREADED_KEYS with values that should round-trip cleanly
    # via to_dict() / from_dict(). Dataclass-valued fields (drive_plan,
    # ingest_plan) still have a re-hydration gap — tested
    # separately below.
    ROUND_TRIP_KEYS: list[tuple[str, object]] = [
        ("region", "ireland"),
        ("publish_day", "wed"),
        ("location", "Fabletown"),
        ("start_date", "2026-02-20"),
        ("shorts_per_day", 3),
        ("shorts_target", 9),
        ("shorts_gap", 60.0),
        ("youtube_upload", True),
        ("skip_upload", True),
        ("fresh", True),
        ("run_start", "2026-02-20T10:00:00Z"),
        ("phases_completed", ["phase_a", "phase_bc"]),
        ("episode_locations", ["Fabletown", "Mockford"]),
        ("episode_dates", ["2026-02-20", "2026-02-21"]),
        ("file_durations", [3600.0, 3600.0]),
        ("source_dir", Path("/tmp/src")),
        ("output_base", Path("/tmp/out")),
        ("file_boundaries", [(0.0, 3600.0)]),
        # ``gps_per_episode`` is intentionally absent — it is no longer serialized
        # (recovered from the .gpx files on resume). See
        # test_state_derived_fields_not_serialized.py.
        ("metadata_per_lf", {1: {"title": "x"}}),
        # ``upload_manifests`` values rehydrate to ``UploadManifest`` instances
        # at ``from_dict`` time. Writing a
        # plain dict simulates the legacy ad-hoc shape but the round-trip
        # produces the dataclass — this is the contract, not a regression.
        (
            "upload_manifests",
            {
                1: _UploadManifest(
                    project_name="proj_lf01",
                    folder_id="folder_x",
                    files={"short_01.mp4": "https://drive.google.com/x"},
                    completed=True,
                )
            },
        ),
        ("youtube_uploads", {1: {"video_id": "xyz"}}),
        ("processed_files", {1: Path("/tmp/ep_001.mp4")}),
        ("delivered_lf_paths", {1: Path("/tmp/lf_01.mp4")}),
        ("delivered_lfs", {0, 1, 2}),
    ]

    @pytest.mark.parametrize("key,value", ROUND_TRIP_KEYS, ids=[k for k, _ in ROUND_TRIP_KEYS])
    def test_to_dict_from_dict_roundtrip(self, key: str, value: object):
        state = PipelineState()
        state[key] = value
        serialized = state.to_dict()
        loaded = PipelineState.from_dict(serialized)
        assert loaded.get(key) == value, (
            f"{key!r} failed round-trip: wrote {value!r}, got {loaded.get(key)!r}"
        )

    def test_delivered_lfs_int_set_round_trips_correctly(self):
        """delivered_lfs is set[int] — integer values must round-trip cleanly.

        This could be a problem if the set contained
        Path objects. Verified: the set contains ints (LF indices), not Paths.
        This test pins that invariant — if a future change ever stores Paths
        here, the assertion about type preservation will catch it.
        """
        state = PipelineState()
        state["delivered_lfs"] = {0, 1, 2, 5, 7}
        loaded = PipelineState.from_dict(state.to_dict())
        assert isinstance(loaded.delivered_lfs, set)
        assert loaded.delivered_lfs == {0, 1, 2, 5, 7}
        # Every element must be an int — catches a future change that
        # accidentally stores Paths or other non-JSON-native types
        assert all(isinstance(x, int) for x in loaded.delivered_lfs)


class TestStateSerializationFindings:
    """Tests that document serialization findings.

    youtube_uploads_sha, numpy arrays and the drive_plan dataclass round-trip.
    """

    def test_youtube_uploads_sha_survives_round_trip(self):
        """youtube_uploads_sha is a declared PipelineState field; the SHA of
        the durable youtube_uploads.json sidecar must survive a save/load
        round-trip so the DoD validator can detect upload-history drift on
        resume.
        """
        state = PipelineState()
        state["youtube_uploads_sha"] = "abc123def456"  # pragma: allowlist secret
        loaded = PipelineState.from_dict(state.to_dict())
        assert loaded.get("youtube_uploads_sha") == "abc123def456"  # pragma: allowlist secret

    def test_drive_plan_dataclass_preserved_on_round_trip(self):
        """drive_plan re-hydrates as a DrivePlan, not a plain dict."""
        plan = DrivePlan(
            source_files=[Path("a.mp4")],
            processed_files=[Path("a.mp4")],
            regular_lfs=[],
            score_threshold=0.5,
            total_source_duration=10.0,
            total_highlight_duration=4.0,
        )
        state = PipelineState()
        state["drive_plan"] = plan
        loaded = PipelineState.from_dict(state.to_dict())
        assert isinstance(loaded.drive_plan, DrivePlan), (
            f"drive_plan reloaded as {type(loaded.drive_plan).__name__}, not DrivePlan"
        )
        assert loaded.drive_plan == plan

    def test_route_profiles_int_keys_preserved_on_round_trip(self):
        """Int keys on `route_profiles` must survive JSON round-trip.

        One might suspect route_profiles is missing from the int-key
        coercion list in `from_dict()`. Verification showed it IS in the
        coercion tuple (_INT_KEY_DICT_FIELDS in drive_state_model.py). This test pins that — if a
        future refactor drops it from the tuple, lookups by int key would
        silently miss and this assertion will catch it.
        """
        state = PipelineState()
        state["route_profiles"] = {1: {"speed": 45}, 2: {"speed": 60}}
        loaded = PipelineState.from_dict(state.to_dict())
        assert all(isinstance(k, int) for k in loaded.route_profiles.keys()), (
            f"route_profiles keys after round-trip: {list(loaded.route_profiles.keys())}"
        )

    def test_drive_scores_omitted_from_write_but_legacy_marker_still_read(self):
        """``drive_scores`` is NOT written into pipeline_state.json — it is
        recovered from ``drive_scores.npz`` on resume, so ``to_dict`` omits it.
        The ``{"__numpy__": ...}`` marker path stays on the READ side so a
        legacy checkpoint that predates the exclusion still rehydrates the
        array (backward compat)."""
        np = pytest.importorskip("numpy")
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = PipelineState()
        state["drive_scores"] = arr
        # Write side: excluded from the serialized state.
        assert "drive_scores" not in state.to_dict()
        # Read side: a legacy JSON carrying the numpy marker still loads.
        legacy = {
            "drive_scores": {
                "__numpy__": True,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "data": arr.tolist(),
            }
        }
        loaded = PipelineState.from_dict(legacy)
        assert isinstance(loaded.drive_scores, np.ndarray), (
            f"drive_scores reloaded as {type(loaded.drive_scores).__name__}"
        )
        np.testing.assert_array_equal(loaded.drive_scores, arr)


class TestStateDictProtocolInvariants:
    """Invariants that underpin the key-threading classifications.

    If any of these break, those classifications need to be revisited.
    """

    def test_setitem_accepts_any_key_as_dynamic_attr(self):
        """The dict-protocol __setitem__ accepts arbitrary keys (not just
        declared fields). This is why dead state writes
        compile silently.

        Documented, not proposed to change.
        """
        state = PipelineState()
        state["totally_made_up_key"] = "value"
        assert state.get("totally_made_up_key") == "value"

    def test_to_dict_drops_dynamic_attrs(self):
        """to_dict iterates dataclasses.fields() only — dynamic attrs
        (non-field keys) never reach the on-disk state file.
        """
        state = PipelineState()
        state["totally_made_up_key"] = "value"
        serialized = state.to_dict()
        assert "totally_made_up_key" not in serialized

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict only copies keys that match declared field names —
        legacy state files with removed fields load gracefully.
        """
        loaded = PipelineState.from_dict({"legacy_removed_field": "x", "region": "ireland"})
        assert loaded.region == "ireland"
        assert not hasattr(loaded, "legacy_removed_field")

    def test_contains_requires_non_none_value(self):
        """__contains__ returns False for fields that are None but True for
        any non-None value (including falsy ones like "" and False).

        This is wider than strict dict semantics — `"location" in state`
        returns True even when location is the empty-string class default —
        but it matches the implementation and the behaviour relied on by
        `state.get(key, default)` calls in the pipeline.
        """
        state = PipelineState()
        # Non-None defaults — present per current semantics
        assert "location" in state  # default is "" (non-None)
        assert "publish_day" in state  # default is "wed"
        assert "fresh" in state  # default is False (non-None)
        # None default — absent
        assert "source_dir" not in state
        state["source_dir"] = Path("/tmp/x")
        assert "source_dir" in state
