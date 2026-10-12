"""Tests for the per-LF provenance sidecar (additive artifact)."""

from __future__ import annotations

import json
from pathlib import Path

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.lf_provenance_sidecar import (
    LF_PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    build_lf_provenance_payload,
    write_lf_provenance_sidecar,
)

from _state_helpers import make_state

# Every per-LF key the schema promises, with its at-write-time value type.
_LF_KEYS: dict[str, tuple[type, ...]] = {
    "lf_index": (int,),
    "burned_ost_text": (str, type(None)),
    "source": (dict,),
    "footage_window": (dict,),
    "gps_coverage_pct": (float, type(None)),
    "endpoints": (dict,),
    "road_vote": (dict, type(None)),
    "awarded_tier": (str, type(None)),
    "tier_evidence": (dict, list, type(None)),
    "cut_from_master_fingerprint": (str, type(None)),
    "master_attestation_verified": (bool, type(None)),
    "null_reasons": (dict,),
}


def _state_full():
    ep = {"ep1.gpx": [(0.0, 900.0)], "ep2.gpx": [(0.0, 930.0), (930.0, 1830.0)]}
    loc = ResolvedLocation(
        start_label="Killarney",
        end_label="Dingle",
        endpoints={"start": [52.05, -9.5], "end": [52.14, -10.27]},
    )
    return make_state(
        per_lf_ost_text={0: "Killarney to Dingle via N86, Kerry"},
        per_lf_titles={0: "Killarney to Dingle via N86, Kerry"},
        per_lf_gpx_path={0: "/data/track.gpx"},
        per_lf_gpx_episode_paths={0: ep},
        per_lf_resolved_location={0: loc},
    )


def test_top_level_and_full_lf_schema_with_types() -> None:
    payload = build_lf_provenance_payload(_state_full())
    assert payload["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert "generated_at" in payload and len(payload["lfs"]) == 1
    lf = payload["lfs"][0]
    for key, allowed in _LF_KEYS.items():
        assert key in lf, f"missing key {key!r}"
        assert isinstance(lf[key], allowed), f"{key}={lf[key]!r} not in {allowed}"
    assert lf["awarded_tier"] is None and lf["tier_evidence"] is None
    assert "awarded_tier" in lf["null_reasons"]
    assert lf["burned_ost_text"] == "Killarney to Dingle via N86, Kerry"


def test_source_window_and_endpoints_derived_from_state() -> None:
    lf = build_lf_provenance_payload(_state_full())["lfs"][0]
    assert lf["source"]["gpx_path"] == "/data/track.gpx"
    assert sorted(lf["source"]["episode_paths"]) == ["ep1.gpx", "ep2.gpx"]
    # footage_window spans min start / max end across every episode range.
    assert lf["footage_window"] == {"start_sec": 0.0, "end_sec": 1830.0}
    assert lf["endpoints"]["start"] == {"lat": 52.05, "lon": -9.5, "label": "Killarney"}
    assert lf["endpoints"]["end"] == {"lat": 52.14, "lon": -10.27, "label": "Dingle"}


def test_missing_inputs_become_null_with_reasons_not_crashes() -> None:
    # LF present only by title — no GPX, no resolved location.
    lf = build_lf_provenance_payload(make_state(per_lf_titles={3: "Somewhere"}))["lfs"][0]
    assert lf["lf_index"] == 3
    assert lf["source"] == {"gpx_path": None, "episode_paths": None}
    assert lf["footage_window"] == {"start_sec": None, "end_sec": None}
    assert lf["endpoints"] == {"start": None, "end": None}
    assert lf["gps_coverage_pct"] is None and lf["road_vote"] is None
    assert lf["burned_ost_text"] is None
    for field in (
        "burned_ost_text",
        "source",
        "footage_window",
        "endpoints",
        "gps_coverage_pct",
        "road_vote",
    ):
        assert lf["null_reasons"].get(field), f"no reason for null {field}"


def test_burned_ost_text_recorded_and_null_reason_when_absent() -> None:
    # The canonical burn source (``per_lf_ost_text``) is recorded verbatim so
    # the sidecar can prove WHAT was actually burned; absent → null + reason.
    lf = build_lf_provenance_payload(_state_full())["lfs"][0]
    assert lf["burned_ost_text"] == "Killarney to Dingle via N86, Kerry"
    assert "burned_ost_text" not in lf["null_reasons"]
    lf = build_lf_provenance_payload(make_state(per_lf_titles={3: "Somewhere"}))["lfs"][0]
    assert lf["burned_ost_text"] is None
    assert lf["null_reasons"].get("burned_ost_text")


def test_road_vote_threaded_from_resolved_model() -> None:
    # The road-vote tally is carried on the persisted ResolvedLocation
    # (threaded from build_lf_title), so the sidecar records it without any
    # separate fwd-compat state map.
    loc = ResolvedLocation(
        start_label="Marion",
        end_label="Wytheville",
        endpoints={"start": [36.84, -81.52], "end": [36.95, -81.09]},
        road_votes={"I-81": 8, "South Main Street": 1},
    )
    state = make_state(
        per_lf_ost_text={0: "Marion to Wytheville via I-81, Virginia"},
        per_lf_resolved_location={0: loc},
    )
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["road_vote"] == {"I-81": 8, "South Main Street": 1}
    assert "road_vote" not in lf["null_reasons"]


def test_awarded_tier_and_evidence_recorded_from_state_map() -> None:
    state = _state_full()
    state.per_lf_awarded_tier = {0: 1}
    state.per_lf_tier_evidence = {0: {"corroborations": 3}}
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["awarded_tier"] == "1"
    assert lf["tier_evidence"] == {"corroborations": 3}
    assert "awarded_tier" not in lf["null_reasons"]


def test_empty_resolved_road_votes_records_the_reason() -> None:
    # ``ResolvedLocation.road_votes`` is the ONLY road-vote source; an empty
    # tally is a null with a reason, not a lookup in some other state map.
    state = make_state(
        per_lf_ost_text={0: "Somewhere"},
        per_lf_resolved_location={0: ResolvedLocation(road_votes={})},
    )
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["road_vote"] is None
    assert (
        lf["null_reasons"]["road_vote"]
        == "no road vote recorded (ResolvedLocation.road_votes empty)"
    )


def test_gps_coverage_pct_lights_up_from_state() -> None:
    state = _state_full()
    state.per_lf_gps_coverage_pct = {0: 87.44}
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["gps_coverage_pct"] == 87.4
    assert "gps_coverage_pct" not in lf["null_reasons"]


def test_tier_fields_survive_a_state_json_round_trip() -> None:
    # The proof that the three ladder-award maps are DECLARED fields rather
    # than dynamically attached: ``from_dict`` drops every unknown key, so a
    # round-trip through JSON is what an undeclared field cannot survive.
    state = _state_full()
    state.per_lf_gps_coverage_pct = {0: 87.4}
    state.per_lf_awarded_tier = {0: 2}
    state.per_lf_tier_evidence = {0: {"corroborations": 2, "gps_point_count": 900}}
    revived = PipelineState.from_dict(json.loads(json.dumps(state.to_dict(), default=str)))
    lf = build_lf_provenance_payload(revived)["lfs"][0]
    assert lf["gps_coverage_pct"] == 87.4
    assert lf["awarded_tier"] == "2"
    assert lf["tier_evidence"] == {"corroborations": 2, "gps_point_count": 900}
    assert "awarded_tier" not in lf["null_reasons"]


def test_indices_union_sorted_and_resolved_dict_from_json_roundtrip() -> None:
    # String keys + a plain-dict ResolvedLocation both survive a resume.
    state = make_state(
        per_lf_titles={"2": "B"},
        per_lf_gpx_path={0: "/a.gpx"},
        per_lf_resolved_location={1: {"start_label": "X", "endpoints": {"start": [1.0, 2.0]}}},
    )
    payload = build_lf_provenance_payload(state)
    assert [lf["lf_index"] for lf in payload["lfs"]] == [0, 1, 2]
    ep = payload["lfs"][1]["endpoints"]
    assert ep["start"] == {"lat": 1.0, "lon": 2.0, "label": "X"} and ep["end"] is None


def test_cut_from_master_fingerprint_recorded_from_per_day_completed() -> None:
    # The LF cites the identity of the day master it was cut from, read
    # from per_day_completed[date] (already recorded at encode time, cheap
    # size:mtime_ns — never a fresh hash of the 60 GB master).
    state = _state_full()
    state.per_lf_recording_date = {0: "2026-02-24"}
    state.per_day_completed = {"2026-02-24": {"master_fingerprint": "123:456"}}
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["cut_from_master_fingerprint"] == "123:456"
    assert "cut_from_master_fingerprint" not in lf["null_reasons"]


def test_cut_from_master_fingerprint_null_reason_when_no_recording_date() -> None:
    lf = build_lf_provenance_payload(_state_full())["lfs"][0]
    assert lf["cut_from_master_fingerprint"] is None
    assert lf["null_reasons"].get("cut_from_master_fingerprint")


def test_cut_from_master_fingerprint_null_reason_when_day_record_missing() -> None:
    state = _state_full()
    state.per_lf_recording_date = {0: "2026-02-24"}
    state.per_day_completed = {}
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["cut_from_master_fingerprint"] is None
    assert lf["null_reasons"].get("cut_from_master_fingerprint")


def test_master_attestation_verified_is_none_without_disk_check() -> None:
    # build_lf_provenance_payload is pure (state only) — the disk cross-check
    # against the actual attestation happens in write_lf_provenance_sidecar.
    state = _state_full()
    state.per_lf_recording_date = {0: "2026-02-24"}
    state.per_day_completed = {"2026-02-24": {"master_fingerprint": "123:456"}}
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["master_attestation_verified"] is None


class TestMasterAttestationCrossCheckOnWrite:
    """write_lf_provenance_sidecar has filesystem access to each day's
    privacy attestation — it cross-checks the LF's claimed master fingerprint
    against what was actually attested, and NEVER decides an enforcement
    policy: a mismatch is recorded, not refused."""

    def _state(self, tmp_path, fp="123:456"):
        state = _state_full()
        state.output_base = tmp_path
        state.per_lf_recording_date = {0: "2026-02-24"}
        state.per_day_completed = {"2026-02-24": {"master_fingerprint": fp}}
        return state

    def _write_attestation(self, tmp_path, master_fingerprint=None):
        day_dir = tmp_path / "_stages" / "day_2026-02-24"
        day_dir.mkdir(parents=True)
        payload = {"schema": "gms.privacy_attestation/v1", "date": "2026-02-24"}
        if master_fingerprint is not None:
            payload["master_fingerprint"] = master_fingerprint
        (day_dir / "day_2026-02-24_privacy_attestation.json").write_text(json.dumps(payload))

    def test_matching_fingerprint_verifies_true(self, tmp_path) -> None:
        self._write_attestation(tmp_path, master_fingerprint="123:456")
        written = write_lf_provenance_sidecar(tmp_path, self._state(tmp_path))
        lf = json.loads(written.read_text())["lfs"][0]
        assert lf["master_attestation_verified"] is True
        assert "master_attestation_verified" not in lf["null_reasons"]

    def test_mismatched_fingerprint_verifies_false_never_raises(self, tmp_path) -> None:
        self._write_attestation(tmp_path, master_fingerprint="999:999")
        written = write_lf_provenance_sidecar(tmp_path, self._state(tmp_path))
        lf = json.loads(written.read_text())["lfs"][0]
        assert lf["master_attestation_verified"] is False
        assert lf["null_reasons"].get("master_attestation_verified")

    def test_absent_attestation_is_none_not_a_pass(self, tmp_path) -> None:
        written = write_lf_provenance_sidecar(tmp_path, self._state(tmp_path))
        lf = json.loads(written.read_text())["lfs"][0]
        assert lf["master_attestation_verified"] is None
        assert lf["null_reasons"].get("master_attestation_verified")

    def test_attestation_without_master_fingerprint_is_none_not_a_pass(self, tmp_path) -> None:
        # An attestation written before this field existed — absent evidence
        # must never read as verified.
        self._write_attestation(tmp_path, master_fingerprint=None)
        written = write_lf_provenance_sidecar(tmp_path, self._state(tmp_path))
        lf = json.loads(written.read_text())["lfs"][0]
        assert lf["master_attestation_verified"] is None
        assert lf["null_reasons"].get("master_attestation_verified")


def test_write_file_skip_empty_and_oserror(tmp_path, monkeypatch) -> None:
    written = write_lf_provenance_sidecar(tmp_path, _state_full())
    assert written is not None and written.name == LF_PROVENANCE_FILENAME
    assert written.parent == tmp_path
    assert json.loads(written.read_text())["lfs"][0]["lf_index"] == 0
    # No per-LF data → no file written; OSError on write is swallowed.
    assert write_lf_provenance_sidecar(tmp_path / "empty", make_state()) is None
    monkeypatch.setattr(Path, "write_text", _boom)
    assert write_lf_provenance_sidecar(tmp_path / "boom", _state_full()) is None


def _boom(self, *args, **kwargs):
    raise OSError("disk full")
