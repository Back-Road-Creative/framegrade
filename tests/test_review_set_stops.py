"""review_set_stops: per-sequence stop states and edit decisions (synthetic npz/receipts)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.edit_timeline import EditTimeline, load_receipt
from gomoveshift.pipeline.review_set_stops import (
    SCHEMA_ID,
    StopsSchemaError,
    build_stops,
    load_stops,
    write_stops,
)
from gomoveshift.pipeline.stop_mask import MotionState, mask_summary

FIXTURE = Path(__file__).parent / "fixtures" / "assembly_receipt_v1.json"
M, S, C, U = (int(x) for x in (0, 1, 2, 3))


def _npz(tmp_path, states, name="stop_mask_2026-01-02.npz", mask=None):
    """Synthetic stop-mask npz built from an int8 state series."""
    st = np.asarray(states, dtype=np.int8)
    m = (st == MotionState.STOPPED) if mask is None else np.asarray(mask, dtype=bool)
    p = tmp_path / name
    np.savez_compressed(p, mask=m, states=st)
    return p


def _runs(doc):
    return [(r["state"], r["start_s"], r["end_s"], r["decision"]) for r in doc["runs"]]


def test_state_runs_match_npz_exactly(tmp_path):
    states = [M, M, S, S, S, C, C, U, M, S]
    doc = build_stops(_npz(tmp_path, states))
    assert _runs(doc) == [
        ("MOVING", 0, 2, "kept"),
        ("STOPPED", 2, 5, "cut"),
        ("CREEPING", 5, 7, "kept"),
        ("UNCERTAIN", 7, 8, "kept"),
        ("MOVING", 8, 9, "kept"),
        ("STOPPED", 9, 10, "cut"),
    ]
    assert doc["total_seconds"] == 10
    assert doc["sequence"] == "2026-01-02"
    assert doc["schema"] == SCHEMA_ID


def test_runs_tile_the_series_with_no_gap_or_overlap(tmp_path):
    rng = np.random.default_rng(7)
    states = rng.integers(0, 4, size=200)
    doc = build_stops(_npz(tmp_path, states))
    pos = 0
    for r in doc["runs"]:
        assert r["start_s"] == pos and r["end_s"] > r["start_s"]
        assert set(states[r["start_s"] : r["end_s"]]) == {MotionState[r["state"]].value}
        pos = r["end_s"]
    assert pos == 200


@pytest.mark.parametrize("states", [[M], [S], [S, S], [C, C, C], [M, S, M]])
def test_boundaries_single_and_edge_runs(tmp_path, states):
    doc = build_stops(_npz(tmp_path, states))
    assert doc["runs"][0]["start_s"] == 0
    assert doc["runs"][-1]["end_s"] == len(states)


def test_empty_series(tmp_path):
    doc = build_stops(_npz(tmp_path, []))
    assert doc["runs"] == [] and doc["total_seconds"] == 0
    assert doc["mask_summary"]["total_seconds"] == 0


def test_decision_consistent_with_mask_and_creeping_never_cut(tmp_path):
    states = [M, S, S, C, C, S, U, M]
    p = _npz(tmp_path, states)
    doc = build_stops(p)
    mask = np.load(p)["mask"]
    for r in doc["runs"]:
        seg = mask[r["start_s"] : r["end_s"]]
        assert seg.all() if r["decision"] == "cut" else not seg.any()
        if r["state"] == "CREEPING":
            assert r["decision"] == "kept"
    assert doc["mask_summary"] == mask_summary(mask)
    assert doc["state_seconds"] == {"MOVING": 2, "STOPPED": 3, "CREEPING": 2, "UNCERTAIN": 1}


def test_run_splits_where_mask_disagrees_with_states(tmp_path):
    # A mask that cuts one MOVING second must not hide inside a "kept" run.
    states = [M, M, M, M]
    doc = build_stops(_npz(tmp_path, states, mask=[0, 1, 1, 0]))
    assert _runs(doc) == [
        ("MOVING", 0, 1, "kept"),
        ("MOVING", 1, 3, "cut"),
        ("MOVING", 3, 4, "kept"),
    ]


def test_legacy_npz_without_states_is_derived_from_mask(tmp_path):
    p = tmp_path / "stop_mask_2026-01-03.npz"
    np.savez_compressed(p, mask=np.array([0, 1, 1, 0], dtype=bool))
    doc = build_stops(p)
    assert doc["states_source"] == "derived_from_mask"
    assert _runs(doc) == [
        ("MOVING", 0, 1, "kept"),
        ("STOPPED", 1, 3, "cut"),
        ("MOVING", 3, 4, "kept"),
    ]
    assert build_stops(_npz(tmp_path, [M]))["states_source"] == "npz"


def test_bad_npz_rejected(tmp_path):
    p = tmp_path / "x.npz"
    np.savez_compressed(p, mask=np.zeros(3, dtype=bool), states=np.array([0, 9, 0], dtype=np.int8))
    with pytest.raises(ValueError, match="states"):
        build_stops(p)
    np.savez_compressed(p, mask=np.zeros(3, dtype=bool), states=np.zeros(4, dtype=np.int8))
    with pytest.raises(ValueError, match="length"):
        build_stops(p)
    np.savez_compressed(p, other=np.zeros(3))
    with pytest.raises(ValueError, match="mask"):
        build_stops(p)


def test_no_receipt_omits_output_times_and_says_why(tmp_path):
    doc = build_stops(_npz(tmp_path, [M, M, S, M]))
    assert doc["output_times"] == {"available": False, "reason": "no assembly receipt supplied"}
    assert all("output_ranges" not in r for r in doc["runs"])
    assert doc["receipt"] is None


def test_receipt_output_ranges_match_edit_timeline(tmp_path):
    receipt = load_receipt(FIXTURE)
    tl = EditTimeline.from_receipt(receipt)
    # 400 s day: moving, a stop spanning the 10-40 span start, creeping, moving...
    states = np.zeros(400, dtype=np.int8)
    states[20:30] = S
    states[305:310] = C
    states[120:125] = S
    states[200:250] = U
    doc = build_stops(_npz(tmp_path, states), receipt=receipt)
    assert doc["output_times"] == {"available": True, "reason": None}
    assert doc["receipt"]["schema"] == "keycut-assembly-receipt/1"
    seen = 0
    for r in doc["runs"]:
        if r["decision"] == "cut":
            assert "output_ranges" not in r
            continue
        for o in r["output_ranges"]:
            seen += 1
            assert o["output_start_s"] == pytest.approx(tl.source_to_output(o["source_start_s"]))
            assert o["output_end_s"] == pytest.approx(
                tl.source_to_output_snapped(o["source_end_s"])
            )
            assert r["start_s"] <= o["source_start_s"] < o["source_end_s"] <= r["end_s"]
    assert seen > 0
    # First kept run [0,20) overlaps only the span 11.5..40 (source), output starts at 0.
    first = doc["runs"][0]
    assert first["output_ranges"] == [
        {
            "span_index": 0,
            "source_start_s": 11.5,
            "source_end_s": 20.0,
            "output_start_s": 0.0,
            "output_end_s": 8.5,
        }
    ]
    # A kept second the receipt never used (e.g. 200-250) has no output ranges.
    unused = [r for r in doc["runs"] if r["start_s"] <= 200 < r["end_s"]][0]
    assert unused["decision"] == "kept" and unused["output_ranges"] == []


def test_receipt_accepts_path_or_loaded(tmp_path):
    p = _npz(tmp_path, np.zeros(400, dtype=np.int8))
    a = build_stops(p, receipt=FIXTURE)
    b = build_stops(p, receipt=load_receipt(FIXTURE))
    assert a == b


def test_json_roundtrip_and_schema_checked_on_load(tmp_path):
    doc = build_stops(_npz(tmp_path, [M, S, S, C]))
    out = tmp_path / "nested" / "stops.json"
    write_stops(doc, out)
    assert json.loads(out.read_text())["schema"] == SCHEMA_ID
    assert load_stops(out) == doc
    assert load_stops(doc) == doc
    bad = copy.deepcopy(doc)
    bad["schema"] = "review-set-stops/99"
    with pytest.raises(StopsSchemaError, match="schema"):
        load_stops(bad)
    del bad["schema"]
    with pytest.raises(StopsSchemaError, match="schema"):
        load_stops(bad)
    out.write_text("{not json")
    with pytest.raises(StopsSchemaError, match="JSON"):
        load_stops(out)
    broken = copy.deepcopy(doc)
    del broken["runs"]
    with pytest.raises(StopsSchemaError, match="runs"):
        load_stops(broken)
    with pytest.raises(StopsSchemaError):
        write_stops({"schema": "other"}, tmp_path / "w.json")


def test_deterministic_output(tmp_path):
    p = _npz(tmp_path, [M, S, S, C, U])
    assert json.dumps(build_stops(p)) == json.dumps(build_stops(p))


def test_default_path_untouched():
    # Library only: importing the module registers nothing on the CLI.
    parser = Path(__file__).parent.parent / "gomoveshift" / "cli" / "parser.py"
    assert "review-set" not in parser.read_text(encoding="utf-8")
