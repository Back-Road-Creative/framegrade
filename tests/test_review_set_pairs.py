"""review_set_pairs: pair scan windows with output times (synthetic scan/stops documents)."""

import json

import pytest

from gomoveshift.pipeline.review_set_pairs import SCHEMA_ID, build_pairs
from gomoveshift.pipeline.review_set_stops import SCHEMA_ID as STOPS_SCHEMA, StopsSchemaError


def _stops(runs, available=True):
    return {
        "schema": STOPS_SCHEMA,
        "sequence": "synthetic",
        "states_source": "npz",
        "total_seconds": 100,
        "runs": runs,
        "state_seconds": {},
        "mask_summary": {},
        "output_times": {"available": available, "reason": None if available else "x"},
        "receipt": None,
    }


def _kept(a, b, out_start=None):
    run = {"state": "MOVING", "start_s": a, "end_s": b, "decision": "kept"}
    if out_start is not None:
        run["output_ranges"] = [
            {
                "span_index": 0,
                "source_start_s": float(a),
                "source_end_s": float(b),
                "output_start_s": out_start,
                "output_end_s": out_start + (b - a),
            }
        ]
    return run


def _cut(a, b):
    return {"state": "STOPPED", "start_s": a, "end_s": b, "decision": "cut"}


def _scan(**conds):
    return {
        "conditions": {
            name: {"status": "ok", "heuristic": False, "windows": [{"start_s": a, "end_s": b}]}
            for name, (a, b) in conds.items()
        }
    }


def test_kept_run_maps_midpoint_to_output_time():
    doc = build_pairs(_scan(tunnel=(10, 20)), _stops([_kept(0, 50, out_start=100.0)]))
    assert doc["schema"] == SCHEMA_ID
    assert doc["pairs"] == [[15.0, 115.0]]
    assert doc["detail"] == [
        {"condition": "tunnel", "window": [10.0, 20.0], "source_t": 15.0, "output_t": 115.0}
    ]
    assert doc["unpaired"] == []


def test_cut_run_is_unpaired():
    doc = build_pairs(_scan(creeping=(10, 20)), _stops([_cut(0, 50)]))
    assert doc["pairs"] == []
    assert doc["unpaired"] == [{"condition": "creeping", "window": [10.0, 20.0], "reason": "cut"}]


def test_no_receipt_is_never_estimated():
    doc = build_pairs(_scan(night=(10, 20)), _stops([_kept(0, 50)], available=False))
    assert doc["pairs"] == []
    assert doc["unpaired"][0]["reason"] == "no_output_times"


def test_outside_every_run_is_unpaired():
    doc = build_pairs(_scan(night=(90, 99)), _stops([_kept(0, 50, out_start=0.0)]))
    assert doc["unpaired"][0]["reason"] == "outside_runs"


def test_kept_second_dropped_by_the_receipt_is_unpaired():
    run = _kept(0, 50, out_start=0.0)
    run["output_ranges"][0]["source_end_s"] = 5.0
    doc = build_pairs(_scan(night=(10, 20)), _stops([run]))
    assert doc["unpaired"][0]["reason"] == "not_in_output"


def test_run_boundary_is_half_open_and_order_is_deterministic():
    scan = {
        "conditions": {
            "b": {"status": "ok", "heuristic": False, "windows": [{"start_s": 50, "end_s": 50}]},
            "a": {
                "status": "ok",
                "heuristic": False,
                "windows": [{"start_s": 30, "end_s": 40}, {"start_s": 2, "end_s": 4}],
            },
            "c": {"status": "no_input", "heuristic": False, "windows": []},
        }
    }
    stops = _stops([_kept(0, 50, out_start=0.0), _kept(50, 100, out_start=60.0)])
    doc = build_pairs(scan, stops)
    assert doc["pairs"] == [[3.0, 3.0], [35.0, 35.0], [50.0, 60.0]]
    assert json.loads(json.dumps(doc)) == doc


def test_bad_inputs_raise_value_errors():
    with pytest.raises(StopsSchemaError):
        build_pairs(_scan(a=(0, 1)), {"schema": "nope"})
    with pytest.raises(ValueError, match="conditions"):
        build_pairs({}, _stops([]))
