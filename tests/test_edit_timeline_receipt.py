"""Schema tests for the assembly receipt (keycut-assembly-receipt/1)."""

import copy
import json
from pathlib import Path

import pytest

from gomoveshift.pipeline.edit_timeline import (
    AssemblyReceipt,
    ReceiptError,
    load_receipt,
)

FIXTURE = Path(__file__).parent / "fixtures" / "assembly_receipt_v1.json"


def _raw():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_loads_from_path_and_dict():
    r = load_receipt(FIXTURE)
    assert isinstance(r, AssemblyReceipt)
    assert load_receipt(_raw()) == r
    assert len(r.spans) == 3
    assert r.spans[0].source_start > r.spans[0].merged_start  # snapped forward
    assert r.spans[2].requested_indices == (1, 2)


def test_round_trip_is_lossless():
    raw = _raw()
    assert load_receipt(raw).to_dict() == raw
    assert load_receipt(load_receipt(raw).to_dict()) == load_receipt(raw)


def test_receipt_is_frozen():
    r = load_receipt(_raw())
    with pytest.raises(Exception):
        r.provenance = "planned"  # type: ignore[misc]
    with pytest.raises(Exception):
        r.spans[0].method = "copy"  # type: ignore[misc]


def test_unknown_file_content_is_receipt_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{nope", encoding="utf-8")
    with pytest.raises(ReceiptError, match="receipt"):
        load_receipt(p)


def _mut(fn):
    raw = copy.deepcopy(_raw())
    fn(raw)
    return raw


BAD = [
    ("schema", lambda r: r.update(schema="keycut-assembly-receipt/2")),
    ("schema", lambda r: r.pop("schema")),
    ("provenance", lambda r: r.update(provenance="guess")),
    ("keycut_version", lambda r: r.update(keycut_version=3)),
    ("master.path", lambda r: r["master"].update(path=5)),
    ("master.size_bytes", lambda r: r["master"].update(size_bytes="big")),
    ("master.size_bytes", lambda r: r["master"].update(size_bytes=-1)),
    ("time_base", lambda r: r.update(time_base="frames")),
    ("max_gap_sec", lambda r: r.update(max_gap_sec=float("nan"))),
    ("max_gap_sec", lambda r: r.update(max_gap_sec=True)),
    ("requested[0]", lambda r: r["requested"].__setitem__(0, [40.0, 10.0])),
    ("requested[1]", lambda r: r["requested"].__setitem__(1, [1.0])),
    ("spans[0].method", lambda r: r["spans"][0].update(method="remux")),
    ("spans[0].reason", lambda r: r["spans"][0].update(reason="because")),
    ("spans[2].duration_source", lambda r: r["spans"][2].update(duration_source="guessed")),
    ("spans[1].index", lambda r: r["spans"][1].update(index=7)),
    ("spans[0].source_start", lambda r: r["spans"][0].update(source_start=float("inf"))),
    ("spans[0].source", lambda r: r["spans"][0].update(source_start=41.0)),
    ("spans[0].output", lambda r: r["spans"][0].update(output_start=29.0)),
    ("spans[0].source_end", lambda r: r["spans"][0].update(source_end=39.0)),
    ("spans[1].source_start", lambda r: r["spans"][1].update(source_start=301.0)),
    ("spans[0].source_start", lambda r: r["spans"][0].update(source_start=9.0)),
    ("spans[1].output_start", lambda r: r["spans"][1].update(output_start=30.0)),
    (
        "spans[0].output_start",
        lambda r: (
            r["spans"][0].update(output_start=1.0),
            r["spans"][1].update(output_start=r["spans"][0]["output_end"]),
        ),
    ),
    ("spans[0].requested_indices", lambda r: r["spans"][0].update(requested_indices=[9])),
    ("spans[0].requested_indices", lambda r: r["spans"][0].update(requested_indices=[])),
    ("spans[0].requested_indices[0]", lambda r: r["spans"][0].update(requested_indices=["0"])),
    ("spans[2].merged_end", lambda r: r["spans"][2].pop("merged_end")),
    ("output.path", lambda r: r["output"].update(path=None)),
    ("output.duration", lambda r: r["output"].update(duration=-2.0)),
    ("receipt", lambda r: None),
]


@pytest.mark.parametrize("field,fn", BAD, ids=[f"{f}-{i}" for i, (f, _) in enumerate(BAD)])
def test_invalid_receipts_name_the_field(field, fn):
    raw = _mut(fn)
    if field == "receipt":
        raw = []  # not an object
    with pytest.raises(ReceiptError) as exc:
        load_receipt(raw)
    assert field in str(exc.value)


def test_contiguity_tolerance_is_one_millisecond():
    raw = _raw()
    raw["spans"][1]["output_start"] += 5e-4
    load_receipt(raw)  # inside tolerance
    raw["spans"][1]["output_start"] += 2e-3
    with pytest.raises(ReceiptError, match=r"spans\[1\]\.output_start"):
        load_receipt(raw)


def test_null_optionals_are_accepted():
    raw = _raw()
    raw["keycut_version"] = None
    raw["provenance"] = "planned"
    raw["master"]["size_bytes"] = None
    raw["output"]["duration"] = None
    r = load_receipt(raw)
    assert r.to_dict() == raw
