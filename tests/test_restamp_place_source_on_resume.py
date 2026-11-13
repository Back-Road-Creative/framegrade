"""A resume re-stamps a staged shorts sidecar's ``place_source`` from the evidence.

2026-11-13, nctopa-2026-03-05-rec0063: the shorts sidecar was written before
``day_pipeline`` recorded ``bracket_route``, so all 21 GPS-less Shorts carried
``lf_label``. A Phase-D-only resume never re-derives the sidecar, so check 20
(``places_verified``) refused every Short although the drive's bracket record
proves the exact burned text. The resume now re-stamps such entries; the place
gate itself is unchanged (``lf_label`` still proves nothing).
"""

from __future__ import annotations

import json
from pathlib import Path

from gomoveshift.pipeline.drive_orchestrator import restamp_unmeasured_place_source_on_resume

from _state_helpers import make_state

DATE = "2026-03-05"
BRACKET = {
    "label": "United States",
    "level": "country",
    "gap_h": 17.5,
    "before": {"time": "2026-03-04T23:28:05Z"},
    "after": {"time": "2026-03-05T17:00:12Z"},
}


def _entry(rank, text="United States", regions=(), source="lf_label"):
    return {
        "rank": rank,
        "ost_text": text,
        "regions": list(regions),
        "start_sec": 0.0,
        "end_sec": 30.0,
        "place_source": source,
    }


def _sidecar(base: Path, entries) -> Path:
    path = base / f"shorts_ost_audit_{DATE}.json"
    path.write_text(json.dumps({"date": DATE, "generated_at": "x", "shorts": entries}))
    return path


def _sources(path: Path) -> dict:
    return {e["rank"]: e["place_source"] for e in json.loads(path.read_text())["shorts"]}


def _restamp(state, tmp_path):
    return restamp_unmeasured_place_source_on_resume(
        state, output_base=tmp_path / "out", stages_dir=tmp_path
    )


def test_a_gps_less_lf_label_short_with_the_bracket_label_is_restamped(tmp_path):
    path = _sidecar(tmp_path, [_entry(1), _entry(2)])
    assert _restamp(make_state(bracket_route=dict(BRACKET)), tmp_path) == 2
    assert _sources(path) == {1: "bracket_route", 2: "bracket_route"}


def test_the_restamped_sidecar_passes_the_place_gate(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _unverified_reasons

    path = _sidecar(tmp_path, [_entry(1)])
    state = make_state(bracket_route=dict(BRACKET))
    _restamp(state, tmp_path)
    e = json.loads(path.read_text())["shorts"][0]
    evidence = {"bracket_route": BRACKET, "placeless_label": None}
    assert _unverified_reasons(e["ost_text"], e["regions"], e["place_source"], evidence) == []


def test_a_claim_free_placeless_label_is_restamped(tmp_path):
    path = _sidecar(tmp_path, [_entry(1, text="Somewhere On The Road")])
    assert _restamp(make_state(placeless_label="Somewhere On The Road"), tmp_path) == 1
    assert _sources(path) == {1: "placeless_label"}


def test_text_the_evidence_does_not_prove_stays_lf_label(tmp_path):
    path = _sidecar(tmp_path, [_entry(1, text="Canada")])
    before = path.read_text()
    assert _restamp(make_state(bracket_route=dict(BRACKET)), tmp_path) == 0
    assert path.read_text() == before


def test_a_bracket_record_past_its_gap_cap_restamps_nothing(tmp_path):
    path = _sidecar(tmp_path, [_entry(1)])
    assert _restamp(make_state(bracket_route={**BRACKET, "gap_h": 99}), tmp_path) == 0
    assert _sources(path) == {1: "lf_label"}


def test_a_short_with_clip_regions_is_never_restamped(tmp_path):
    path = _sidecar(tmp_path, [_entry(1, regions=["Ohio"])])
    assert _restamp(make_state(bracket_route=dict(BRACKET)), tmp_path) == 0
    assert _sources(path) == {1: "lf_label"}


def test_a_clip_gps_entry_is_left_alone(tmp_path):
    path = _sidecar(tmp_path, [_entry(1, regions=["Ohio"], source="clip_gps")])
    assert _restamp(make_state(bracket_route=dict(BRACKET)), tmp_path) == 0
    assert _sources(path) == {1: "clip_gps"}


def test_no_evidence_and_no_sidecar_are_no_ops(tmp_path):
    assert _restamp(make_state(), tmp_path) == 0
    path = _sidecar(tmp_path, [_entry(1)])
    assert _restamp(make_state(), tmp_path) == 0
    assert _sources(path) == {1: "lf_label"}


def test_an_unreadable_sidecar_is_skipped(tmp_path):
    (tmp_path / f"shorts_ost_audit_{DATE}.json").write_text("{not json")
    assert _restamp(make_state(bracket_route=dict(BRACKET)), tmp_path) == 0
