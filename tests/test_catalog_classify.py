"""Tests for ``gomoveshift catalog-classify`` (catalog/classify.py).

Synthetic data only: invented towns, a fake geocoder and fake GPS points."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.catalog.classify import (
    CLASS_COLUMNS,
    classify_rows,
    load_lf_points,
    write_outputs,
)
from gomoveshift.cli import _COMMAND_MAP
from gomoveshift.cli.parser import _build_parser
from tools import retitle_apply as ra

_TOWNS = {
    0: ("Alderton", "Ash County", "Virginia"),
    1: ("Brookfield", "Ash County", "Virginia"),
    2: ("Cedarvale", "Elm County", "Virginia"),
}


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        if lon < -85.0:  # west of the synthetic trace: another state
            return {"town": None, "county": "Oak County", "state": "Tennessee"}
        town, county, state = _TOWNS.get(
            int(round((37.0 - lat) * 10)), (None, "Far County", "Virginia")
        )
        return {"town": town, "county": county, "state": state}


def _points(bands=(0, 1, 2), step=10.0):
    return [
        {"lat": 37.0 - b / 10, "lon": -81.0, "time": float(i * 4 * step + k * step)}
        for i, b in enumerate(bands)
        for k in range(4)
    ]


def _row(vid, title, project="projX", lf="1", kind="long", duration="120"):
    return {
        "video_id": vid,
        "kind": kind,
        "duration_s": duration,
        "title": title,
        "project": project,
        "lf_index": lf,
        "evidence_tier": "segment" if project else "none",
    }


def _points_for(project, lf_index):
    return _points() if project == "projX" else None


#: Forward-geocode fake: where a title's place sits, independent of the trace.
_PLACES = {
    "Dunmore, VA": (35.0, -81.0),  # ~180 km south of the trace
    "Eastfork, VA": (36.75, -81.0),  # ~5.6 km past the trace's end (Cedarvale 36.8)
    "Farley, VA": (36.35, -81.0),  # ~50 km past the trace's end
    "Glenrock, VA": (36.665, -81.0),  # ~15 km past the trace's end
    "Hollins, VA": (36.0, -90.0),  # the lookup's pick sits in another state
}


def _forward(query):
    return _PLACES.get(query)


def _by_id(rows, forward=_forward):
    return {
        r["video_id"]: r
        for r in classify_rows(rows, points_for=_points_for, geocoder=_Geo(), forward=forward)
    }


def test_d_title_whose_places_the_trace_supports():
    r = _by_id([_row("v1", "Alderton to Cedarvale, VA")])["v1"]
    assert r["class"] == "D" and r["suggested_title"] == ""


def test_a_wrong_order_title():
    r = _by_id([_row("v1", "Cedarvale to Alderton, VA")])["v1"]
    assert r["class"] == "A"
    assert "travel order" in r["reason"]
    assert r["suggested_title"] == "Alderton to Cedarvale, VA"


def test_a_place_the_trace_never_entered():
    r = _by_id([_row("v1", "Alderton to Dunmore, VA")])["v1"]
    assert r["class"] == "A" and "never entered" in r["reason"]
    assert "km from the trace" in r["reason"]
    assert r["suggested_title"] == "Alderton to Cedarvale, VA"


def test_the_lookup_uses_the_state_code_not_the_title_tail():
    # "in 4K" after the state must not reach the forward geocoder's query.
    asked = []

    def forward(query):
        asked.append(query)
        return _forward(query)

    r = _by_id([_row("v1", "Alderton to Dunmore, VA in 4K")], forward=forward)["v1"]
    assert asked == ["Dunmore, VA"]
    assert r["class"] == "A" and "km from the trace" in r["reason"]


def _one(title, points, duration, forward=_forward):
    return classify_rows(
        [_row("v1", title, duration=duration)],
        points_for=lambda project, lf: points,
        geocoder=_Geo(),
        forward=forward,
    )[0]


def _gapped_points():
    # Alderton and Brookfield at 0-70 s, then ~49 min unobserved, then Cedarvale.
    pts = _points(bands=(0, 1))
    pts += [{"lat": 36.8, "lon": -81.0, "time": 3000.0 + k * 10} for k in range(4)]
    return pts


def test_a_far_place_on_a_dense_trace_is_refuted():
    r = _one("Alderton to Farley, VA", _points(), "120")
    assert r["class"] == "A" and "50 km from the trace" in r["reason"]


def test_e_when_an_interior_gap_could_have_reached_the_place():
    # Spans 98% of the video, but the 49-minute hole could reach a place 50 km off.
    r = _one("Alderton to Farley, VA", _gapped_points(), "3100")
    assert r["class"] == "E" and r["suggested_title"] == ""
    assert "unobserved" in r["reason"] and "'Farley'" in r["reason"]


def test_e_when_the_missing_terminal_stretch_could_have_reached_the_place():
    # Fixes span 1100 s of 1370 s (80%): 270 s past the last fix are unobserved.
    r = _one("Alderton to Glenrock, VA", _points(step=100.0), "1370")
    assert r["class"] == "E" and "unobserved" in r["reason"]


def test_a_when_the_trace_covers_the_whole_video_and_the_place_is_beyond_reach():
    r = _one("Alderton to Glenrock, VA", _points(step=100.0), "1100")
    assert r["class"] == "A" and "15 km from the trace" in r["reason"]


def test_e_when_the_lookup_lands_in_another_region():
    # A same-name place elsewhere is no evidence about the title's place.
    r = _one("Alderton to Hollins, VA", _points(), "120")
    assert r["class"] == "E" and r["suggested_title"] == ""
    assert "'Hollins'" in r["reason"] and "Tennessee" in r["reason"]


def test_e_when_a_missing_place_lies_near_the_trace():
    # Absent from the record but within NEAR_KM: the reverse geocode simply
    # named no town there, so the title is not shown wrong.
    r = _by_id([_row("v1", "Alderton to Eastfork, VA")])["v1"]
    assert r["class"] == "E" and r["suggested_title"] == ""
    assert "'Eastfork'" in r["reason"] and "km of the trace" in r["reason"]


def test_e_when_a_missing_place_cannot_be_located():
    r = _by_id([_row("v1", "Alderton to Nowhere, VA")])["v1"]
    assert r["class"] == "E" and "cannot locate 'Nowhere'" in r["reason"]


def test_e_when_no_forward_geocoder_is_given():
    r = _by_id([_row("v1", "Alderton to Dunmore, VA")], forward=None)["v1"]
    assert r["class"] == "E" and "cannot locate 'Dunmore'" in r["reason"]


def test_wrong_order_stays_a_without_any_forward_lookup():
    def forward(query):
        raise AssertionError(f"unexpected lookup {query!r}")

    r = _by_id([_row("v1", "Cedarvale to Alderton, VA")], forward=forward)["v1"]
    assert r["class"] == "A"


def test_a_forward_lookup_failure_is_an_e_row():
    def forward(query):
        raise RuntimeError("nominatim timed out")

    r = _by_id([_row("v1", "Alderton to Dunmore, VA")], forward=forward)["v1"]
    assert r["class"] == "E" and "nominatim timed out" in r["reason"]


def test_b_endpoint_the_trace_only_passed_through():
    r = _by_id([_row("v1", "Brookfield to Cedarvale, VA")])["v1"]
    assert r["class"] == "B" and "passed through" in r["reason"]
    assert r["suggested_title"] == "Alderton to Cedarvale, VA"


def test_e_names_the_missing_evidence():
    rows = _by_id(
        [
            _row("noproj", "Alderton to Cedarvale, VA", project=""),
            _row("nolf", "Alderton to Cedarvale, VA", lf=""),
            _row("nogps", "Alderton to Cedarvale, VA", project="projNone"),
            _row("short", "Alderton to Cedarvale, VA", kind="short"),
        ]
    )
    assert {r["class"] for r in rows.values()} == {"E"}
    assert "no project" in rows["noproj"]["reason"]
    assert "lf_index" in rows["nolf"]["reason"]
    assert "no GPS points" in rows["nogps"]["reason"]
    assert "clip window" in rows["short"]["reason"]


def test_e_when_title_makes_no_claim_the_record_can_check():
    r = _by_id([_row("v1", "Scenic Drive Through the Valley")])["v1"]
    assert r["class"] == "E" and "no 'A to B' claim" in r["reason"]


def test_e_when_the_record_resolves_no_place():
    class _Blank:
        def reverse_geocode_detailed(self, lat, lon):
            return {}

    out = classify_rows(
        [_row("v1", "Alderton to Cedarvale, VA")], points_for=_points_for, geocoder=_Blank()
    )
    assert out[0]["class"] == "E" and "no geocoded place" in out[0]["reason"]
    assert out[0]["record"] is None


def test_e_when_the_trace_covers_too_little_of_the_video():
    # The trace spans 110 s; a 1000 s video is mostly unseen, so a place the
    # title names may be in the part with no GPS.
    out = classify_rows(
        [_row("v1", "Alderton to Brookfield, VA", duration="1000")],
        points_for=_points_for,
        geocoder=_Geo(),
    )
    assert out[0]["class"] == "E"
    assert "GPS covers 110 s of a 1000 s video" in out[0]["reason"]
    assert out[0]["suggested_title"] == ""


def test_e_when_the_video_duration_is_unknown():
    out = classify_rows(
        [_row("v1", "Alderton to Cedarvale, VA", duration="")],
        points_for=_points_for,
        geocoder=_Geo(),
    )
    assert out[0]["class"] == "E" and "no video duration" in out[0]["reason"]


def test_a_geocoder_failure_is_one_e_row_not_a_failed_batch():
    class _Flaky(_Geo):
        def reverse_geocode_detailed(self, lat, lon):
            if lat == 37.0:
                raise RuntimeError("Geocoding failed after 3 attempts")
            return super().reverse_geocode_detailed(lat, lon)

    def points_for(project, lf_index):
        return _points() if project == "projX" else _points(bands=(1, 2))

    out = classify_rows(
        [
            _row("v1", "Alderton to Cedarvale, VA"),
            _row("v2", "Brookfield to Cedarvale, VA", project="projY", duration="80"),
        ],
        points_for=points_for,
        geocoder=_Flaky(),
    )
    assert out[0]["class"] == "E" and "geocoder failed" in out[0]["reason"]
    assert "Geocoding failed after 3 attempts" in out[0]["reason"]
    assert out[0]["record"] is None
    assert out[1]["class"] == "D"


def test_a_without_a_supportable_replacement_moves_to_e():
    # Endpoints in two states: title_from_evidence refuses, so no suggestion.
    towns = dict(_TOWNS)
    towns[2] = ("Cedarvale", "Elm County", "Ohio")

    class _Geo2:
        def reverse_geocode_detailed(self, lat, lon):
            t, c, s = towns[int(round((37.0 - lat) * 10))]
            return {"town": t, "county": c, "state": s}

    out = classify_rows(
        [_row("v1", "Cedarvale to Alderton")], points_for=_points_for, geocoder=_Geo2()
    )
    assert out[0]["class"] == "E" and "no supportable replacement" in out[0]["reason"]


def test_order_is_by_video_id():
    out = classify_rows(
        [_row("vz", "Alderton to Cedarvale, VA"), _row("va", "Alderton to Cedarvale, VA")],
        points_for=_points_for,
        geocoder=_Geo(),
    )
    assert [r["video_id"] for r in out] == ["va", "vz"]


def test_outputs_and_worklist_ab_is_approved_by_retitle_apply(tmp_path):
    rows = classify_rows(
        [
            _row("v1", "Cedarvale to Alderton, VA"),
            _row("v2", "Brookfield to Cedarvale, VA"),
            _row("v3", "Alderton to Cedarvale, VA"),
            _row("v4", "x", project=""),
        ],
        points_for=_points_for,
        geocoder=_Geo(),
    )
    counts = write_outputs(rows, tmp_path)
    assert counts == {"A": 1, "B": 1, "D": 1, "E": 1}
    cls = list(csv.DictReader((tmp_path / "classification.csv").open()))
    assert list(cls[0]) == CLASS_COLUMNS
    assert [r["video_id"] for r in cls] == ["v1", "v2", "v3", "v4"]
    ev = [json.loads(x) for x in (tmp_path / "evidence.jsonl").read_text().splitlines()]
    assert [e["video_id"] for e in ev] == ["v1", "v2", "v3"]  # E (no project) has no record
    work = ra.read_worklist(tmp_path / "worklist-AB.csv")
    assert [w["video_id"] for w in work] == ["v1", "v2"]
    out = io.StringIO()
    results = ra.dry_run(work, out=out)
    assert [x["verdict"] for x in results] == ["APPROVED", "APPROVED"], out.getvalue()
    assert work[0]["current_title"] == "Cedarvale to Alderton, VA"


def test_load_lf_points_uses_the_persisted_state_loader(tmp_path):
    from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
    from gomoveshift.pipeline.drive_state_model import PipelineState

    gpx = tmp_path / "ep0.gpx"
    gpx.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1"><trk><trkseg>\n'
        '<trkpt lat="32.40" lon="-80.70"><time>2026-04-01T12:00:00Z</time></trkpt>\n'
        '<trkpt lat="32.41" lon="-80.71"><time>2026-04-01T12:00:25Z</time></trkpt>\n'
        "</trkseg></trk></gpx>\n"
    )
    seg = LFSegment(
        source_file_index=0,
        source_path=Path("/src/ep0.mp4"),
        start_time=0.0,
        end_time=1_000_000.0,
        original_start_time=10.0,
        original_end_time=40.0,
        composite_score=0.9,
        gps_center_lat=32.4,
        gps_center_lon=-80.7,
        terrain_type="coastal",
        contiguous_with_previous=False,
    )
    lf = RegularLFPlan(
        index=1,
        segments=[seg],
        montage_peaks=[],
        shorts_target=7,
        region="usa",
        total_duration=30.0,
    )
    plan = DrivePlan(
        source_files=[Path("/src/ep0.mp4")],
        processed_files=[Path("/proc/ep0.mp4")],
        regular_lfs=[lf],
        score_threshold=0.5,
        total_source_duration=100.0,
        total_highlight_duration=30.0,
    )
    state = PipelineState()
    state.drive_plan = plan
    state.gpx_files = [gpx]
    stages = tmp_path / "projX" / "_stages"
    stages.mkdir(parents=True)
    (stages / "pipeline_state.json").write_text(json.dumps(state.to_dict()))
    loader = load_lf_points(lambda p: stages if p == "projX" else tmp_path / "none")
    assert len(loader("projX", "1")) == 2
    assert loader("projY", "1") is None
    assert loader("projX", "9") is None


def test_cli_wiring_and_end_to_end(tmp_path, capsys, monkeypatch):
    from gomoveshift.cli.handlers import catalog_classify as h

    assert "catalog-classify" in _COMMAND_MAP
    args = _build_parser().parse_args(["catalog-classify", "--inventory", "a", "--out-dir", "b"])
    assert args.inventory == "a" and args.out_dir == "b" and args.kind == "long"
    inv = tmp_path / "longform.csv"
    with inv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["video_id", "kind", "title", "project", "lf_index", "duration_s"]
        )
        w.writeheader()
        w.writerow(
            {
                "video_id": "v1",
                "kind": "long",
                "title": "Cedarvale to Alderton, VA",
                "project": "projX",
                "lf_index": "1",
                "duration_s": "120",
            }
        )
        w.writerow({"video_id": "v2", "kind": "short", "title": "t", "project": "", "lf_index": ""})
    monkeypatch.setattr(h, "_points_loader", lambda output_dir: _points_for)
    monkeypatch.setattr(h, "_geocoder", lambda: _Geo())
    monkeypatch.setattr(h, "_forward", lambda: _forward)
    ns = SimpleNamespace(
        inventory=str(inv), out_dir=str(tmp_path / "out"), kind="long", limit=None, output_dir=None
    )
    assert h.cmd_catalog_classify(ns, SimpleNamespace()) == 0
    printed = capsys.readouterr().out
    assert "A: 1" in printed and "classification.csv" in printed
    assert "v2" not in (tmp_path / "out" / "worklist-AB.csv").read_text()


def test_cli_forward_geocoder_is_the_nominatim_search():
    from gomoveshift.cli.handlers import catalog_classify as h
    from gomoveshift.pipeline.chapters.nominatim_source import _nominatim_search

    assert h._forward() is _nominatim_search


def test_missing_geocoder_is_refused(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers import catalog_classify as h

    inv = tmp_path / "l.csv"
    inv.write_text("video_id,kind,title,project,lf_index\n")
    monkeypatch.setattr(h, "_geocoder", lambda: None)
    ns = SimpleNamespace(
        inventory=str(inv), out_dir=str(tmp_path / "o"), kind="long", limit=None, output_dir=None
    )
    with pytest.raises(SystemExit):
        h.cmd_catalog_classify(ns, SimpleNamespace())
