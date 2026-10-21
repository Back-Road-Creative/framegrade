"""Shorts drift in ``gomoveshift catalog-classify`` (catalog/classify.py).

A Short's place evidence is its own entry in ``shorts_ost_audit_<date>.json``
(the clip's GPS-voted place and regions), joined by the global short index the
drive persisted in ``pipeline_state.json``. Synthetic data only."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from gomoveshift.catalog.classify import classify_rows, load_short_evidence, write_outputs


def _stages(tmp_path, entries, index=None):
    stages = tmp_path / "projX" / "_stages"
    shorts = stages / "day_2026-04-01" / "shorts"
    shorts.mkdir(parents=True)
    paths = [str(shorts / f"short_{i:02d}.mp4") for i in range(1, len(entries) + 1)]
    gidx = index or {p: i for i, p in enumerate(paths, start=1)}
    (stages / "pipeline_state.json").write_text(json.dumps({"short_global_index": gidx}))
    audit = {
        "date": "2026-04-01",
        "shorts": [dict(e, rank=i, path=p) for i, (e, p) in enumerate(zip(entries, paths), 1)],
    }
    (stages / "shorts_ost_audit_2026-04-01.json").write_text(json.dumps(audit))
    return stages


def _entry(ost="Brookfield, Virginia", regions=("Ash County, Virginia",), source="clip_gps"):
    return {
        "parent_lf_index": 1,
        "ost_text": ost,
        "start_sec": 100.0,
        "end_sec": 140.0,
        "regions": list(regions),
        "place_source": source,
    }


def _short(vid, title, idx="1", project="projX"):
    return {
        "video_id": vid,
        "kind": "short",
        "duration_s": "40",
        "title": title,
        "project": project,
        "lf_index": "",
        "short_index": idx,
    }


def _loader(tmp_path, stages):
    return load_short_evidence(lambda p: stages if p == "projX" else tmp_path / "none")


def _classify(rows, short_for):
    out = classify_rows(
        rows, points_for=lambda p, lf: None, geocoder=SimpleNamespace(), short_for=short_for
    )
    return {r["video_id"]: r for r in out}


def test_loader_joins_the_global_short_index_to_the_clip_audit_entry(tmp_path):
    stages = _stages(tmp_path, [_entry(), _entry(ost="Cedarvale, Virginia")])
    short_for = _loader(tmp_path, stages)
    assert short_for("projX", "2")["ost_text"] == "Cedarvale, Virginia"
    assert short_for("projX", 1)["ost_text"] == "Brookfield, Virginia"
    assert short_for("projX", "9") is None
    assert short_for("projY", "1") is None
    assert short_for("projX", "") is None


def test_loader_matches_a_moved_stages_dir_by_its_day_relative_path(tmp_path):
    stages = _stages(tmp_path, [_entry()])
    moved = {"/old/root/projX/_stages/day_2026-04-01/shorts/short_01.mp4": 4}
    (stages / "pipeline_state.json").write_text(json.dumps({"short_global_index": moved}))
    assert _loader(tmp_path, stages)("projX", "4")["ost_text"] == "Brookfield, Virginia"


def test_d_when_the_title_names_the_clips_burned_place(tmp_path):
    stages = _stages(tmp_path, [_entry()])
    r = _classify(
        [_short("s1", "March on the road: Brookfield, Virginia #POVDriving #Shorts")],
        _loader(tmp_path, stages),
    )["s1"]
    assert r["class"] == "D" and "Brookfield" in r["reason"]
    assert r["record"]["ost_text"] == "Brookfield, Virginia"


def test_d_when_the_title_names_a_clip_region_or_its_smashed_hashtag(tmp_path):
    stages = _stages(
        tmp_path,
        [_entry(ost="Seven Mile Ford, Virginia", regions=("Smyth County, Virginia",))] * 2,
    )
    rows = _classify(
        [
            _short("s1", "A 38-second drive: Smyth County #Virginia #Shorts", "1"),
            _short("s2", "Back roads #SevenMileFord #Shorts", "2"),
        ],
        _loader(tmp_path, stages),
    )
    assert rows["s1"]["class"] == "D" and rows["s2"]["class"] == "D"


def test_drift_when_the_title_names_none_of_the_clips_places(tmp_path):
    stages = _stages(tmp_path, [_entry()])
    r = _classify(
        [_short("s1", "Mar 4, 1st stretch: Cedarvale, Virginia #Cedarvale #Shorts")],
        _loader(tmp_path, stages),
    )["s1"]
    assert r["class"] == "E" and r["drift"] is True
    assert "names none of the clip's places" in r["reason"]
    assert "Brookfield, Virginia" in r["reason"] and "Ash County, Virginia" in r["reason"]


def test_e_not_drift_when_the_burned_place_was_inherited_from_the_lf(tmp_path):
    stages = _stages(tmp_path, [_entry(source="lf_label")])
    r = _classify([_short("s1", "Cedarvale, Virginia #Shorts")], _loader(tmp_path, stages))["s1"]
    assert r["class"] == "E" and not r.get("drift")
    assert "inherited" in r["reason"]


def test_e_names_the_missing_short_evidence(tmp_path):
    stages = _stages(tmp_path, [_entry()])
    rows = _classify(
        [
            _short("s1", "t", idx="7"),
            _short("s2", "t", idx=""),
            _short("s3", "t", project=""),
        ],
        _loader(tmp_path, stages),
    )
    assert "no shorts OST audit entry for projX short 7" in rows["s1"]["reason"]
    assert "no short_index" in rows["s2"]["reason"]
    assert "no project" in rows["s3"]["reason"]
    assert {r["class"] for r in rows.values()} == {"E"}
    assert not any(r.get("drift") for r in rows.values())


def test_without_a_short_loader_shorts_stay_unresolved():
    r = classify_rows(
        [_short("s1", "Cedarvale, Virginia")], points_for=lambda p, lf: None, geocoder=None
    )[0]
    assert r["class"] == "E" and "clip window is not resolved" in r["reason"]


def test_drift_rows_go_to_a_review_worklist_never_to_worklist_ab(tmp_path):
    stages = _stages(tmp_path, [_entry(), _entry()])
    results = classify_rows(
        [
            _short("s1", "Cedarvale, Virginia #Shorts", "1"),
            _short("s2", "Brookfield, Virginia #Shorts", "2"),
        ],
        points_for=lambda p, lf: None,
        geocoder=SimpleNamespace(),
        short_for=_loader(tmp_path, stages),
    )
    counts = write_outputs(results, tmp_path / "out")
    assert counts == {"D": 1, "E": 1}
    drift = list(csv.DictReader((tmp_path / "out" / "worklist-shorts-drift.csv").open()))
    assert [d["video_id"] for d in drift] == ["s1"]
    assert drift[0]["burned_place"] == "Brookfield, Virginia"
    assert drift[0]["clip_regions"] == "Ash County, Virginia"
    assert drift[0]["live_title"] == "Cedarvale, Virginia #Shorts"
    assert "s1" not in (tmp_path / "out" / "worklist-AB.csv").read_text()


def test_cli_passes_the_short_loader(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers import catalog_classify as h

    stages = _stages(tmp_path, [_entry()])
    inv = tmp_path / "shorts.csv"
    with inv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_short("s1", "x")))
        w.writeheader()
        w.writerow(_short("s1", "Cedarvale, Virginia #Shorts"))
    monkeypatch.setattr(h, "_points_loader", lambda output_dir: lambda p, lf: None)
    monkeypatch.setattr(h, "_shorts_loader", lambda output_dir: _loader(tmp_path, stages))
    monkeypatch.setattr(h, "_geocoder", lambda: SimpleNamespace())
    monkeypatch.setattr(h, "_forward", lambda: None)
    ns = SimpleNamespace(
        inventory=str(inv), out_dir=str(tmp_path / "o"), kind="short", limit=None, output_dir=None
    )
    assert h.cmd_catalog_classify(ns, SimpleNamespace()) == 0
    drift = (tmp_path / "o" / "worklist-shorts-drift.csv").read_text()
    assert "s1" in drift
