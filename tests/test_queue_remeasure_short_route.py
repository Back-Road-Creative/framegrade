"""``gomoveshift queue remeasure-short-route``: re-derive the ``bins`` a Short's route
record lacks, from the day's persisted GPS points and the Short's clip range.

Dry run by default; ``--apply`` writes the sidecar only. Never defaults a missing
source, never touches the zero-tolerance gate. Synthetic data, ``tmp_path`` only.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from test_publishing_plan_io import _record, _write

from gomoveshift.cli.handlers import queue_remeasure
from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.pipeline.route_claims import recheck_place_text
from gomoveshift.publishing.models import MediaFingerprint, PublishPlan
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.route_remeasure import RemeasureRefusedError, remeasure_short_route

DATE = "2026-08-01"


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        return {
            "county": "Vernon County",
            "state": "Wisconsin",
            "town": "Viroqua",
            "city": "Viroqua",
        }


def _points():
    return [[float(t), 43.5, -90.9] for t in range(0, 400, 1)]


def _stages(tmp_path, *, rank=2, start=100.0, end=121.5, lf=2, points=True, audit=True):
    d = tmp_path / "stages"
    d.mkdir()
    if points:
        (d / f"gps_points_{DATE}.json").write_text(json.dumps(_points()))
    if audit:
        entry = {
            "rank": rank,
            "parent_lf_index": lf,
            "start_sec": start,
            "end_sec": end,
            "route_record": None,
        }
        (d / f"shorts_ost_audit_{DATE}.json").write_text(json.dumps({"shorts": [entry]}))
    return d


def _sidecar(tmp_path, *, media="/v/short_02.mp4", duration=21.5):
    media_fp = MediaFingerprint(media, 1234, duration, 1080, 1920, "h264", "a" * 64)
    plan = PublishPlan(shorts=(dataclasses.replace(_record(), media=media_fp),))
    return _write(tmp_path, plan)


def _row(sidecar):
    return {"project_name": "proj", "short_index": 3, "sidecar_path": str(sidecar)}


def _route_record(sidecar):
    return load_publish_plan(sidecar).shorts[0].route_record


def test_dry_run_measures_and_writes_nothing(tmp_path):
    sc = _sidecar(tmp_path)
    before = sc.read_bytes()
    out = remeasure_short_route(_row(sc), _stages(tmp_path), _Geo(), apply=False)
    assert out["bins"] > 0 and out["applied"] is False
    assert sc.read_bytes() == before


def test_apply_writes_a_record_the_place_gate_can_check(tmp_path):
    sc = _sidecar(tmp_path)
    remeasure_short_route(_row(sc), _stages(tmp_path), _Geo(), apply=True)
    record = _route_record(sc)
    assert record["bins"] and record["retained_seconds"] == pytest.approx(21.5)
    assert recheck_place_text(record, "Vernon County, Wisconsin") == []
    assert recheck_place_text({}, "Vernon County, Wisconsin") != []  # the unmeasured refusal


def test_apply_changes_only_route_record(tmp_path):
    sc = _sidecar(tmp_path)
    before = json.loads(sc.read_text())
    remeasure_short_route(_row(sc), _stages(tmp_path), _Geo(), apply=True)
    after = json.loads(sc.read_text())
    assert after["shorts"][0].pop("route_record")["bins"]
    before["shorts"][0].pop("route_record", None)
    assert after == before


def test_source_gone_refuses_and_writes_nothing(tmp_path):
    sc = _sidecar(tmp_path)
    before = sc.read_bytes()
    with pytest.raises(RemeasureRefusedError, match="gps_points_2026-08-01.json"):
        remeasure_short_route(_row(sc), _stages(tmp_path, points=False), _Geo(), apply=True)
    assert sc.read_bytes() == before


def test_audit_missing_refuses(tmp_path):
    sc = _sidecar(tmp_path)
    with pytest.raises(RemeasureRefusedError, match="shorts_ost_audit"):
        remeasure_short_route(_row(sc), _stages(tmp_path, audit=False), _Geo(), apply=True)


def test_clip_range_that_does_not_match_the_media_refuses(tmp_path):
    sc = _sidecar(tmp_path)
    with pytest.raises(RemeasureRefusedError, match="duration"):
        remeasure_short_route(_row(sc), _stages(tmp_path, end=160.0), _Geo(), apply=True)


def test_other_parent_lf_refuses(tmp_path):
    sc = _sidecar(tmp_path)
    with pytest.raises(RemeasureRefusedError, match="parent LF"):
        remeasure_short_route(_row(sc), _stages(tmp_path, lf=1), _Geo(), apply=True)


def test_rank_not_in_audit_refuses(tmp_path):
    sc = _sidecar(tmp_path)
    with pytest.raises(RemeasureRefusedError, match="rank 2"):
        remeasure_short_route(_row(sc), _stages(tmp_path, rank=5), _Geo(), apply=True)


def test_no_fixes_in_the_clip_refuses_instead_of_inventing_bins(tmp_path):
    sc = _sidecar(tmp_path)
    d = _stages(tmp_path, start=1000.0, end=1021.5)
    with pytest.raises(RemeasureRefusedError, match="no GPS fix"):
        remeasure_short_route(_row(sc), d, _Geo(), apply=True)


def test_record_that_already_has_bins_is_left_alone(tmp_path):
    sc = _sidecar(tmp_path)
    remeasure_short_route(_row(sc), _stages(tmp_path), _Geo(), apply=True)
    with pytest.raises(RemeasureRefusedError, match="already has"):
        remeasure_short_route(_row(sc), tmp_path / "stages", _Geo(), apply=True)


def test_media_name_without_a_rank_refuses(tmp_path):
    sc = _sidecar(tmp_path, media="/v/clip.mp4")
    with pytest.raises(RemeasureRefusedError, match="rank"):
        remeasure_short_route(_row(sc), _stages(tmp_path), _Geo(), apply=True)


def test_cli_dry_run_then_apply(tmp_path, monkeypatch, capsys):
    sc = _sidecar(tmp_path)
    conn = open_queue_db(str(tmp_path))
    enqueue(conn, "proj", 3, "youtube", "p1", sidecar_path=str(sc))
    conn.close()
    monkeypatch.setattr(queue_remeasure, "_geocoder", lambda: _Geo())
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    base = dict(
        queue_action="remeasure-short-route", id=1, stages_dir=_stages(tmp_path), apply=False
    )
    assert cmd_queue(SimpleNamespace(**base), config) == 0
    assert "dry run" in capsys.readouterr().out
    assert _route_record(sc) is None
    assert cmd_queue(SimpleNamespace(**{**base, "apply": True}), config) == 0
    assert _route_record(sc)["bins"]


def test_cli_refuses_with_the_reason(tmp_path, monkeypatch, capsys):
    sc = _sidecar(tmp_path)
    conn = open_queue_db(str(tmp_path))
    enqueue(conn, "proj", 3, "youtube", "p1", sidecar_path=str(sc))
    conn.close()
    monkeypatch.setattr(queue_remeasure, "_geocoder", lambda: _Geo())
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(
        queue_action="remeasure-short-route",
        id=1,
        stages_dir=_stages(tmp_path, points=False),
        apply=True,
    )
    assert cmd_queue(args, config) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_the_no_bins_refusal_names_the_command():
    (issue,) = recheck_place_text({}, "Vernon County, Wisconsin")
    assert "queue remeasure-short-route" in issue
