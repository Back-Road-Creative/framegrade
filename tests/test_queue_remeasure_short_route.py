"""``gomoveshift queue remeasure-short-route``: re-derive the ``bins`` a Short's route
record lacks, from the day's persisted GPS points and the Short's clip range.

Dry run by default; ``--apply`` writes the sidecar only. Never defaults a missing
source, never touches the zero-tolerance gate. Synthetic data, ``tmp_path`` only.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
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


# --- stages from a RE-RENDER: the clip is identified by its frames, not its rank -------
# 2026-11-13, ireland-2025-09-17-mizen-s5-s7: the original stages are gone, and a
# re-render re-plans the day's Shorts, so "rank 3 of LF 1, 38 s long" may be another
# clip. With ``rendered_dir`` the published media must frame-match exactly one of the
# re-render's Shorts; that Short's clip range is the one measured.

SAME, NEAR, FAR = bytes([100] * 64), bytes([104] * 64), bytes([160] * 64)


def _rerender(tmp_path, entries):
    """A re-render's stages + shorts dir; ``entries`` = (rank, lf, start, end)."""
    d = tmp_path / "restages"
    d.mkdir()
    (d / f"gps_points_{DATE}.json").write_text(json.dumps(_points()))
    audit = [
        {"rank": r, "parent_lf_index": lf, "start_sec": s, "end_sec": e} for r, lf, s, e in entries
    ]
    (d / f"shorts_ost_audit_{DATE}.json").write_text(json.dumps({"shorts": audit}))
    out = tmp_path / "rerender" / "shorts"
    for r, lf, _s, _e in entries:
        (out / f"lf_{lf:02d}").mkdir(parents=True, exist_ok=True)
        (out / f"lf_{lf:02d}" / f"short_{r:02d}.mp4").write_bytes(b"x")
    return d, out


def _published(tmp_path):
    media = tmp_path / "old" / "short_02.mp4"
    media.parent.mkdir()
    media.write_bytes(b"x")
    return _sidecar(tmp_path, media=str(media)), media


def _sig(table):
    """Signature per file: the published media is ``old``, a re-render file its name."""
    return lambda path: table["old" if Path(path).parent.name == "old" else Path(path).name]


def _run_rerender(tmp_path, entries, table, apply=True):
    sc, _media = _published(tmp_path)
    stages, out = _rerender(tmp_path, entries)
    result = remeasure_short_route(
        _row(sc), stages, _Geo(), apply=apply, rendered_dir=out, signature_fn=_sig(table)
    )
    return sc, result


def test_rerender_measures_the_clip_whose_frames_match_whatever_its_rank(tmp_path):
    # The published Short was rank 2; in the re-render the same clip is rank 5.
    entries = [(2, 2, 300.0, 321.5), (5, 2, 100.0, 121.5)]
    table = {"short_02.mp4": FAR, "short_05.mp4": NEAR}
    sc, out = _run_rerender(tmp_path, entries, {**table, "old": SAME})
    assert out["rank"] == 5 and out["frame_distance"] == pytest.approx(4 / 255)
    assert _route_record(sc)["bins"]


def test_rerender_with_no_matching_clip_refuses_and_writes_nothing(tmp_path):
    entries = [(2, 2, 100.0, 121.5), (5, 2, 200.0, 221.5)]
    sc, _media = _published(tmp_path)
    before = sc.read_bytes()
    stages, out = _rerender(tmp_path, entries)
    table = {"short_02.mp4": FAR, "short_05.mp4": FAR, "old": SAME}
    with pytest.raises(RemeasureRefusedError, match="frame-match"):
        remeasure_short_route(
            _row(sc), stages, _Geo(), apply=True, rendered_dir=out, signature_fn=_sig(table)
        )
    assert sc.read_bytes() == before


def test_rerender_with_two_matching_clips_refuses(tmp_path):
    entries = [(2, 2, 100.0, 121.5), (5, 2, 200.0, 221.5)]
    with pytest.raises(RemeasureRefusedError, match="frame-match"):
        _run_rerender(tmp_path, entries, {"short_02.mp4": NEAR, "short_05.mp4": SAME, "old": SAME})


def test_rerender_refuses_a_match_with_a_near_miss_beside_it(tmp_path):
    almost = bytes([118] * 64)  # 18/255 = 0.07: neither a match nor clearly another clip
    entries = [(2, 2, 100.0, 121.5), (5, 2, 200.0, 221.5)]
    with pytest.raises(RemeasureRefusedError, match="ambiguous"):
        _run_rerender(
            tmp_path, entries, {"short_02.mp4": almost, "short_05.mp4": SAME, "old": SAME}
        )


def test_rerender_ignores_clips_of_another_lf_or_length(tmp_path):
    entries = [(2, 1, 100.0, 121.5), (5, 2, 100.0, 140.0), (7, 2, 200.0, 221.5)]
    table = {"short_02.mp4": SAME, "short_05.mp4": SAME, "short_07.mp4": NEAR, "old": SAME}
    _sc, out = _run_rerender(tmp_path, entries, table)
    assert out["rank"] == 7


def test_rerender_refuses_when_the_published_media_is_gone(tmp_path):
    sc = _sidecar(tmp_path, media=str(tmp_path / "gone" / "short_02.mp4"))
    stages, out = _rerender(tmp_path, [(2, 2, 100.0, 121.5)])
    with pytest.raises(RemeasureRefusedError, match="published media"):
        remeasure_short_route(
            _row(sc), stages, _Geo(), apply=True, rendered_dir=out, signature_fn=_sig({})
        )


def test_signature_distance_is_the_mean_absolute_difference():
    from gomoveshift.publishing.route_remeasure import signature_distance

    assert signature_distance(SAME, SAME) == 0.0
    assert signature_distance(SAME, FAR) == pytest.approx(60 / 255)
    with pytest.raises(RemeasureRefusedError, match="signature"):
        signature_distance(SAME, SAME[:10])


def test_cli_passes_rendered_shorts_through(tmp_path, monkeypatch, capsys):
    sc, _media = _published(tmp_path)
    stages, out = _rerender(tmp_path, [(5, 2, 100.0, 121.5)])
    conn = open_queue_db(str(tmp_path))
    enqueue(conn, "proj", 3, "youtube", "p1", sidecar_path=str(sc))
    conn.close()
    monkeypatch.setattr(queue_remeasure, "_geocoder", lambda: _Geo())
    monkeypatch.setattr("gomoveshift.publishing.route_remeasure.media_signature", lambda path: SAME)
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(
        queue_action="remeasure-short-route",
        id=1,
        stages_dir=stages,
        apply=True,
        rendered_shorts=out,
    )
    assert cmd_queue(args, config) == 0
    assert "frame distance" in capsys.readouterr().out
    assert _route_record(sc)["bins"]
