"""Live-catalog drift on the enqueue/dispatch gate.

`publishing-status --validate --live-catalog` names a published short the
channel no longer has, or whose live title drifted from the plan -- but only
when someone runs it. The publishing gate now reads the latest
`catalog-fetch` snapshot (`<storage_path>/live_catalog.latest.json`) when it
is at most 24 h old, so those problems hold their platform/project pair like
any other validate problem. It never fetches during a drain; a stale or
missing snapshot is a named warning, never a refusal.

Ids and titles below are synthetic.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from test_pub_status import _published_youtube

from gomoveshift.cli.handlers import catalog_fetch as cf
from gomoveshift.cli.handlers.pub_status import (
    LIVE_CATALOG_SNAPSHOT,
    gating_problems,
    live_catalog_snapshot_path,
)
from gomoveshift.publishing.queue import open_queue_db

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
_GONE = "SYNTHgone02"
_LIVE = "SYNTHlive02"


def _snapshot(tmp_path, videos, *, age_hours=1.0, fetched=True):
    body = {"enumerated": len(videos), "listed": len(videos), "videos": videos}
    if fetched:
        body["fetched_at"] = (NOW - timedelta(hours=age_hours)).isoformat()
    path = tmp_path / LIVE_CATALOG_SNAPSHOT
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _gate(conn, tmp_path):
    return gating_problems(conn, None, now=NOW.isoformat(), storage_path=tmp_path)


def test_the_snapshot_lives_beside_the_queue_db(tmp_path):
    assert live_catalog_snapshot_path(tmp_path) == tmp_path / "live_catalog.latest.json"


def test_a_fresh_snapshot_missing_a_published_video_gates_its_pair(tmp_path):
    conn, delivery_id = _published_youtube(tmp_path, _GONE)
    _snapshot(tmp_path, {})
    try:
        problems = _gate(conn, tmp_path)
    finally:
        conn.close()
    found = {(p.delivery_id, p.kind, p.platform, p.project) for p in problems}
    assert (delivery_id, "published_video_missing", "youtube", "proj") in found


def test_a_fresh_snapshot_with_a_drifted_title_gates_its_pair(tmp_path):
    conn, delivery_id = _published_youtube(tmp_path, _LIVE)
    _snapshot(tmp_path, {_LIVE: {"title": "Edited On YouTube", "privacy": "public"}})
    try:
        problems = _gate(conn, tmp_path)
    finally:
        conn.close()
    assert [(p.delivery_id, p.kind) for p in problems] == [(delivery_id, "published_title_drift")]


def test_a_matching_snapshot_adds_nothing(tmp_path, capsys):
    conn, _ = _published_youtube(tmp_path, _LIVE)
    _snapshot(tmp_path, {_LIVE: {"title": "The Title", "privacy": "public"}})
    try:
        assert _gate(conn, tmp_path) == []
    finally:
        conn.close()
    assert "live_catalog_" not in capsys.readouterr().out


def test_a_stale_snapshot_is_a_named_warning_not_a_refusal(tmp_path, capsys):
    conn, _ = _published_youtube(tmp_path, _GONE)
    _snapshot(tmp_path, {}, age_hours=25)
    try:
        assert _gate(conn, tmp_path) == []
    finally:
        conn.close()
    out = capsys.readouterr().out
    assert "warning: live_catalog_stale" in out
    assert "catalog-fetch --known-ids-from-queue" in out


def test_a_snapshot_with_no_fetch_time_counts_as_stale(tmp_path, capsys):
    conn, _ = _published_youtube(tmp_path, _GONE)
    _snapshot(tmp_path, {}, fetched=False)
    try:
        assert _gate(conn, tmp_path) == []
    finally:
        conn.close()
    assert "warning: live_catalog_stale" in capsys.readouterr().out


def test_a_missing_snapshot_is_a_named_warning_not_a_refusal(tmp_path, capsys):
    conn, _ = _published_youtube(tmp_path, _GONE)
    try:
        assert _gate(conn, tmp_path) == []
    finally:
        conn.close()
    assert "warning: live_catalog_missing" in capsys.readouterr().out


def test_an_unreadable_snapshot_is_a_named_warning_not_a_crash(tmp_path, capsys):
    conn, _ = _published_youtube(tmp_path, _GONE)
    (tmp_path / LIVE_CATALOG_SNAPSHOT).write_text("{not json", encoding="utf-8")
    try:
        assert _gate(conn, tmp_path) == []
    finally:
        conn.close()
    assert "warning: live_catalog_unreadable" in capsys.readouterr().out


def test_the_gate_never_fetches(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the gate called YouTube")

    monkeypatch.setattr(cf, "_build_readonly_youtube_service", boom)
    monkeypatch.setattr(cf, "fetch_catalog", boom)
    conn, _ = _published_youtube(tmp_path, _GONE)
    _snapshot(tmp_path, {}, age_hours=30)
    try:
        _gate(conn, tmp_path)
    finally:
        conn.close()


def test_without_a_storage_path_the_gate_ignores_the_catalog(tmp_path, capsys):
    conn, _ = _published_youtube(tmp_path, _GONE)
    _snapshot(tmp_path, {})
    try:
        assert gating_problems(conn, None, now=NOW.isoformat()) == []
    finally:
        conn.close()
    assert "live_catalog_" not in capsys.readouterr().out


def test_dispatch_holds_the_pair_a_fresh_snapshot_names(tmp_path, capsys, monkeypatch):
    from test_validate_gate import _dispatch

    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))
    conn, delivery_id = _published_youtube(tmp_path, _GONE)
    conn.close()
    _snapshot(tmp_path, {}, age_hours=0)
    monkeypatch.setattr("gomoveshift.cli.handlers.pub_status._utcnow", lambda: NOW, raising=True)

    rc, drain = _dispatch(tmp_path)

    assert rc == 0
    assert ("youtube", "proj") in drain.call_args.kwargs["held_scopes"]
    assert f"delivery={delivery_id} kind=published_video_missing" in capsys.readouterr().out


# --- catalog-fetch writes the snapshot -------------------------------------


def _fetch(tmp_path, monkeypatch, *, from_queue):
    open_queue_db(tmp_path).close()
    monkeypatch.setattr(cf, "_build_readonly_youtube_service", lambda: object())
    monkeypatch.setattr(cf, "enumerate_uploads", lambda svc, known_ids: [_LIVE])
    monkeypatch.setattr(
        cf,
        "fetch_catalog",
        lambda svc, ids, known_ids: {
            "enumerated": 1,
            "listed": 1,
            "videos": {_LIVE: {"title": "The Title"}},
            "checked_ids": sorted(set(ids) | set(known_ids)),
        },
    )
    monkeypatch.setattr(cf, "_utcnow", lambda: NOW)
    out = tmp_path / "run" / "live_catalog.json"
    args = SimpleNamespace(out=str(out), known_ids=None, known_ids_from_queue=from_queue)
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    assert cf.cmd_catalog_fetch(args, config) == 0
    return out


def test_catalog_fetch_from_the_queue_writes_a_stamped_snapshot(tmp_path, monkeypatch):
    out = _fetch(tmp_path, monkeypatch, from_queue=True)
    snap = json.loads((tmp_path / LIVE_CATALOG_SNAPSHOT).read_text())
    assert snap["fetched_at"] == NOW.isoformat()
    assert snap["videos"] == {_LIVE: {"title": "The Title"}}
    assert json.loads(out.read_text())["fetched_at"] == NOW.isoformat()


def test_catalog_fetch_without_the_queue_ids_leaves_the_snapshot_alone(
    tmp_path, monkeypatch, capsys
):
    """A pass that did not check every queue id by id could miss a video the
    queue holds, and the gate would hold that pair for nothing."""
    _fetch(tmp_path, monkeypatch, from_queue=False)
    assert not (tmp_path / LIVE_CATALOG_SNAPSHOT).exists()
    assert "--known-ids-from-queue" in capsys.readouterr().out


@pytest.mark.parametrize("age", [0, 23.9])
def test_fresh_means_at_most_24_hours(tmp_path, age):
    conn, _ = _published_youtube(tmp_path, _GONE)
    _snapshot(tmp_path, {}, age_hours=age)
    try:
        assert [p.kind for p in _gate(conn, tmp_path)] == ["published_video_missing"]
    finally:
        conn.close()
