"""``gomoveshift queue edit-facebook-caption`` and ``FacebookReelsAdapter.update_description``.

No network: a Mock session and a fake adapter. Synthetic data, ``tmp_path`` only.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_publishing_plan_io import _record, _write

from gomoveshift.cli.handlers import queue_facebook_caption
from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.publishing.facebook_adapter import (
    FacebookPermanentError,
    FacebookReelsAdapter,
)
from gomoveshift.publishing.models import FacebookPayload, PlatformPayloads, PublishPlan
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db

CREDS = {"page_id": "page-1", "page_access_token": "tok", "graph_version": "v21.0"}
OLD = "Fort Fremont — worth the detour? Filmed February 2026."
NEW = "Somewhere On The Road"


def _resp(status_code=200, body=None):
    return Mock(status_code=status_code, json=Mock(return_value=body or {}))


def test_update_description_posts_to_the_video_node_and_reads_back():
    session = Mock(
        post=Mock(return_value=_resp(200, {"success": True})),
        get=Mock(return_value=_resp(200, {"id": "v-1", "description": NEW})),
    )
    out = FacebookReelsAdapter(CREDS, session=session).update_description("v-1", NEW)
    (url,), kwargs = session.post.call_args
    assert url == "https://graph.facebook.com/v21.0/v-1"
    assert kwargs["data"]["description"] == NEW
    assert kwargs["data"]["access_token"] == "tok"  # noqa: S105
    (gurl,), gkwargs = session.get.call_args
    assert gurl == "https://graph.facebook.com/v21.0/v-1"
    assert gkwargs["params"]["fields"] == "description"
    assert out == {"video_id": "v-1", "description": NEW}


def test_update_description_graph_error_raises_and_skips_readback():
    err = {"error": {"code": 100, "message": "bad"}}
    session = Mock(post=Mock(return_value=_resp(400, err)), get=Mock())
    with pytest.raises(FacebookPermanentError):
        FacebookReelsAdapter(CREDS, session=session).update_description("v-1", NEW)
    session.get.assert_not_called()


class _FakeAdapter:
    def __init__(self, readback=None):
        self.calls = []
        self._readback = readback

    def update_description(self, video_id, text):
        self.calls.append((video_id, text))
        return {"video_id": video_id, "description": self._readback or text}


def _sidecar(tmp_path):
    rec = dataclasses.replace(_record(), payloads=PlatformPayloads(facebook=FacebookPayload(OLD)))
    return _write(tmp_path, PublishPlan(shorts=(rec,)))


def _setup(tmp_path, monkeypatch, *, platform="facebook", status="published", upload="v-1"):
    sc = _sidecar(tmp_path)
    conn = open_queue_db(str(tmp_path))
    enqueue(conn, "proj", 3, platform, "p1", sidecar_path=str(sc))
    conn.execute(
        "UPDATE deliveries SET status=?, platform_upload_id=? WHERE id=1", (status, upload)
    )
    conn.commit()
    conn.close()
    fake = _FakeAdapter()
    monkeypatch.setattr(queue_facebook_caption, "_adapter", lambda config: fake)
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    return sc, fake, config


def _args(text=NEW, apply=False):
    return SimpleNamespace(queue_action="edit-facebook-caption", id=1, text=text, apply=apply)


def _desc(sc):
    return load_publish_plan(sc).shorts[0].payloads.facebook.description


def _row_state(tmp_path):
    conn = open_queue_db(str(tmp_path))
    try:
        return tuple(conn.execute("SELECT status, platform_upload_id FROM deliveries").fetchone())
    finally:
        conn.close()


def test_dry_run_prints_and_touches_nothing(tmp_path, monkeypatch, capsys):
    sc, fake, config = _setup(tmp_path, monkeypatch)
    before = sc.read_bytes()
    assert cmd_queue(_args(), config) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and OLD in out and NEW in out and "v-1" in out
    assert fake.calls == [] and sc.read_bytes() == before


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"platform": "youtube"}, "facebook"),
        ({"status": "queued"}, "published"),
        ({"upload": None}, "upload id"),
    ],
)
def test_refuses_a_row_that_is_not_a_published_facebook_post(
    tmp_path, monkeypatch, capsys, kw, reason
):
    sc, fake, config = _setup(tmp_path, monkeypatch, **kw)
    before = sc.read_bytes()
    assert cmd_queue(_args(apply=True), config) == 1
    assert reason in capsys.readouterr().err
    assert fake.calls == [] and sc.read_bytes() == before


def test_refuses_a_missing_row(tmp_path, monkeypatch, capsys):
    _sc, fake, config = _setup(tmp_path, monkeypatch)
    args = _args(apply=True)
    args.id = 99
    assert cmd_queue(args, config) == 1
    assert "no delivery 99" in capsys.readouterr().err
    assert fake.calls == []


def test_refuses_text_that_names_a_place(tmp_path, monkeypatch, capsys):
    sc, fake, config = _setup(tmp_path, monkeypatch)
    before = sc.read_bytes()
    text = "Fort Fremont — worth the detour?"
    assert cmd_queue(_args(text, apply=True), config) == 1
    err = capsys.readouterr().err
    assert "Fort" in err and "REFUSED" in err
    assert fake.calls == [] and sc.read_bytes() == before


def test_apply_edits_the_post_and_records_it_in_the_sidecar(tmp_path, monkeypatch, capsys):
    sc, fake, config = _setup(tmp_path, monkeypatch)
    before = json.loads(sc.read_text())
    assert cmd_queue(_args(apply=True), config) == 0
    assert fake.calls == [("v-1", NEW)]
    assert _desc(sc) == NEW
    after = json.loads(sc.read_text())
    after["shorts"][0]["payloads"]["facebook"]["description"] = OLD
    assert after == before  # only the description changed
    assert not list(tmp_path.glob("*.tmp"))
    assert _row_state(tmp_path) == ("published", "v-1")
    assert "wrote" in capsys.readouterr().out


def test_apply_readback_mismatch_exits_nonzero_and_leaves_the_sidecar(
    tmp_path, monkeypatch, capsys
):
    sc, _fake, config = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        queue_facebook_caption, "_adapter", lambda config: _FakeAdapter(readback="other")
    )
    before = sc.read_bytes()
    assert cmd_queue(_args(apply=True), config) == 1
    assert "read-back" in capsys.readouterr().err
    assert sc.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
