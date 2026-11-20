"""Q4b: every ``dispatch-queue --apply`` tick confirms scheduled YouTube Shorts
that went public. Fakes only, no network."""

from __future__ import annotations

import logging
from unittest import mock

import pytest

from gomoveshift.cli.handlers.dispatch_queue import cmd_dispatch_queue
from gomoveshift.publishing import queue as q
from gomoveshift.publishing.dispatch import DrainResult

from test_cli_dispatch_queue import _args, _config, _record, _sidecar

MOD = "gomoveshift.cli.handlers.dispatch_queue"
SLOT = "2026-11-17T09:00:00+00:00"
REAL = "2026-11-17T09:03:41+00:00"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))


class _Svc:
    """videos().list(...).execute() fake returning one public video."""

    def __init__(self):
        self.calls = 0

    def videos(self):
        return self

    def list(self, part, id):  # noqa: A002
        return self

    def execute(self):
        self.calls += 1
        return {
            "items": [
                {
                    "id": "vid1",
                    "status": {"privacyStatus": "public", "publishAt": SLOT},
                    "snippet": {"publishedAt": REAL},
                }
            ]
        }


def _seed(tmp_path):
    conn = q.open_queue_db(tmp_path)
    did = q.enqueue(conn, "proj", 3, "youtube", "HASH1", scheduled_at=SLOT)
    conn.execute(
        "UPDATE deliveries SET status='uploaded', upload_kind='short', upload_account='a', "
        "platform_upload_id='vid1' WHERE id=?",
        (did,),
    )
    conn.commit()
    q.transition(
        conn,
        did,
        "uploaded",
        "published",
        platform_post_id="pl",
        published_at_source=q.PUBLISHED_AT_PLATFORM_SCHEDULED,
        platform_published_at=SLOT,
    )
    conn.close()
    _sidecar(tmp_path, _record())
    return did


def _tick(tmp_path, *, apply=True, youtube="first_party", service=None, confirm=None, **routes):
    with (
        mock.patch(f"{MOD}._build_routed_handlers"),
        mock.patch(f"{MOD}.drain_queue", return_value=DrainResult()),
        mock.patch(f"{MOD}.drain_first_comments"),
        mock.patch(f"{MOD}._build_youtube_comments_service", return_value=service),
        mock.patch(f"{MOD}._run_queue_repair"),
    ):
        if confirm is not None:
            with mock.patch(f"{MOD}.confirm_scheduled_live", confirm):
                return cmd_dispatch_queue(
                    _args(apply=apply), _config(tmp_path, youtube=youtube, **routes)
                )
        return cmd_dispatch_queue(_args(apply=apply), _config(tmp_path, youtube=youtube, **routes))


def _row(tmp_path, did):
    conn = q.open_queue_db(tmp_path)
    try:
        return conn.execute(
            "SELECT published_at_source, platform_published_at FROM deliveries WHERE id=?",
            (did,),
        ).fetchone()
    finally:
        conn.close()


def test_tick_confirms_live_and_logs_count(tmp_path, caplog):
    did = _seed(tmp_path)
    svc = _Svc()
    with caplog.at_level(logging.INFO):
        assert _tick(tmp_path, service=svc) == 0
    row = _row(tmp_path, did)
    assert row[0] == q.PUBLISHED_AT_PLATFORM_LIVE
    assert row[1] == REAL
    assert "CONFIRMED_LIVE count=1" in caplog.text


def test_tick_calls_confirm_once_with_conn_and_service(tmp_path):
    _seed(tmp_path)
    svc = _Svc()
    seen = []

    def confirm(conn, service):
        seen.append(service.videos())  # the lazy client builds on first use
        return 0

    assert _tick(tmp_path, service=svc, confirm=confirm) == 0
    assert seen == [svc]


def test_dry_run_never_calls_confirm_or_api(tmp_path):
    did = _seed(tmp_path)
    svc = _Svc()
    confirm = mock.Mock(return_value=0)
    assert _tick(tmp_path, apply=False, service=svc, confirm=confirm) == 0
    confirm.assert_not_called()
    assert svc.calls == 0
    assert _row(tmp_path, did)[0] == q.PUBLISHED_AT_PLATFORM_SCHEDULED


def test_raising_confirm_warns_and_keeps_exit_code(tmp_path, caplog):
    _seed(tmp_path)
    confirm = mock.Mock(side_effect=RuntimeError("quota"))
    with caplog.at_level(logging.WARNING):
        assert _tick(tmp_path, service=_Svc(), confirm=confirm) == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == f"{MOD}"]
    assert len(warnings) == 1
    assert "CONFIRMED_LIVE" in warnings[0].getMessage()
    assert "quota" in warnings[0].getMessage()


def test_service_build_failure_warns_and_keeps_exit_code(tmp_path, caplog):
    did = _seed(tmp_path)
    with (
        mock.patch(f"{MOD}._build_youtube_comments_service", side_effect=OSError("no creds")),
        caplog.at_level(logging.WARNING),
    ):
        with (
            mock.patch(f"{MOD}._build_routed_handlers"),
            mock.patch(f"{MOD}.drain_queue", return_value=DrainResult()),
            mock.patch(f"{MOD}.drain_first_comments"),
            mock.patch(f"{MOD}._run_queue_repair"),
        ):
            rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, youtube="first_party"))
    assert rc == 0
    assert "no creds" in caplog.text
    assert _row(tmp_path, did)[0] == q.PUBLISHED_AT_PLATFORM_SCHEDULED


def test_youtube_not_routed_skips_with_log_line(tmp_path, caplog):
    _seed(tmp_path)
    confirm = mock.Mock(return_value=0)
    with caplog.at_level(logging.INFO):
        assert _tick(tmp_path, youtube="metricool", tiktok="first_party", confirm=confirm) == 0
    confirm.assert_not_called()
    assert "CONFIRMED_LIVE skipped" in caplog.text
