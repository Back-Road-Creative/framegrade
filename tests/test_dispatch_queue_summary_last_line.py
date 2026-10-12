"""The cron wrapper logs only ``tail -1`` of ``dispatch-queue --apply``, so the
summary line must be the LAST line printed -- per-delivery ``error:`` lines go
above it, never below."""

from __future__ import annotations

from unittest import mock

import pytest

from gomoveshift.cli.handlers.dispatch_queue import cmd_dispatch_queue
from gomoveshift.publishing.dispatch import DrainResult
from gomoveshift.publishing.queue import enqueue, open_queue_db

from test_cli_dispatch_queue import _args, _config, _record, _sidecar


@pytest.fixture(autouse=True)
def _no_pack_sweep(monkeypatch, tmp_path):
    """These tests are about the summary line, not the TikTok pack sweep: keep
    the sweep off and, belt and braces, off the real pack directory. Also keeps
    ``dispatch-queue --apply``'s notify seen file out of the live console dir."""
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))


def _run(tmp_path, capsys, result):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())
    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
    ):
        drain.return_value = result
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, tiktok="first_party"))
    assert rc == 0
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]


def test_summary_is_last_line_when_a_delivery_errored(tmp_path, capsys):
    err = "tiktok#41: privacy hold: no privacy attestation covers this delivery"
    lines = _run(tmp_path, capsys, DrainResult(held_privacy_unverified=4, errors=[err]))
    assert lines[-1].startswith("attempted=")
    assert "held_privacy_unverified=4" in lines[-1]
    assert f"  error: {err}" in lines[:-1]


def test_summary_is_last_line_on_a_clean_pass(tmp_path, capsys):
    lines = _run(tmp_path, capsys, DrainResult(attempted=1, succeeded=1))
    assert lines[-1].startswith("attempted=1 succeeded=1")
    assert not any(ln.startswith("  error:") for ln in lines)


def test_summary_is_last_line_and_carries_the_pack_counts(tmp_path, capsys, monkeypatch):
    """The cron log keeps ``tail -1``: the pack's own lines would be lost, so its
    counts ride on the summary."""
    from gomoveshift.publishing.tiktok_pack import PackSweepResult

    monkeypatch.setenv("GMS_TIKTOK_MODE", "pack")
    swept = PackSweepResult(considered=3, packed=2, errors=["delivery 9: drive offline"])
    with mock.patch("gomoveshift.cli.handlers.dispatch_queue.sweep_queued", return_value=swept):
        lines = _run(tmp_path, capsys, DrainResult(attempted=1, succeeded=1))
    assert lines[-1].startswith("attempted=1 succeeded=1")
    assert "tiktok_packed=2 tiktok_pack_errors=1" in lines[-1]
    assert "  error: delivery 9: drive offline" in lines[:-1]
