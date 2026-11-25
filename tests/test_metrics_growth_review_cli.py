"""``gomoveshift metrics growth-review``: the weekly suggest-only review.
Reads the tracking store and the queue, prints lines, always exits 0."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from gomoveshift.cli.handlers.metrics import cmd_metrics
from gomoveshift.tracking import partner_progress


def _args(offline=True):
    return SimpleNamespace(metrics_command="growth-review", offline=offline)


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _no_records(monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.metrics.TrackingStore",
        lambda: SimpleNamespace(load_all_records=lambda: []),
    )


def test_offline_without_queue_reports_and_exits_zero(monkeypatch, capsys, tmp_path):
    _no_records(monkeypatch)
    rc = cmd_metrics(_args(), config=_config(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    assert "PARTNER_PROGRESS skipped=offline" in out
    assert "POST_CHECK_SKIPPED reason=no queue at" in out
    assert out.strip().splitlines()[-1].startswith("GROWTH_REVIEW window_days=28 groups=0")


def test_unreadable_queue_is_a_line_not_a_crash(monkeypatch, capsys, tmp_path):
    _no_records(monkeypatch)
    (tmp_path / "social_queue.sqlite3").write_text("not a database")
    rc = cmd_metrics(_args(), config=_config(tmp_path))
    assert rc == 0
    assert "POST_CHECK_FAILED reason=" in capsys.readouterr().out


def test_empty_queue_finds_no_problems(monkeypatch, capsys, tmp_path):
    _no_records(monkeypatch)
    conn = sqlite3.connect(tmp_path / "social_queue.sqlite3")
    conn.execute(
        "CREATE TABLE deliveries (id INTEGER PRIMARY KEY, project_name TEXT, short_index INTEGER, "
        "platform TEXT, status TEXT, error_class TEXT, scheduled_at TEXT, published_at TEXT, "
        "platform_post_id TEXT, platform_upload_id TEXT, updated_at TEXT, shadow INTEGER DEFAULT 0)"
    )
    conn.commit()
    conn.close()
    rc = cmd_metrics(_args(), config=_config(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    assert "failed=0 stuck=0 duplicate=0" in out
    assert "POST_CHECK" not in out


def test_live_run_prints_partner_lines(monkeypatch, capsys, tmp_path):
    _no_records(monkeypatch)
    monkeypatch.setattr("gomoveshift.cli.handlers.metrics.YouTubeCollector", lambda: object())
    monkeypatch.setattr(
        partner_progress,
        "fetch_channel_counts",
        lambda collector, records, today: partner_progress.ChannelCounts(subscribers=250),
    )
    rc = cmd_metrics(_args(offline=False), config=_config(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0
    assert "PARTNER_PROGRESS tier=fan_funding met=no subs=250/500" in out
    assert "PARTNER_PROGRESS tier=ad_revenue met=no subs=250/1000" in out
