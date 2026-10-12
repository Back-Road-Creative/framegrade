"""``gomoveshift metrics feed-share``: the interim-cadence exit rule
as a CLI line -- Shorts-feed share of channel views on each of the last
N measured days must be >= threshold. Report-only, always exits 0.
"""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.cli.handlers.metrics import cmd_metrics


def _args(**kwargs):
    base = dict(metrics_command="feed-share", days=7, threshold=0.70)
    base.update(kwargs)
    return SimpleNamespace(**base)


def _row(day, value, views):
    return {"day": day, "dimension": "insightTrafficSourceType", "value": value, "views": views}


def _patch(monkeypatch, rows, brand="gomoveshift"):
    monkeypatch.setattr("gomoveshift.cli.handlers.metrics.current_brand", lambda: brand)
    monkeypatch.setattr("gomoveshift.cli.handlers.metrics.load_channel_daily_rows", lambda b: rows)


def _days_rows(shares_by_day, other_source_views=0):
    """Build rows for {day: shorts_views_fraction_of_100} style shares --
    each day gets 100 total traffic-source views, `share*100` from SHORTS."""
    rows = []
    for day, share in shares_by_day.items():
        shorts_views = round(share * 100)
        other_views = 100 - shorts_views
        rows.append(_row(day, "SHORTS", shorts_views))
        if other_views:
            rows.append(_row(day, "YT_SEARCH", other_views))
    return rows


def test_met_when_all_days_meet_threshold(monkeypatch, capsys):
    shares = {f"2026-09-{10 + i:02d}": 0.75 for i in range(7)}
    _patch(monkeypatch, _days_rows(shares))

    rc = cmd_metrics(_args(), config=SimpleNamespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert "feed-share: gomoveshift last 7 measured days (threshold 0.7)" in out
    assert "2026-09-10 0.75 OK" in out
    assert "2026-09-16 0.75 OK" in out
    assert "verdict: MET (7/7 days >= 0.7)" in out


def test_not_met_when_a_day_is_below_threshold(monkeypatch, capsys):
    shares = {f"2026-09-{10 + i:02d}": 0.75 for i in range(7)}
    shares["2026-09-13"] = 0.40
    _patch(monkeypatch, _days_rows(shares))

    rc = cmd_metrics(_args(), config=SimpleNamespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert "2026-09-13 0.40 LOW" in out
    assert "verdict: NOT MET (6/7 days >= 0.7)" in out


def test_insufficient_when_fewer_than_n_measured_days(monkeypatch, capsys):
    shares = {f"2026-09-{10 + i:02d}": 0.75 for i in range(3)}
    _patch(monkeypatch, _days_rows(shares))

    rc = cmd_metrics(_args(), config=SimpleNamespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert "verdict: INSUFFICIENT (3 measured days, need 7)" in out


def test_no_rows_prints_one_line(monkeypatch, capsys):
    _patch(monkeypatch, [])

    rc = cmd_metrics(_args(), config=SimpleNamespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == "feed-share: no channel_daily rows for gomoveshift"


def test_zero_total_day_is_omitted_from_the_count(monkeypatch, capsys):
    shares = {f"2026-09-{10 + i:02d}": 0.75 for i in range(7)}
    rows = _days_rows(shares)
    # A day with a row but zero total traffic-source views must never be
    # fabricated into a share -- it's simply absent, same as
    # `shorts_feed_share` already guarantees.
    rows.append(_row("2026-09-09", "SHORTS", 0))
    _patch(monkeypatch, rows)

    rc = cmd_metrics(_args(), config=SimpleNamespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert "2026-09-09" not in out
    assert "verdict: MET (7/7 days >= 0.7)" in out


def test_days_and_threshold_are_honoured(monkeypatch, capsys):
    shares = {f"2026-09-{10 + i:02d}": 0.55 for i in range(3)}
    _patch(monkeypatch, _days_rows(shares))

    rc = cmd_metrics(_args(days=3, threshold=0.50), config=SimpleNamespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert "feed-share: gomoveshift last 3 measured days (threshold 0.5)" in out
    assert "verdict: MET (3/3 days >= 0.5)" in out
