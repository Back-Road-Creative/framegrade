"""``metrics collect`` wires the bounded response into the nightly
run, shadow by default.

Only YouTube is classified -- there is no Facebook reel-views series in
``channel_daily`` yet, so Facebook must never appear in the decisions this
prints or writes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.cli import _build_parser
from gomoveshift.cli.handlers import metrics as metrics_handler
from gomoveshift.publishing.queue import get_platform_daily_cap, open_queue_db


def _config(storage_path) -> SimpleNamespace:
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(storage_path)))


def _rows():
    return [
        {
            "day": "2026-09-17",
            "dimension": "insightTrafficSourceType",
            "value": "SHORTS",
            "views": 80,
        },
        {
            "day": "2026-09-17",
            "dimension": "insightTrafficSourceType",
            "value": "OTHER",
            "views": 20,
        },
    ]


def test_parser_default_and_choices():
    ns = _build_parser().parse_args(["metrics", "collect", "proj"])
    assert ns.response_mode == "shadow"

    ns = _build_parser().parse_args(["metrics", "collect", "proj", "--response-mode", "live"])
    assert ns.response_mode == "live"

    try:
        _build_parser().parse_args(["metrics", "collect", "proj", "--response-mode", "bogus"])
    except SystemExit:
        pass
    else:  # pragma: no cover
        raise AssertionError("--response-mode must reject an unknown value")


def test_apply_bounded_response_writes_a_cap_row_and_prints_a_decision(tmp_path, capsys):
    config = _config(tmp_path)
    open_queue_db(tmp_path).close()

    with (
        patch.object(metrics_handler, "load_channel_daily_rows", return_value=_rows()),
        patch.object(
            metrics_handler, "load_brand_policy", return_value=SimpleNamespace(daily_ladder=None)
        ),
    ):
        metrics_handler._apply_bounded_response(config, "gomoveshift", "shadow")

    out = capsys.readouterr().out
    assert (
        "Bounded response (shadow): youtube state=HEALTHY cap=3 healthy_days=0 auto_pause=False"
        in out
    )

    conn = open_queue_db(tmp_path)
    try:
        cap_row = get_platform_daily_cap(conn, "youtube")
    finally:
        conn.close()
    assert cap_row["mode"] == "shadow"
    assert cap_row["cap"] == 3


def test_apply_bounded_response_passes_live_mode_through(tmp_path, capsys):
    config = _config(tmp_path)
    open_queue_db(tmp_path).close()

    with (
        patch.object(metrics_handler, "load_channel_daily_rows", return_value=_rows()),
        patch.object(
            metrics_handler, "load_brand_policy", return_value=SimpleNamespace(daily_ladder=None)
        ),
    ):
        metrics_handler._apply_bounded_response(config, "gomoveshift", "live")

    assert "Bounded response (live):" in capsys.readouterr().out
    conn = open_queue_db(tmp_path)
    try:
        cap_row = get_platform_daily_cap(conn, "youtube")
    finally:
        conn.close()
    assert cap_row["mode"] == "live"


def test_apply_bounded_response_short_circuits_on_no_rows(tmp_path, capsys):
    config = _config(tmp_path)

    with patch.object(metrics_handler, "load_channel_daily_rows", return_value=[]):
        metrics_handler._apply_bounded_response(config, "gomoveshift", "shadow")

    assert "Bounded response: no channel-daily rows for gomoveshift" in capsys.readouterr().out
    # No DB was even created -- nothing to open for an empty pull.
    assert not (tmp_path / "social_queue.sqlite3").exists()


def test_apply_bounded_response_never_classifies_facebook(tmp_path, capsys):
    config = _config(tmp_path)
    open_queue_db(tmp_path).close()

    with (
        patch.object(metrics_handler, "load_channel_daily_rows", return_value=_rows()),
        patch.object(
            metrics_handler, "load_brand_policy", return_value=SimpleNamespace(daily_ladder=None)
        ),
    ):
        metrics_handler._apply_bounded_response(config, "gomoveshift", "shadow")

    out = capsys.readouterr().out
    assert "facebook" not in out
    conn = open_queue_db(tmp_path)
    try:
        assert get_platform_daily_cap(conn, "facebook") is None
    finally:
        conn.close()


def test_apply_bounded_response_swallows_an_exception(tmp_path, capsys):
    config = _config(tmp_path)

    with patch.object(metrics_handler, "load_channel_daily_rows", side_effect=RuntimeError("boom")):
        # Must not raise -- collection is already persisted by this point.
        metrics_handler._apply_bounded_response(config, "gomoveshift", "shadow")

    assert "Bounded response" not in capsys.readouterr().out


def test_cmd_metrics_collect_calls_apply_bounded_response(tmp_path, capsys):
    record = SimpleNamespace(project="p", shorts=[])
    args = SimpleNamespace(metrics_command="collect", project="p", response_mode="shadow")
    config = _config(tmp_path)

    with (
        patch.object(metrics_handler, "TrackingStore") as mock_store_cls,
        patch.object(metrics_handler, "MetricsCollector") as mock_collector_cls,
        patch.object(metrics_handler, "_apply_bounded_response") as mock_apply,
    ):
        mock_store_cls.return_value.exists.return_value = True
        mock_store_cls.return_value.load.return_value = record
        mock_collector_cls.return_value.collect_all.return_value = 0
        rc = metrics_handler.cmd_metrics(args, config)

    assert rc == 0
    mock_apply.assert_called_once()
    call_args = mock_apply.call_args.args
    assert call_args[0] is config
    assert call_args[2] == "shadow"
