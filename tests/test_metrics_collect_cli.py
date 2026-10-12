"""``gomoveshift metrics collect <project>`` subcommand.

The ``tracking/`` module had no call-sites until the CLI surface was added. These pin that
CLI surface: ``metrics collect`` builds a ``MetricsCollector`` +
``TrackingStore``, runs ``collect_all`` for the project, and persists.
Collectors/store are mocked so no network access happens.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.cli import _build_parser
from gomoveshift.cli.handlers import metrics as metrics_handler
from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_RECONCILED as PLATFORM,
    enqueue,
    open_queue_db,
    transition,
)
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord, TrackingRecord
from gomoveshift.tracking.slot_performance import slots_ranked


def _args(metrics_command="collect", project="20250101Example") -> SimpleNamespace:
    return SimpleNamespace(metrics_command=metrics_command, project=project)


def _config(storage_path) -> SimpleNamespace:
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(storage_path)))


def test_metrics_collect_parses() -> None:
    ns = _build_parser().parse_args(["metrics", "collect", "20250101Example"])
    assert (ns.command, ns.metrics_command, ns.project) == (
        "metrics",
        "collect",
        "20250101Example",
    )


def test_collect_dispatches_to_collector_and_persists(tmp_path, capsys) -> None:
    record = SimpleNamespace(project="p", shorts=[1, 2, 3])
    with (
        patch.object(metrics_handler, "TrackingStore") as mock_store_cls,
        patch.object(metrics_handler, "MetricsCollector") as mock_collector_cls,
    ):
        store = mock_store_cls.return_value
        store.exists.return_value = True
        store.load.return_value = record
        collector = mock_collector_cls.return_value
        collector.collect_all.return_value = 2
        rc = metrics_handler.cmd_metrics(_args(), _config(tmp_path))
    assert rc == 0
    collector.collect_all.assert_called_once_with(record)
    store.save.assert_called_once_with(record)
    assert "2" in capsys.readouterr().out


def test_collect_missing_project_returns_one(tmp_path, capsys) -> None:
    with (
        patch.object(metrics_handler, "TrackingStore") as mock_store_cls,
        patch.object(metrics_handler, "MetricsCollector"),
    ):
        mock_store_cls.return_value.exists.return_value = False
        rc = metrics_handler.cmd_metrics(_args(project="no-such"), _config(tmp_path))
    assert rc == 1
    assert "no-such" in capsys.readouterr().err


def test_unknown_subcommand_returns_one(capsys) -> None:
    rc = metrics_handler.cmd_metrics(_args(metrics_command=None), MagicMock())
    assert rc == 1
    assert "Usage" in capsys.readouterr().out


def test_backfill_ids_parses() -> None:
    ns = _build_parser().parse_args(["metrics", "backfill-ids"])
    assert (ns.command, ns.metrics_command) == ("metrics", "backfill-ids")


def test_backfill_ids_registers_every_published_row(tmp_path, capsys) -> None:
    conn = open_queue_db(tmp_path)
    d1 = enqueue(conn, "proj", 1, "instagram", "hash-a")
    transition(
        conn, d1, "queued", "published", platform_post_id="ig_1", published_at_source=PLATFORM
    )
    d2 = enqueue(conn, "proj", 2, "youtube", "hash-b")
    transition(
        conn,
        d2,
        "queued",
        "published",
        platform_upload_id="yt_2",
        platform_post_id="pl_2",
        published_at_source=PLATFORM,
    )
    # A non-published row must be skipped.
    enqueue(conn, "proj", 3, "tiktok", "hash-c")
    conn.close()

    registered = []
    with patch.object(metrics_handler, "TrackingStore") as mock_store_cls:
        store = mock_store_cls.return_value
        store.update_platform_id.side_effect = lambda *a, **kw: registered.append((a, kw))
        rc = metrics_handler.cmd_metrics(_args(metrics_command="backfill-ids"), _config(tmp_path))
    assert rc == 0
    assert sorted(a for a, _ in registered) == [
        ("proj", 1, "instagram", "ig_1"),
        ("proj", 2, "youtube", "yt_2"),
    ]
    # The posting time crosses with its provenance. ``transition``
    # derives ``published_at`` itself and requires the source, so every row it
    # wrote carries both -- the backfill must not drop half of the pair.
    for _, kwargs in registered:
        assert kwargs["posted_at_source"] == PLATFORM
        assert kwargs["posted_at"]
    assert "2" in capsys.readouterr().out


def test_backfill_ids_is_idempotent_on_a_second_run(tmp_path) -> None:
    conn = open_queue_db(tmp_path)
    d1 = enqueue(conn, "proj", 1, "instagram", "hash-a")
    transition(
        conn, d1, "queued", "published", platform_post_id="ig_1", published_at_source=PLATFORM
    )
    conn.close()

    with patch.object(metrics_handler, "TrackingStore") as mock_store_cls:
        store = mock_store_cls.return_value
        rc1 = metrics_handler.cmd_metrics(_args(metrics_command="backfill-ids"), _config(tmp_path))
        rc2 = metrics_handler.cmd_metrics(_args(metrics_command="backfill-ids"), _config(tmp_path))
    assert (rc1, rc2) == (0, 0)
    assert store.update_platform_id.call_count == 2


# ── Collect folds into slot_performance ─────────────────────────────


def _tracking_record(project="proj", index=1):
    return TrackingRecord(
        project=project,
        shorts=[
            ShortRecord(
                index=index,
                collected_at="2026-09-09T00:00:00+00:00",
                metrics={"youtube": PlatformMetrics(views=100, likes=6, comments=2, shares=2)},
            )
        ],
    )


def _write_sidecar(tmp_path, project="proj", short_index=1):
    path = tmp_path / f"{project}_publish_plan.json"
    path.write_text(
        json.dumps(
            {
                "shorts": [
                    {
                        "short_index": short_index,
                        "schedule": {
                            # 2026-09-01 is a Tuesday; 18:00 UTC -> 14:00 EDT.
                            "scheduled_at_utc": "2026-09-01T18:00:00+00:00",
                            "display_timezone": "America/New_York",
                        },
                        "sampled_regions": ["Portfable"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_collect(tmp_path, record, project="proj"):
    with (
        patch.object(metrics_handler, "TrackingStore") as mock_store_cls,
        patch.object(metrics_handler, "MetricsCollector") as mock_collector_cls,
    ):
        store = mock_store_cls.return_value
        store.exists.return_value = True
        store.load.return_value = record
        mock_collector_cls.return_value.collect_all.return_value = 1
        return metrics_handler.cmd_metrics(_args(project=project), _config(tmp_path))


def test_collect_folds_collected_metrics_into_slot_performance(tmp_path, capsys) -> None:
    sidecar = _write_sidecar(tmp_path)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "hash-a", sidecar_path=str(sidecar))
    conn.close()

    rc = _run_collect(tmp_path, _tracking_record())

    assert rc == 0
    out = capsys.readouterr().out
    # The cron script parses this line verbatim — it must survive the fold.
    assert "Collected metrics for 1 short(s) in project proj" in out
    assert "olded 1" in out

    conn = open_queue_db(tmp_path)
    try:
        (slot,) = slots_ranked(conn, min_n=1)
    finally:
        conn.close()
    assert (slot.region, slot.platform, slot.weekday, slot.hour_local) == ("Portfable", "youtube", 1, 14)
    assert (slot.n, slot.mean_views) == (1, 100.0)


def test_collect_names_a_swallowed_fold_failure_in_its_own_output(tmp_path, capsys) -> None:
    sidecar = _write_sidecar(tmp_path)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "hash-a", sidecar_path=str(sidecar))
    conn.close()

    with patch.object(metrics_handler, "upsert_from_metrics", side_effect=RuntimeError("boom")):
        rc = _run_collect(tmp_path, _tracking_record())

    # The swallow itself is deliberate (a fold error must never read as a
    # lost collection) -- the exit code stays 0.
    assert rc == 0
    out = capsys.readouterr().out
    assert "Collected metrics for 1 short(s) in project proj" in out
    assert "Folded 0 of 1 metric row(s) into slot_performance" in out
    assert "fold skipped: RuntimeError: boom" in out


def test_collect_without_a_sidecar_still_collects_and_folds_nothing(tmp_path, capsys) -> None:
    rc = _run_collect(tmp_path, _tracking_record())

    assert rc == 0
    out = capsys.readouterr().out
    assert "Collected metrics for 1 short(s) in project proj" in out
    assert "olded 0" in out

    conn = open_queue_db(tmp_path)
    try:
        assert slots_ranked(conn, min_n=1) == []
    finally:
        conn.close()


# ── Collect refreshes audience_online ────────────────────────────────


class _StubPlatformCollector:
    """Minimal stand-in exposing only ``fetch_audience_online``."""

    def __init__(self, hour_counts):
        self._hour_counts = hour_counts

    def fetch_audience_online(self):
        return self._hour_counts


def _run_collect_with_platform_collectors(tmp_path, platform_collectors, project="proj"):
    record = _tracking_record(project=project)
    with (
        patch.object(metrics_handler, "TrackingStore") as mock_store_cls,
        patch.object(metrics_handler, "MetricsCollector") as mock_collector_cls,
    ):
        store = mock_store_cls.return_value
        store.exists.return_value = True
        store.load.return_value = record
        collector = mock_collector_cls.return_value
        collector.collect_all.return_value = 1
        collector._collectors = platform_collectors
        return metrics_handler.cmd_metrics(_args(project=project), _config(tmp_path))


def test_collect_refreshes_audience_online_from_platform_histograms(tmp_path, capsys) -> None:
    rc = _run_collect_with_platform_collectors(
        tmp_path,
        {
            "instagram": _StubPlatformCollector({14: 20, 18: 35}),
            "youtube": SimpleNamespace(),  # no fetch_audience_online -- skipped, not a fault
        },
    )
    assert rc == 0

    conn = open_queue_db(tmp_path)
    try:
        rows = conn.execute(
            "SELECT platform, hour_local, n_online FROM audience_online ORDER BY hour_local"
        ).fetchall()
    finally:
        conn.close()
    assert [tuple(r) for r in rows] == [("instagram", 14, 20), ("instagram", 18, 35)]
    assert "Refreshed audience-online: 2 hour(s)" in capsys.readouterr().out


def test_collect_reports_empty_audience_online_honestly(tmp_path, capsys) -> None:
    rc = _run_collect_with_platform_collectors(
        tmp_path,
        {"instagram": _StubPlatformCollector({})},
    )
    assert rc == 0

    assert "Refreshed audience-online: 0 hour(s)" in capsys.readouterr().out


# ── The run reports what each platform said, refusals included ───────


def test_collect_reports_a_measured_interval_and_a_refusal(tmp_path, capsys) -> None:
    """``collect_all`` is mocked here (no network), so this test populates
    ``observation_log`` the way a real run would, and pins that the CLI
    reports from it: a measured trend for youtube, and instagram's own
    refusal reason -- never an undifferentiated silence for either."""
    from gomoveshift.tracking.observation import MetricObservation, ObservationStatus

    record = _tracking_record()
    short = record.shorts[0]
    short.observation_log["youtube"] = [
        MetricObservation(
            platform="youtube",
            status=ObservationStatus.MEASURED,
            post_id="yt_1",
            post_format="short",
            fetched_at="2026-09-01T00:00:00+00:00",
            cumulative=PlatformMetrics(views=100),
        ).to_dict(),
        MetricObservation(
            platform="youtube",
            status=ObservationStatus.MEASURED,
            post_id="yt_1",
            post_format="short",
            fetched_at="2026-09-02T00:00:00+00:00",
            cumulative=PlatformMetrics(views=150),
        ).to_dict(),
    ]
    short.observation_log["instagram"] = [
        MetricObservation(
            platform="instagram",
            status=ObservationStatus.PERMISSION_DENIED,
            post_id="ig_1",
            post_format="reel",
            fetched_at="2026-09-01T00:00:00+00:00",
            detail="token is missing the read_insights scope",
        ).to_dict(),
    ]

    rc = _run_collect(tmp_path, record)
    assert rc == 0
    out = capsys.readouterr().out

    assert "youtube: measured" in out
    assert "50" in out  # the derived view interval
    assert "instagram: permission_denied" in out
    assert "read_insights" in out
