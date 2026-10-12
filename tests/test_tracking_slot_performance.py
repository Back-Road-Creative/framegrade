"""Tests for the slot-keyed performance aggregate store."""

from __future__ import annotations

import json
import sqlite3

import pytest

from gomoveshift.tracking.slot_performance import (
    MetricRow,
    slot_performance_trend,
    slots_ranked,
    upsert_from_metrics,
)


def _sidecar(tmp_path, project_name, entries):
    path = tmp_path / f"{project_name}_publish_plan.json"
    path.write_text(json.dumps({"shorts": entries}), encoding="utf-8")
    return path


def _entry(short_index, scheduled_at_utc, display_timezone, sampled_regions):
    return {
        "short_index": short_index,
        "schedule": {"scheduled_at_utc": scheduled_at_utc, "display_timezone": display_timezone},
        "sampled_regions": sampled_regions,
    }


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    yield connection
    connection.close()


def test_upsert_derives_slot_and_averages_repeats(tmp_path, conn):
    # 2026-09-01 is a Tuesday; 18:00 UTC -> 14:00 EDT (America/New_York).
    sidecar = _sidecar(
        tmp_path,
        "proj",
        [
            _entry(1, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"]),
            _entry(2, "2026-09-08T18:00:00+00:00", "America/New_York", ["Portfable"]),
        ],
    )
    rows = [
        MetricRow("proj", 1, "youtube", 100, 0.10),
        MetricRow("proj", 2, "youtube", 300, 0.20),
    ]

    folded = upsert_from_metrics(conn, rows, sidecar_lookup=lambda project: sidecar)

    assert folded == 2
    (slot,) = slots_ranked(conn, region="Portfable", platform="youtube", min_n=1)
    assert (slot.region, slot.platform, slot.weekday, slot.hour_local) == ("Portfable", "youtube", 1, 14)
    assert slot.n == 2
    assert slot.mean_views == pytest.approx(200.0)
    assert slot.mean_engagement == pytest.approx(0.15)


@pytest.mark.parametrize(
    "sidecar_lookup_factory,short_index",
    [
        (lambda sidecar: lambda project: None, 1),  # missing sidecar
        (lambda sidecar: lambda project: sidecar, 99),  # short not in sidecar
    ],
)
def test_upsert_skips_unresolvable_rows(tmp_path, conn, sidecar_lookup_factory, short_index):
    sidecar = _sidecar(
        tmp_path, "proj", [_entry(1, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"])]
    )
    rows = [MetricRow("proj", short_index, "youtube", 999, 0.5)]

    folded = upsert_from_metrics(conn, rows, sidecar_lookup=sidecar_lookup_factory(sidecar))

    assert folded == 0
    assert slots_ranked(conn, min_n=1) == []


def test_slots_ranked_min_n_and_ordering(tmp_path, conn):
    sidecar = _sidecar(
        tmp_path,
        "proj",
        [
            _entry(1, "2026-09-01T14:00:00+00:00", "America/New_York", ["Portfable"]),
            _entry(2, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"]),
        ],
    )
    rows = [
        MetricRow("proj", 1, "youtube", 50, 0.1),
        MetricRow("proj", 2, "youtube", 500, 0.1),
    ]
    upsert_from_metrics(conn, rows, sidecar_lookup=lambda project: sidecar)

    assert slots_ranked(conn, min_n=2) == []
    ranked = slots_ranked(conn, min_n=1)
    assert [r.hour_local for r in ranked] == [14, 10]


def test_refold_of_the_same_short_does_not_inflate_n(tmp_path, conn):
    """Replaying a collection for the same short must not add a sample."""
    sidecar = _sidecar(
        tmp_path,
        "proj",
        [_entry(1, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"])],
    )
    row = MetricRow("proj", 1, "youtube", 100, 0.10)

    folded_first = upsert_from_metrics(conn, [row], sidecar_lookup=lambda project: sidecar)
    folded_second = upsert_from_metrics(conn, [row], sidecar_lookup=lambda project: sidecar)

    assert folded_first == 1
    assert folded_second == 1
    (slot,) = slots_ranked(conn, region="Portfable", platform="youtube", min_n=1)
    assert slot.n == 1
    assert slot.mean_views == pytest.approx(100.0)
    assert slot.mean_engagement == pytest.approx(0.10)


def test_refold_with_corrected_metrics_replaces_not_adds(tmp_path, conn):
    """A recollection with corrected values replaces the sample, not adds."""
    sidecar = _sidecar(
        tmp_path,
        "proj",
        [_entry(1, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"])],
    )

    upsert_from_metrics(
        conn,
        [MetricRow("proj", 1, "youtube", 100, 0.10)],
        sidecar_lookup=lambda project: sidecar,
    )
    upsert_from_metrics(
        conn,
        [MetricRow("proj", 1, "youtube", 150, 0.20)],
        sidecar_lookup=lambda project: sidecar,
    )

    (slot,) = slots_ranked(conn, region="Portfable", platform="youtube", min_n=1)
    assert slot.n == 1
    assert slot.mean_views == pytest.approx(150.0)
    assert slot.mean_engagement == pytest.approx(0.20)


def test_refold_with_unknown_metric_keeps_earlier_observation(tmp_path, conn):
    """A later collection reporting None must not erase an earlier observation."""
    sidecar = _sidecar(
        tmp_path,
        "proj",
        [_entry(1, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"])],
    )

    upsert_from_metrics(
        conn,
        [MetricRow("proj", 1, "youtube", 100, 0.10)],
        sidecar_lookup=lambda project: sidecar,
    )
    folded_second = upsert_from_metrics(
        conn,
        [MetricRow("proj", 1, "youtube", None, None)],
        sidecar_lookup=lambda project: sidecar,
    )

    assert folded_second == 0
    (slot,) = slots_ranked(conn, region="Portfable", platform="youtube", min_n=1)
    assert slot.n == 1
    assert slot.mean_views == pytest.approx(100.0)
    assert slot.mean_engagement == pytest.approx(0.10)


def test_upsert_skips_unknown_measurements_without_voiding_the_batch(tmp_path, conn):
    """One row with an unreported metric must not discard the whole fold.

    ``PlatformMetrics.engagement_rate`` is ``None`` whenever a platform
    reports no shares, which is routine for YouTube shorts.
    """
    sidecar = _sidecar(
        tmp_path,
        "proj",
        [
            _entry(1, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"]),
            _entry(2, "2026-09-08T18:00:00+00:00", "America/New_York", ["Portfable"]),
            _entry(3, "2026-09-15T18:00:00+00:00", "America/New_York", ["Portfable"]),
            _entry(4, "2026-09-22T18:00:00+00:00", "America/New_York", ["Portfable"]),
        ],
    )
    rows = [
        MetricRow("proj", 1, "youtube", 100, 0.10),
        MetricRow("proj", 2, "youtube", 300, None),  # no shares reported
        MetricRow("proj", 3, "youtube", None, 0.90),  # views unavailable
        MetricRow("proj", 4, "youtube", 300, 0.20),
    ]

    folded = upsert_from_metrics(conn, rows, sidecar_lookup=lambda project: sidecar)

    assert folded == 2
    (slot,) = slots_ranked(conn, region="Portfable", platform="youtube", min_n=1)
    assert slot.n == 2
    # An unknown measurement is never folded in as a zero.
    assert slot.mean_views == pytest.approx(200.0)
    assert slot.mean_engagement == pytest.approx(0.15)


def test_fold_appends_a_history_snapshot_never_rebuilds_it(tmp_path, conn):
    """Each fold APPENDS to ``slot_performance_history`` -- unlike
    ``slot_performance`` itself, a second fold must never overwrite or
    delete the first snapshot."""
    sidecar = _sidecar(
        tmp_path,
        "proj",
        [_entry(1, "2026-09-01T18:00:00+00:00", "America/New_York", ["Portfable"])],
    )

    upsert_from_metrics(
        conn,
        [MetricRow("proj", 1, "youtube", 100, 0.10)],
        sidecar_lookup=lambda project: sidecar,
        folded_at="2026-09-01T00:00:00+00:00",
    )
    upsert_from_metrics(
        conn,
        [MetricRow("proj", 1, "youtube", 300, 0.30)],
        sidecar_lookup=lambda project: sidecar,
        folded_at="2026-09-08T00:00:00+00:00",
    )

    trend = slot_performance_trend(conn, "Portfable", "youtube", weekday=1, hour_local=14, weeks=4)
    assert [snap["folded_at"] for snap in trend] == [
        "2026-09-01T00:00:00+00:00",
        "2026-09-08T00:00:00+00:00",
    ]
    # First snapshot is untouched by the second, corrected-metric fold.
    assert trend[0]["n"] == 1
    assert trend[0]["mean_views"] == pytest.approx(100.0)
    assert trend[1]["mean_views"] == pytest.approx(300.0)
