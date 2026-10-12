"""Platform decline: a halved day-7 series is reported, a noisy flat one and
thin data are not, an empty store exits 0 saying so."""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.cli.handlers.platform_decline import cmd_platform_decline
from gomoveshift.tracking.decline import DeclineFinding, detect_platform_decline, platform_series
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)

PLATFORM = "facebook"
HALVED = [1400, 1470, 1330, 1400, 1470, 1330, 1400] * 2 + [700, 770, 630, 700, 770, 630, 700] * 2
NOISY = [1400, 700, 1330, 770, 1260, 840, 1400] * 4
# Median falls 82% but both halves straddle the same two far-apart values: p lands well above 0.05.
NOISY_DROP = [7000] * 7 + [700] * 7 + [7000] * 3 + [700] * 11


def _short(index: int, day7: int) -> ShortRecord:
    """One short whose first seven of ten daily points sum to ``day7``."""
    n = day7 // 7
    daily = [DailyMetricPoint(day=f"2026-01-{d:02d}", views=n) for d in range(10)]
    return ShortRecord(
        index=index,
        posted_at=f"2026-01-{index + 1:02d}T12:00:00+00:00",
        # A synthetic exact instant, so the honest label is the one that
        # promises exactness. The pair is inseparable precisely because
        # THIS module reads `posted_at` as an age: an `operator_confirmed`
        # time hours late would make these series start earlier than they did.
        posted_at_source="platform_confirmed",
        platform_ids={PLATFORM: f"vid{index}"},
        metrics={PLATFORM: PlatformMetrics(views=n * 10, daily=daily)},
    )


def _records(values):
    return [TrackingRecord(project="proj", shorts=[_short(i, v) for i, v in enumerate(values)])]


def _detect(values):  # SimpleNamespace stands in for TrackingStore: no I/O, no network
    recs = _records(values)
    store = SimpleNamespace(load_all_records=lambda: recs)
    return detect_platform_decline(recs, PLATFORM, store=store)


def test_a_halved_day7_series_is_reported_with_a_real_p_value():
    finding = _detect(HALVED)
    assert isinstance(finding, DeclineFinding)
    assert finding.platform == PLATFORM and -55 < finding.delta_pct < -45
    assert finding.p_value <= 0.01 and finding.significance == "strong"
    line = finding.log_line()
    assert line.startswith("PLATFORM_DECLINE platform=facebook window=14 ")
    assert "metric=day7_views" in line and "velocity=" in line and "\n" not in line
    assert " p=" in line and "significance=strong" in line


def test_neither_a_noisy_flat_series_nor_thin_data_is_a_decline():
    assert _detect(NOISY) is None and _detect([1400] * 5 + [700] * 5) is None


def test_a_big_drop_that_misses_significance_is_not_reported():
    assert _detect(NOISY_DROP) is None


def _run_cli(monkeypatch, records, capsys):
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.platform_decline.TrackingStore",
        lambda *a: SimpleNamespace(load_all_records=lambda: records),
    )
    rc = cmd_platform_decline(SimpleNamespace(window=14, min_drop_pct=20.0), None)
    return rc, capsys.readouterr().out


def test_cli_on_an_empty_store_exits_zero_saying_so(monkeypatch, capsys):
    rc, out = _run_cli(monkeypatch, [], capsys)
    assert rc == 0 and "insufficient data" in out.lower() and "PLATFORM_DECLINE" not in out


def test_cli_prints_one_line_per_declining_platform(monkeypatch, capsys):
    rc, out = _run_cli(monkeypatch, _records(HALVED), capsys)
    lines = [ln for ln in out.splitlines() if ln.startswith("PLATFORM_DECLINE")]
    assert rc == 0 and len(lines) == 1 and "platform=facebook" in lines[0]


def _short_with_daily(index, daily_views, video_id="v", project=None):
    daily = [
        DailyMetricPoint(day=f"2026-01-{d:02d}", views=v) for d, v in enumerate(daily_views, 1)
    ]
    return ShortRecord(
        index=index,
        posted_at="2026-01-01T12:00:00+00:00",
        posted_at_source="platform_confirmed",
        platform_ids={PLATFORM: video_id},
        metrics={PLATFORM: PlatformMetrics(views=sum(v or 0 for v in daily_views), daily=daily)},
    )


def test_a_null_day_inside_the_day7_window_does_not_crash_the_series():
    """A gap in one day's collection is unreported, not a measured zero — the short
    is simply unqualified for a day-7 figure, same idiom as velocity's _sum_reported_views."""
    daily = [None, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    short = _short_with_daily(0, daily)
    records = [TrackingRecord(project="proj", shorts=[short])]
    assert platform_series(records, PLATFORM) == []


def test_a_null_day_does_not_disqualify_a_short_whose_day7_window_is_clean():
    daily = [100] * 7 + [None, 100, 100]  # the null day is day 8, outside DAY7_DAYS
    short = _short_with_daily(0, daily)
    records = [TrackingRecord(project="proj", shorts=[short])]
    points = platform_series(records, PLATFORM)
    assert len(points) == 1 and points[0][1] == 700


def test_the_same_video_tracked_under_two_projects_counts_once():
    """A backfill/rename can leave the same platform video_id tracked under two
    project files; the reader must not let one video vote twice in a window."""
    a = _short_with_daily(0, [200] * 10, video_id="vshared")
    b = _short_with_daily(0, [200] * 10, video_id="vshared")
    records = [
        TrackingRecord(project="proj", shorts=[a]),
        TrackingRecord(project="proj-backfill-copy", shorts=[b]),
    ]
    points = platform_series(records, PLATFORM)
    assert len(points) == 1


def test_distinct_videos_posted_at_the_identical_instant_both_count():
    """Same-timestamp is not the same signal as same-video: two real posts that
    happen to share a posted_at must both survive."""
    a = _short_with_daily(0, [200] * 10, video_id="v1")
    b = _short_with_daily(1, [300] * 10, video_id="v2")
    records = [TrackingRecord(project="proj", shorts=[a, b])]
    points = platform_series(records, PLATFORM)
    assert len(points) == 2


def test_out_of_order_records_come_back_oldest_first():
    """Already correct: platform_series sorts explicitly, so callers never see records
    in file/collection order — pinned so a future refactor can't drop the sort."""
    newer = ShortRecord(
        index=0,
        posted_at="2026-01-10T00:00:00+00:00",
        posted_at_source="platform_confirmed",
        platform_ids={PLATFORM: "v-new"},
        metrics={
            PLATFORM: PlatformMetrics(
                views=700, daily=[DailyMetricPoint(f"2026-01-{d:02d}", 100) for d in range(1, 11)]
            )
        },
    )
    older = ShortRecord(
        index=1,
        posted_at="2026-01-01T00:00:00+00:00",
        posted_at_source="platform_confirmed",
        platform_ids={PLATFORM: "v-old"},
        metrics={
            PLATFORM: PlatformMetrics(
                views=700, daily=[DailyMetricPoint(f"2026-02-{d:02d}", 100) for d in range(1, 11)]
            )
        },
    )
    records = [TrackingRecord(project="proj", shorts=[newer, older])]
    points = platform_series(records, PLATFORM)
    assert [vid for _, _, vid in points] == ["v-old", "v-new"]


def test_a_platform_absent_from_the_requested_series_is_never_mixed_in():
    """Already correct: a short with metrics only on another platform contributes
    nothing to this platform's series, even sharing a project with real rows."""
    other_platform_only = ShortRecord(
        index=0,
        posted_at="2026-01-01T12:00:00+00:00",
        posted_at_source="platform_confirmed",
        platform_ids={"instagram": "vig"},
        metrics={
            "instagram": PlatformMetrics(
                views=700, daily=[DailyMetricPoint(f"2026-01-{d:02d}", 100) for d in range(1, 11)]
            )
        },
    )
    records = [TrackingRecord(project="proj", shorts=[other_platform_only])]
    assert platform_series(records, PLATFORM) == []


def test_a_backfill_that_rewrites_history_is_reflected_on_the_next_read():
    """Already correct: the reader holds no verdict cache, so a corrected day-7
    value changes the very next computed series instead of a stale one lingering."""
    short = _short_with_daily(0, [100] * 10, video_id="v1")
    records = [TrackingRecord(project="proj", shorts=[short])]
    before = platform_series(records, PLATFORM)[0][1]
    short.metrics[PLATFORM].daily[2].views = 900  # a backfill corrects one day's value
    after = platform_series(records, PLATFORM)[0][1]
    assert before == 700 and after == 1500
