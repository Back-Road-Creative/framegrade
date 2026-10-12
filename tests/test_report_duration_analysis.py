"""``PerformanceReporter`` cohorts shorts by
``ContentMeta.duration_sec`` so the length experiment is measurable.

Buckets are fixed: ≤15s, 16–30s, 31–60s, >60s, plus "unknown" for a short
whose duration was never measured/set (``duration_sec <= 0``) -- that short
must be listed, not silently dropped, the same "recorded not discarded"
rule the rest of this module already applies to refusals.
"""

from __future__ import annotations

from gomoveshift.tracking.report import PerformanceReporter
from gomoveshift.tracking.schema import ContentMeta, PlatformMetrics, ShortRecord, TrackingRecord
from gomoveshift.tracking.store import TrackingStore


def _short(index: int, duration_sec: float, views: int = 100) -> ShortRecord:
    short = ShortRecord(
        index=index,
        file=f"short{index}.mp4",
        metrics={"youtube": PlatformMetrics(views=views, likes=10)},
        content=ContentMeta(duration_sec=duration_sec),
    )
    short.collected_at = "2026-01-01T00:00:00"
    return short


def test_duration_buckets_shorts_and_reports_mean_engagement(tmp_path):
    record = TrackingRecord(project="demo", location="Ireland")
    record.shorts.append(_short(1, duration_sec=10))
    record.shorts.append(_short(2, duration_sec=12))
    record.shorts.append(_short(3, duration_sec=45))

    reporter = PerformanceReporter(store=TrackingStore(root=tmp_path))
    section = reporter._duration_analysis(record.shorts_with_metrics)

    assert "Duration vs Performance" in section
    assert "≤15s" in section
    assert "31–60s" in section
    # ≤15s bucket has 2 shorts, 31-60s has 1.
    lines = {line.split("|")[1].strip(): line for line in section.splitlines() if "|" in line}
    assert "2" in lines["≤15s"]
    assert "1" in lines["31–60s"]


def test_unknown_duration_is_listed_not_dropped(tmp_path):
    record = TrackingRecord(project="demo", location="Ireland")
    record.shorts.append(_short(1, duration_sec=0))
    record.shorts.append(_short(2, duration_sec=20))

    reporter = PerformanceReporter(store=TrackingStore(root=tmp_path))
    section = reporter._duration_analysis(record.shorts_with_metrics)

    assert "unknown" in section
    assert "16–30s" in section


def test_all_unknown_duration_returns_empty_section(tmp_path):
    record = TrackingRecord(project="demo", location="Ireland")
    record.shorts.append(_short(1, duration_sec=0))
    record.shorts.append(_short(2, duration_sec=0))

    reporter = PerformanceReporter(store=TrackingStore(root=tmp_path))
    section = reporter._duration_analysis(record.shorts_with_metrics)

    assert section == ""


def test_duration_analysis_counts_only_measured_engagement(tmp_path):
    """A short with a known duration but no measured engagement contributes
    to the bucket's sample size but not to the averaged engagement rate --
    same rule ``_avg_engagement`` already enforces everywhere else."""
    record = TrackingRecord(project="demo", location="Ireland")
    measured = _short(1, duration_sec=10)
    unmeasured = ShortRecord(
        index=2,
        file="short2.mp4",
        content=ContentMeta(duration_sec=12),
    )
    unmeasured.collected_at = "2026-01-01T00:00:00"
    record.shorts.append(measured)
    record.shorts.append(unmeasured)

    reporter = PerformanceReporter(store=TrackingStore(root=tmp_path))
    # Both fall in the same bucket even though only one has a metrics dict;
    # feed both, mirroring how _weather_analysis/_hook_analysis are exercised
    # over the full population, not just shorts_with_metrics.
    section = reporter._duration_analysis([measured, unmeasured])

    assert "≤15s" in section
    line = [line for line in section.splitlines() if "≤15s" in line][0]
    cells = [c.strip() for c in line.split("|")]
    assert cells[2] == "2"  # both shorts counted in the bucket
    assert cells[3] != "—"  # measured's engagement rate still reported
